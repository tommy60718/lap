"""Single-active-session WebSocket policy server for pi05_cover."""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from typing import Any
from typing import Callable

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)

SessionEndCallback = Callable[[], None]


class SingleSessionWebsocketPolicyServer:
    """OpenPI-compatible WebSocket server that admits one robot client.

    A second concurrent connection is rejected without reading observations or
    calling ``infer``. When the active client disconnects, ``on_session_end``
    runs so cover history can be discarded before another client is accepted.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        on_session_end: SessionEndCallback | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._on_session_end = on_session_end
        self._active_session = False
        self._session_lock = asyncio.Lock()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        async with self._session_lock:
            if self._active_session:
                logger.warning(
                    "Rejecting second concurrent robot client from %s",
                    websocket.remote_address,
                )
                await websocket.close(
                    code=websockets.frames.CloseCode.TRY_AGAIN_LATER,
                    reason="pi05_cover admits exactly one active robot client",
                )
                return
            self._active_session = True

        logger.info("Connection from %s opened (single-session owner)", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        try:
            await websocket.send(packer.pack(self._metadata))
            prev_total_time = None
            while True:
                try:
                    start_time = time.monotonic()
                    obs = msgpack_numpy.unpackb(await websocket.recv())

                    infer_time = time.monotonic()
                    action = self._policy.infer(obs)
                    infer_time = time.monotonic() - infer_time

                    action["server_timing"] = {"infer_ms": infer_time * 1000}
                    if prev_total_time is not None:
                        action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                    await websocket.send(packer.pack(action))
                    prev_total_time = time.monotonic() - start_time
                except websockets.ConnectionClosed:
                    logger.info("Connection from %s closed", websocket.remote_address)
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            try:
                if self._on_session_end is not None:
                    self._on_session_end()
            finally:
                async with self._session_lock:
                    self._active_session = False


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
