from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import w3_cover


def _args(mode: str, *, checkpoint: Path | None = None, evidence_root: Path | None = None) -> Namespace:
    return Namespace(
        mode=mode,
        w2_root=Path("w2"),
        bridge_artifact=Path("bridge.pt"),
        audit_manifest=Path("audit.json"),
        output_root=Path("output"),
        protocol_dir=Path("protocol"),
        validator=None,
        fixture=False,
        per_rank_batch_size=16,
        batch_probe_receipt=None,
        checkpoint=checkpoint,
        evidence_root=evidence_root,
    )


def test_public_modes_have_distinct_responsibilities(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(w3_cover, "run_train_mode", lambda **_: calls.append("train") or {"mode": "train"})
    monkeypatch.setattr(w3_cover, "run_evaluate_mode", lambda **_: calls.append("evaluate") or {"mode": "evaluate"})
    monkeypatch.setattr(w3_cover, "run_package_mode", lambda **_: calls.append("package") or {"mode": "package"})
    monkeypatch.setattr(w3_cover, "run_canonical_acceptance", lambda **_: calls.append("accept") or {"mode": "accept"})

    assert w3_cover.dispatch(_args("train")) == {"mode": "train"}
    assert w3_cover.dispatch(_args("evaluate", checkpoint=Path("best.pt"))) == {"mode": "evaluate"}
    assert w3_cover.dispatch(_args("package", evidence_root=Path("evidence"))) == {"mode": "package"}
    assert w3_cover.dispatch(_args("accept")) == {"mode": "accept"}
    assert calls == ["train", "evaluate", "package", "accept"]


def test_evaluate_and_package_require_explicit_inputs():
    with pytest.raises(ValueError, match="checkpoint"):
        w3_cover.dispatch(_args("evaluate"))
    with pytest.raises(ValueError, match="evidence-root"):
        w3_cover.dispatch(_args("package"))
