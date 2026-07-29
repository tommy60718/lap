from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import w3_cover


def _args(
    mode: str,
    *,
    checkpoint: Path | None = None,
    evidence_root: Path | None = None,
    fixture: bool = False,
) -> Namespace:
    return Namespace(
        mode=mode,
        w2_root=Path("w2"),
        bridge_artifact=Path("bridge.pt"),
        audit_manifest=Path("audit.json"),
        output_root=Path("output"),
        protocol_dir=Path("protocol"),
        validator=None,
        fixture=fixture,
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


def test_accept_fixture_does_not_alias_fixture_acceptance(monkeypatch):
    monkeypatch.setattr(
        w3_cover,
        "run_fixture_end_to_end",
        lambda **_: (_ for _ in ()).throw(AssertionError("accept must not alias fixture execution")),
    )
    monkeypatch.setattr(
        w3_cover,
        "run_canonical_acceptance",
        lambda **_: (_ for _ in ()).throw(AssertionError("canonical accept must not run in this negative test")),
    )

    with pytest.raises(ValueError, match=r"accept.*fixture|fixture.*accept|alias"):
        w3_cover.dispatch(_args("accept", fixture=True))


def test_fixture_acceptance_mode_remains_distinct(monkeypatch):
    monkeypatch.setattr(w3_cover, "run_fixture_end_to_end", lambda **_: {"mode": "fixture-acceptance"})
    assert w3_cover.dispatch(_args("fixture-acceptance")) == {"mode": "fixture-acceptance"}


def test_evaluate_and_package_require_explicit_inputs():
    with pytest.raises(ValueError, match="checkpoint"):
        w3_cover.dispatch(_args("evaluate"))
    with pytest.raises(ValueError, match="evidence-root"):
        w3_cover.dispatch(_args("package"))
