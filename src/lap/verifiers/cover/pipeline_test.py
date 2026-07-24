from __future__ import annotations

import json

from lap.verifiers.cover.pipeline import run_fixture_end_to_end


def test_fixture_acceptance_publishes_complete_resumable_and_deployment_bundle(tmp_path):
    root = tmp_path / "w3-output"
    report = run_fixture_end_to_end(output_root=root)
    assert report["schema"] == "osx_cover_w3_fixture_acceptance_v1"
    assert (root / "latest.pt").is_file()
    assert (root / "deployment" / "FIXTURE_ONLY_W3_BUNDLE").is_file()
    payload = json.loads((root / "acceptance.json").read_text())
    assert payload["content_hash"]
