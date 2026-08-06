"""Load the accepted W3 deployment bundle as a CoverScorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lap.verifiers.cover.batch_probe import _resolve_pinned_snapshot
from lap.verifiers.cover.canonical import require_published_deployment_payloads
from lap.verifiers.cover.canonical import validate_package_identity
from lap.verifiers.cover.checkpoint import DeploymentScorer
from lap.verifiers.cover.checkpoint import load_deployment_bundle
from lap.verifiers.cover.model import OpenClipSigLIP2Backbone
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel

_VERIFIER_CONFIG_FIELDS = frozenset(VerifierConfig.__dataclass_fields__)


def _verifier_config_from_metadata(model_config: dict[str, Any]) -> VerifierConfig:
    kwargs = {key: value for key, value in model_config.items() if key in _VERIFIER_CONFIG_FIELDS}
    return VerifierConfig(**kwargs)


def load_accepted_w3_deployment_scorer(
    package_root: Path,
    *,
    expected_normalization_hash: str,
) -> DeploymentScorer:
    """Validate PACKAGE_IDENTITY then load the sealed deployment scorer."""

    package_root = Path(package_root)
    validate_package_identity(package_root)
    deployment_root = package_root / "deployment"
    require_published_deployment_payloads(deployment_root)

    metadata = json.loads((deployment_root / "metadata.json").read_text(encoding="utf-8"))
    config = _verifier_config_from_metadata(dict(metadata["model_config"]))
    snapshot = _resolve_pinned_snapshot()

    def model_factory() -> VerifierModel:
        backbone = OpenClipSigLIP2Backbone(model_name=f"local-dir:{snapshot}")
        return VerifierModel(config, backbone)

    backbone = OpenClipSigLIP2Backbone(model_name=f"local-dir:{snapshot}")
    return load_deployment_bundle(
        deployment_root,
        model_factory=model_factory,
        expected_normalization_hash=expected_normalization_hash,
        preprocessing=backbone.preprocess,
    )
