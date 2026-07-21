"""Stable CoVer runtime fault and fallback contracts."""

from __future__ import annotations


class FallbackReason:
    MISSING_COVER_METADATA = "missing_cover_metadata"
    STATE_DISCONTINUITY = "state_discontinuity"
    INSTRUCTION_CHANGED = "instruction_changed"
    MISSING_OR_ZERO_WRIST_VIEW = "missing_or_zero_wrist_view"
    INCOMPATIBLE_VERIFIER_ASSETS = "incompatible_verifier_assets"
    VERIFIER_UNAVAILABLE = "verifier_unavailable"
    HISTORY_INVALID = "history_invalid"
    SCORES_INVALID = "scores_invalid"
    DIAGNOSTIC_SERIALIZATION_FAILED = "diagnostic_serialization_failed"


class CoverStateError(ValueError):
    """Episode/timestep/instruction/metadata continuity fault."""

    def __init__(self, message: str, *, fallback_reason: str) -> None:
        super().__init__(message)
        self.fallback_reason = fallback_reason


class CoverVerifierError(ValueError):
    """Post-candidate verifier-side fault."""

    def __init__(
        self,
        message: str,
        *,
        fallback_reason: str,
        store_pending: bool = False,
    ) -> None:
        super().__init__(message)
        self.fallback_reason = fallback_reason
        self.store_pending = store_pending
