"""Pipeline business exceptions."""
from __future__ import annotations


class PipelineError(Exception):
    """Raised when a pipeline stage fails or violates its contract.

    The `stage` field carries the offending stage name so callers can
    branch on it without parsing the message.
    """

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message
