import numpy as np
import pytest

from adaptive_reservoir import (
    AdaptiveReservoir,
    ReservoirConfig,
    SNAPSHOT_API_STAGE,
    SNAPSHOT_SCHEMA_VERSION,
)


def test_snapshot_contains_math_state_only() -> None:
    model = AdaptiveReservoir(_config())
    model.step([0.5, -0.25])

    snapshot = model.snapshot()
    state = snapshot["state"]