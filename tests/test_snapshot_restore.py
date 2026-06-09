import copy

import numpy as np
import pytest

from adaptive_reservoir import (
    AdaptiveReservoir,
    ReservoirConfig,
    SNAPSHOT_API_STAGE,
    SNAPSHOT_SCHEMA_VERSION,
    restore_state,
    snapshot_state,
    validate_runtime_snapshot,
)


def test_snapshot_contains_schema_version_and_math_state_only() -> None:
    model = AdaptiveReservoir(_config())
   