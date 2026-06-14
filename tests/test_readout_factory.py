from __future__ import annotations

import pytest

from adaptive_reservoir import ReadoutConfig
from adaptive_reservoir.readout import (
    NLMSReadout,
    ReplayRidgeReadout,
    SlidingWindowRidgeReadout,
    create_readout,
)


def test_create_readout_returns_nlms() -> None:
    readout = create_readout(
        config=ReadoutConfig(name="nlms", learning_rate=0.2),
        feature_dim=3,
        dtype="float64",
    )

    assert isinstance(readout, NLMSReadout)
    assert readout.feature_dim == 3
    assert readout.learning_rate == 0.2


def test_create_readout_returns_replay_ridge() -> None:
    readout = create_readout(
        config=ReadoutConfig(
            name="replay_ridge",
            ridge_alpha=1e-4,
            buffer_size=7,
            update_interval=3,
        ),
        feature_dim=3,
        dtype="float64",
    )

    assert isinstance(readout, ReplayRidgeReadout)
    assert readout.feature_dim == 3
    assert readout.alpha == 1e-4
    assert readout.buffer_size == 7
    assert readout.refit_interval == 3


def test_create_readout_returns_sliding_ridge() -> None:
    readout = create_readout(
        config=ReadoutConfig(
            name="sliding_ridge",
            ridge_alpha=1e-4,
            window_size=5,
            update_interval=2,
        ),
        feature_dim=3,
        dtype="float64",
    )

    assert isinstance(readout, SlidingWindowRidgeReadout)
    assert readout.feature_dim == 3
    assert readout.alpha == 1e-4
    assert readout.window_size == 5
    assert readout.update_interval == 2


def test_create_readout_does_not_expose_experimental_rls() -> None:
    with pytest.raises(ValueError, match="readout.name must be one of"):
        ReadoutConfig(name="experimental_rls")  # type: ignore[arg-type]
