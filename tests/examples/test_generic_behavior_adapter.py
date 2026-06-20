from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.generic_behavior_adapter import (
    BehaviorEvent,
    default_events,
    GenericBehaviorVectorizer,
    run_example,
)


def test_generic_behavior_vectorizer_conforms_to_adapter_protocol() -> None:
    from adaptive_reservoir.adapters import EventVectorizer

    vectorizer: EventVectorizer[BehaviorEvent] = GenericBehaviorVectorizer()

    assert isinstance(vectorizer, EventVectorizer)


def test_generic_behavior_vectorizer_returns_finite_numeric_vector() -> None:
    vectorizer = GenericBehaviorVectorizer()

    vector = vectorizer.transform(
        BehaviorEvent(
            latency_ms=250.0,
            error_count=2,
            activity_score=0.75,
            target_score=0.50,
        )
    )

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (3,)
    assert vector.dtype == np.float64
    assert np.all(np.isfinite(vector))
    assert np.allclose(vector, np.asarray((0.25, 0.2, 0.75), dtype=np.float64))


def test_generic_behavior_adapter_example_runs_model_steps() -> None:
    events = default_events()

    predictions = run_example(events)

    assert len(predictions) == len(events)
    assert all(isinstance(prediction, float) for prediction in predictions)
    assert all(np.isfinite(prediction) for prediction in predictions)


def test_generic_behavior_adapter_example_has_no_project_specific_imports() -> None:
    source = Path("examples/generic_behavior_adapter.py").read_text(encoding="utf-8")

    assert "Character_OS" not in source
    assert "character_os" not in source
    assert "hde_core" not in source
    assert "hde_docs" not in source
