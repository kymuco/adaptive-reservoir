from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import adaptive_reservoir
from adaptive_reservoir.adapters import (
    __all__ as adapters_exports,
    EventVectorizer,
    FloatArray,
)


@dataclass(frozen=True)
class BehaviorEvent:
    signal: float
    pressure: float


class BehaviorEventVectorizer:
    def transform(self, event: BehaviorEvent) -> FloatArray:
        return np.asarray((event.signal, event.pressure), dtype=np.float64)


class MissingTransform:
    pass


def test_event_vectorizer_accepts_structural_implementation() -> None:
    vectorizer = BehaviorEventVectorizer()

    assert isinstance(vectorizer, EventVectorizer)

    vector = vectorizer.transform(BehaviorEvent(signal=0.75, pressure=-0.25))

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (2,)
    assert vector.dtype == np.float64
    assert np.all(np.isfinite(vector))
    assert np.allclose(vector, np.asarray((0.75, -0.25), dtype=np.float64))


def test_event_vectorizer_supports_concrete_event_type_annotations() -> None:
    vectorizer: EventVectorizer[BehaviorEvent] = BehaviorEventVectorizer()

    vector = vectorizer.transform(BehaviorEvent(signal=1.0, pressure=0.5))

    assert np.allclose(vector, np.asarray((1.0, 0.5), dtype=np.float64))


def test_event_vectorizer_rejects_objects_without_transform() -> None:
    assert not isinstance(MissingTransform(), EventVectorizer)


def test_adapters_package_exports_protocol_contract() -> None:
    assert set(adapters_exports) == {"EventVectorizer", "FloatArray"}


def test_event_vectorizer_is_not_root_exported_before_stable_api_review() -> None:
    assert not hasattr(adaptive_reservoir, "EventVectorizer")
