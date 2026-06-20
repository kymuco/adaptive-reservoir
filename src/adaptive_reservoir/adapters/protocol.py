"""Adapter protocols for host-event vectorization boundaries."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
_EventT_contra = TypeVar("_EventT_contra", contravariant=True)

__all__ = ["EventVectorizer", "FloatArray"]


@runtime_checkable
class EventVectorizer(Protocol[_EventT_contra]):
    """Protocol for host-owned event-to-vector adapters.

    Implementations may interpret application-specific events, but the reservoir
    boundary receives only a one-dimensional finite numeric vector. Semantic
    meaning, policy, consent, identity, memory, and action ownership remain in
    the host application rather than in :mod:`adaptive_reservoir`.
    """

    def transform(self, event: _EventT_contra) -> FloatArray:
        """Transform one host event into a numeric reservoir input vector."""

        ...
