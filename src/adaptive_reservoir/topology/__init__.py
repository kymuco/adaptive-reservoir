"""Concrete topology builders."""

from adaptive_reservoir.topology.random_sparse import RandomSparseTopologyBuilder
from adaptive_reservoir.topology.ring_shortcuts import RingShortcutsTopologyBuilder

__all__ = ["RandomSparseTopologyBuilder", "RingShortcutsTopologyBuilder"]
