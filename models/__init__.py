"""Models package — BreakHis CNN with configurable normalization."""

from .cnn import BreakHisCNN, ClientUpdate, fedavg

__all__ = ["BreakHisCNN", "ClientUpdate", "fedavg"]
