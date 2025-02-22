from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np


class TerminalStateBuffer:
    def __init__(self, buffer_size: int, prioritized="none", rank_weight=1e-2):
        self.buffer_size = buffer_size
        self.prioritized = prioritized
        self.rank_weight = rank_weight

        # key: x, value: (r, priority)
        self.buffer = OrderedDict()

    def add(self, x: Tuple[int], r: float, priority=1.0) -> None:
        if x in self.buffer:
            # Remove old entry to keep the latest one
            del self.buffer[x]
        elif len(self.buffer) >= self.buffer_size:
            # Remove the oldest item (FIFO)
            self.buffer.popitem(last=False)
        # Add new entry
        self.buffer[x] = (r, priority)

    def add_batch(self, xs: List[Tuple[int]], rewards: List[float], priorities: Union[List[float], None]) -> None:
        for i in range(len(xs)):
            priority = priorities[i] if priorities is not None else 1.0
            self.add(xs[i], rewards[i], priority)

    def sample(self, batch_size: int) -> List[Tuple[Tuple[int], float]]:
        if not self.buffer:
            return []

        data_items = list(self.buffer.items())
        data_len = len(data_items)

        if self.prioritized == "none":
            indices = np.random.choice(data_len, batch_size, replace=True)
        else:
            priorities = np.array([v[1] for v in self.buffer.values()])  # Extract priorities
            ranks = np.argsort(np.argsort(-priorities))
            weights = 1.0 / (self.rank_weight * data_len + ranks)
            weights /= weights.sum()
            indices = np.random.choice(data_len, batch_size, replace=True, p=weights)

        sampled_data = []
        for i in indices:
            x, (r, _) = data_items[i]
            sampled_data.append((x, r))

        return sampled_data


class StateBuffer:
    def __init__(self, buffer_size: int, prioritized="none", rank_weight=1e-2):
        self.buffer_size = buffer_size
        self.prioritized = prioritized
        self.rank_weight = rank_weight

        # key: (s, s_next), value: (a, r, is_terminal, priority)
        self.buffer = OrderedDict()

    def add(self, s: Tuple[int], a: int, s_next: Tuple[int], r: float, is_terminal: bool, priority=1.0) -> None:
        key = (s, s_next)
        if key in self.buffer:
            # Remove old entry to keep the latest one
            del self.buffer[key]
        elif len(self.buffer) >= self.buffer_size:
            # Remove the oldest item (FIFO)
            self.buffer.popitem(last=False)
        # Add new entry
        self.buffer[key] = (a, r, is_terminal, priority)

    def sample(self, batch_size: int) -> List[Tuple[Tuple[int], int, Tuple[int], float, bool]]:
        if not self.buffer:
            return []

        data_items = list(self.buffer.items())
        data_len = len(data_items)

        if self.prioritized == "none":
            indices = np.random.choice(data_len, batch_size, replace=True)
        else:
            priorities = np.array([v[3] for _, v in data_items])
            ranks = np.argsort(np.argsort(-priorities))
            weights = 1.0 / (self.rank_weight * data_len + ranks)
            weights /= weights.sum()
            indices = np.random.choice(data_len, batch_size, replace=True, p=weights)

        sampled_data = []
        for i in indices:
            (s, s_next), (a, r, is_terminal, _) = data_items[i]
            sampled_data.append((s, a, s_next, r, is_terminal))

        return sampled_data
