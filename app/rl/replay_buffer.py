from __future__ import annotations

from collections import deque
import random

from app.rl.dataset import TrainingSample


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._items: deque[TrainingSample] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, sample: TrainingSample) -> None:
        self._items.append(sample)

    def add_many(self, samples: list[TrainingSample]) -> None:
        self._items.extend(samples)

    def sample(self, batch_size: int, rng: random.Random | None = None) -> list[TrainingSample]:
        if not self._items:
            return []
        rng = rng or random.Random()
        batch_size = max(1, min(int(batch_size), len(self._items)))
        return rng.sample(list(self._items), batch_size)
