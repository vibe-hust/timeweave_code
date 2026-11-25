"""Chunk layout extracted from the research runtime; scoring dependencies omitted."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class Chunk:
    index: int
    latent_start: int
    latent_stop: int
    rgb_start: int
    rgb_stop: int

    @property
    def latent_count(self) -> int:
        return self.latent_stop - self.latent_start

def chunk_layout(lengths: Sequence[int] = (5, 6, 6, 6, 6, 6, 6),
                 stride: int = 4) -> tuple[Chunk, ...]:
    if not lengths or stride < 1 or any(int(n) != n or n <= 0 for n in lengths):
        raise ValueError("Positive integer chunk lengths and stride required")
    result, start = [], 0
    for index, n in enumerate(lengths):
        stop = start + int(n)
        rgb_start = 0 if start == 0 else (start - 1) * stride + 1
        result.append(Chunk(index, start, stop, rgb_start, (stop - 1) * stride))
        start = stop
    return tuple(result)
