# TODO: Classes pydoc
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class EvalContext:
    ref_point: np.ndarray | None = None
    weights: np.ndarray | None = None
    ref_front: np.ndarray | None = None
    utility: Callable = np.dot

    def missing(self, requires: frozenset[str]) -> set[str]:
        return {r for r in requires if getattr(self, r, None) is None}