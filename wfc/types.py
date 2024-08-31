from typing import Callable, NamedTuple

import numpy as np
import numpy.typing as npt


options = np.array([1 << x for x in range(0, 10)], dtype=np.uint16)
options_sum = np.sum(options)

COLLAPSED = np.uint16(1 << 15)

board_size = 81

index_board = np.arange(board_size)

invalid_board_dt = np.dtype("41B")
board_dt = np.dtype(f"{board_size}u2")


type Board = npt.NDArray[board_size, np.dtype[np.uint16]]
type CellIndex = int
type Constraint = Callable[[Board, CellIndex], None]


class BrokenConstraintsError(Exception):
    """Custom exception for control flow when a constraint is broken"""


class Result(NamedTuple):
    board: Board
    num_backtracks: int


class NoSolutions(Exception):
    pass
