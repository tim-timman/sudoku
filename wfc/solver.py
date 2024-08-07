"""A speedy solver based on Wave Function Collapse and backtracking

-- Pseudocode --
Create a stack for storing boards.
Create a list for storing invalid boards.
Push an initial board with all cells containing all options to the stack.

Loop:
  Copy the top of the stack into board.
  If all cells are collapsed. Done: return board. (assert all constraints are valid)

  If we're on an invalid board, pop the stack and continue loop.

  Find the lowest entropy non-collapsed cell(s) of board.
    If multiple cells share the same entropy, pick one at random.

  If there are multiple options of the cell.
    Pick one of the cell options at random.
    Remove the selected digit from options of the cell the top of the stack.
  Else:
    Pop one from the stack as we will have exhausted this cell branch.

  Assign the cell that option and mark it as collapsed.

  Update constraints.
  If at any point, an updated constraint empties the options of any other cell; this board is not valid.
    Store the board's current collapsed cell digits in the invalid boards; continue loop.
  Else, push the current board onto the stack; continue loop.
"""
import math
import time
import signal
from itertools import chain
from typing import Callable, NamedTuple, Optional

import numpy as np
import numpy.ma as ma
import numpy.typing as npt

signal.signal(signal.SIGUSR1, lambda *args: breakpoint())

options = np.array([1 << x for x in range(0, 10)], dtype=np.uint16)

COLLAPSED = np.uint16(1 << 15)

board_size = 81

index_board = np.arange(board_size)

# Each array contains board indices that must all contain unique digits.
# Note: Keep the row, column, box constraints in 1st, 2nd, and 3rd positions respectively.
# Hint: Extract the specific row, column, box as follows:
#   row, col, box = unique_constraints[np.isin(unique_constraints, cell_idx).any(axis=1)][:3]
unique_constraints = np.array([
    # Sudoku row constraints
    *index_board.reshape((9, 9)),
    # Sudoku column constraints
    *index_board.reshape((9, 9)).T,
    # Sudoku box constraints
    *chain.from_iterable(index_board.reshape((27, 3))[i::3].reshape((3, 9)) for i in range(3))
    # ---[ extra constraints below ]---
])

type Board = npt.NDArray[board_size, np.dtype[np.uint16]]
type CellIndex = int


class BrokenConstraintsError(Exception):
    """Custom exception for control flow when a constraint is broken"""


def apply_unique_constraints(board: Board, idx: CellIndex):
    for constraint in unique_constraints[np.isin(unique_constraints, idx).any(axis=1)]:
        mask = ma.masked_less(board[constraint], COLLAPSED, copy=False)
        if np.all(mask.mask):
            # All cells already collapsed.
            continue
        collapsed = ma.sum(mask ^ COLLAPSED)
        board[constraint[mask.mask]] &= ~collapsed

        # If any cell has no (zero) options left.
        if np.any(board[constraint] & ~COLLAPSED == 0):
            raise BrokenConstraintsError


def apply_zero_constraint(board: Board, _: CellIndex):
    """Special global zero constraint

    A column with a zero in it must match the digits of the zero's box.
    """
    collapsed_zero_indices = np.flatnonzero(board == options[0] | COLLAPSED)

    for cell_idx in collapsed_zero_indices:
        col, box = constraints = unique_constraints[np.isin(unique_constraints, cell_idx).any(axis=1)][1:3]
        # Find the symmetric difference between box and column;
        #  these must be match.
        box_column_intersection = np.intersect1d(col, box, assume_unique=True)
        box_part = np.setdiff1d(box, box_column_intersection, assume_unique=True)
        column_part = np.setdiff1d(col, box_column_intersection, assume_unique=True)

        column_options = np.bitwise_or.reduce(board[column_part]) | COLLAPSED
        board[box_part] &= column_options

        box_options = np.bitwise_or.reduce(board[box_part]) | COLLAPSED
        board[column_part] &= box_options

        if np.any(board[constraints] & ~COLLAPSED == 0):
            raise BrokenConstraintsError


# Note: order matters
constraints: list[Callable[[Board, CellIndex], None]] = [
    apply_unique_constraints,
    apply_zero_constraint
]


@np.vectorize
def entropy(x: int):
    return x.bit_count()


invalid_board_dt = np.dtype("41B")
board_dt = np.dtype(f"{board_size}u2")


class Result(NamedTuple):
    board: Board
    num_backtracks: int


def solve(initial_board: Optional[Board] = None, *, seed=None) -> Result:
    rng = np.random.default_rng(seed)
    if seed is not None:
        print(rng.bit_generator.seed_seq)

    invalid_boards = np.zeros(128, dtype=invalid_board_dt)
    invalid_boards_idx = -1
    board_stack = np.empty((256,), dtype=board_dt)
    board_stack[0] = initial_board if initial_board is not None else sum(options)
    board_idx = 0
    board = np.empty((), dtype=board_dt)
    num_backtracks = 0

    while True:
        board[:] = board_stack[board_idx]

        ma_non_collapsed = ma.masked_greater_equal(board, COLLAPSED, copy=False)
        if np.all(ma_non_collapsed.mask):
            # Unique set constraints
            for constraint in unique_constraints:
                assert np.all(np.unique(board[constraint], return_counts=True)[1] == 1)

            # Zero constraint
            for cell_idx in np.flatnonzero(board == options[0] | COLLAPSED):
                col, box = unique_constraints[np.isin(unique_constraints, cell_idx).any(axis=1)][1:3]
                assert np.intersect1d(board[col], board[box]).size == col.size

            return Result(board, num_backtracks)

        current_board = np.zeros((), dtype=invalid_board_dt)
        for (idx,), val in ma.ndenumerate(ma.masked_array(board, ~ma_non_collapsed.mask, copy=False)):
            current_board[idx // 2] = int(math.log2(val ^ COLLAPSED)) << (idx % 2) * 4

        # Check if current board has already been made invalid.
        if invalid_boards_idx >= 0 and (invalid_boards[:invalid_boards_idx + 1] & current_board == invalid_boards[:invalid_boards_idx + 1]).all(axis=1).any():
            board_idx -= 1
            assert board_idx >= 0
            continue

        entropy_arr = entropy(ma_non_collapsed)
        lowest_entropy_indices = np.flatnonzero(entropy_arr == entropy_arr.min())
        cell_idx = rng.choice(lowest_entropy_indices)
        cell = board[cell_idx]

        if entropy_arr[cell_idx] > 1:
            # Multiple options remaining; select one of the options.
            digit = options[rng.choice(np.flatnonzero(options & cell))]
            # Update the current board on the stack, removing the branch we're now exploring.
            board_stack[board_idx][cell_idx] ^= digit
        else:
            digit = board[cell_idx]
            # We're exploring the final option of this board, pop the stack.
            board_idx -= 1
            assert board_idx >= 0
            assert digit != 0

        # Assign the collapsed cell.
        board[cell_idx] = digit | COLLAPSED
        try:
            # Reduce search-space using constraints
            for constraint in constraints:
                constraint(board, cell_idx)

        except BrokenConstraintsError:
            invalid_boards_idx += 1
            if invalid_boards_idx >= invalid_boards.shape[0]:
                new_shape = (invalid_boards.shape[0] * 2, *invalid_boards.shape[1:])
                try:
                    invalid_boards.resize(new_shape)
                except ValueError:
                    # When using debugger, due to inspection of variables, the resize above fails
                    tmp_stack = np.empty(new_shape, invalid_boards.dtype)
                    tmp_stack[:] = invalid_boards
                    invalid_boards = tmp_stack
            # @Performance: we could potentially compact these, to the minimum invalid boards
            invalid_boards[invalid_boards_idx] = current_board
            num_backtracks += 1
            continue
        else:
            board_idx += 1

            # Do we need to resize?
            if board_idx >= board_stack.shape[0]:
                # Double for amortized constant time
                new_shape = (board_stack.shape[0] * 2, *board_stack.shape[1:])
                try:
                    board_stack.resize(new_shape)
                except ValueError:
                    # When using debugger, due to inspection of variables, the resize above fails
                    tmp_stack = np.empty(new_shape, board_stack.dtype)
                    tmp_stack[:board_stack.shape[0]] = board_stack
                    board_stack = tmp_stack

            # Put board onto the stack
            board_stack[board_idx] = board
            continue


def fmt_invalid_board(invalid_board: np.ndarray):
    board = np.zeros((board_size,), dtype=np.uint16)
    for (idx,), x in np.ndenumerate(invalid_board):
        if x > 0:
            if x & 0xf:
                board[idx] = options[x & 0xf] | COLLAPSED
            if x >> 4:
                board[idx + 1] = options[x >> 4] | COLLAPSED
    return fmt_board(board)


def fmt_board(board: Board):
    output: list[str] = []
    for (idx, ), cell in np.ndenumerate(board):
        if idx % 27 == 0:
            output += "—" * 31, "\n"

        if idx % 9 == 0:
            output += "|"

        output += " ", "*" if cell < COLLAPSED else str(np.flatnonzero(options == cell ^ COLLAPSED)[0]), " "

        if idx % 3 == 2:
            output += "|"

        if idx % 9 == 8:
            output += "\n"

    output += "—" * 31
    return "".join(output)


def fmt_board_options(board: Board, collapsed_idx=None):
    output: list[str] = []
    for (idx, ), cell in np.ndenumerate(board):
        if idx % 27 == 0:
            output += "—" * 31, "\n"

        if idx % 9 == 0:
            output += "|"

        if idx == collapsed_idx:
            output += "\033[31m"
        output += " ", "".join(str(x) for x in np.flatnonzero(options & cell)).center(10), " "
        if idx == collapsed_idx:
            output += "\033[0m"

        if idx % 3 == 2:
            output += "|"

        if idx % 9 == 8:
            output += "\n"

    output += "—" * 31
    return "".join(output)


def main(initial_board: Optional[Board] = None, *, seed=None):
    t = time.perf_counter_ns()
    result = solve(initial_board, seed=seed)
    print("\033[2J")  # clear the screen
    t = time.perf_counter_ns() - t
    print(fmt_board(result.board))
    print(f"Took {t/1e6:.3f} ms with {result.num_backtracks} backtracks")


if __name__ == "__main__":
    main()
