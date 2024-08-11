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
from typing import Optional

import numpy as np
import numpy.ma as ma
from rich.progress import Progress, TextColumn, TimeElapsedColumn, BarColumn

from . constraints import constraints_list, assert_board_validity
from .types import (
    board_size,
    board_dt,
    invalid_board_dt,
    Result,
    options,
    COLLAPSED,
    Board,
    BrokenConstraintsError,
    NoSolutions,
)


@np.vectorize
def entropy(x: int):
    return x.bit_count()


invalid_boards = np.zeros(128, dtype=invalid_board_dt)
invalid_boards_idx = -1


def solve(initial_board: Optional[Board] = None, *, seed=None, debug=False, quiet=False) -> Result:
    global invalid_boards, invalid_boards_idx
    rng = np.random.default_rng(seed)
    if seed is None:
        print(rng.bit_generator.seed_seq)
    board_stack = np.empty((256,), dtype=board_dt)
    board_stack[0] = initial_board if initial_board is not None else sum(options)
    board_idx = 0
    board = np.empty((), dtype=board_dt)
    num_backtracks = 0
    prevented_recalculations = 0
    if not (debug or quiet):
        p = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TextColumn("Iterations: {task.completed} | Backtracks: {task.fields[backtracks]} | Invalid Boards: {task.fields[invalid_boards]} | Prevented Recalculations: {task.fields[prevented_recalculations]}"),
        )
    else:
        from unittest.mock import MagicMock
        p = MagicMock()
    with p as progress:
        task = progress.add_task("Generating solution...", total=None, backtracks=num_backtracks, invalid_boards=invalid_boards_idx, prevented_recalculations=prevented_recalculations)
        while True:
            if board_idx < 0:
                raise NoSolutions
            progress.update(task, advance=1, backtracks=num_backtracks, invalid_boards=invalid_boards_idx, prevented_recalculations=prevented_recalculations)
            board[:] = board_stack[board_idx]
            if debug and not quiet:
                print("\033[1;1H", fmt_debug_board(board), sep="")
            ma_non_collapsed = ma.masked_greater_equal(board, COLLAPSED, copy=False)
            if np.all(ma_non_collapsed.mask):
                assert_board_validity(board)
                return Result(board, num_backtracks)

            current_board = np.zeros((), dtype=invalid_board_dt)
            for (idx,), val in ma.ndenumerate(ma.masked_array(board, ~ma_non_collapsed.mask, copy=False)):
                current_board[idx // 2] = int(math.log2(val ^ COLLAPSED)) << (idx % 2) * 4

            # Check if current board has already been made invalid.
            s = slice(0, invalid_boards_idx + 1)
            if invalid_boards_idx >= 0 and (invalid_boards[s] & current_board == invalid_boards[s]).all(axis=1).any():
                board_idx -= 1
                assert board_idx >= 0
                prevented_recalculations += 1
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
                assert digit != 0

            # Assign the collapsed cell.
            board[cell_idx] = digit | COLLAPSED
            try:
                # Reduce search-space using constraints
                for constraint in constraints_list:
                    constraint(board, cell_idx)
                    if debug:
                        print("\033[1;1H", fmt_debug_board(board), sep="")


            except BrokenConstraintsError:
                if debug:
                    print("\033[1;1H", fmt_debug_board(board), sep="")

                invalid_boards_idx += 1
                if invalid_boards_idx >= invalid_boards.shape[0]:
                    new_shape = (invalid_boards.shape[0] * 2, *invalid_boards.shape[1:])
                    try:
                        invalid_boards.resize(new_shape)
                    except ValueError:
                        # When using debugger, due to inspection of variables, the resize above fails
                        tmp_stack = np.empty(new_shape, invalid_boards.dtype)
                        tmp_stack[:invalid_boards.shape[0]] = invalid_boards
                        invalid_boards = tmp_stack
                invalid_boards[invalid_boards_idx] = current_board
                num_backtracks += 1
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


def fmt_debug_board(board: Board, current_idx=None):
    output: list[str | None] = [
        "╔═══════╤═══════╤═══════╦═══════╤═══════╤═══════╦═══════╤═══════╤═══════╗\n║",
        *[None] * 9 * 2,
        "╟───────┼───────┼───────╫───────┼───────┼───────╫───────┼───────┼───────╢\n║",
        *[None] * 9 * 2,
        "╟───────┼───────┼───────╫───────┼───────┼───────╫───────┼───────┼───────╢\n║",
        *[None] * 9 * 2,
        "╠═══════╪═══════╪═══════╬═══════╪═══════╪═══════╬═══════╪═══════╪═══════╣\n║",
        *[None] * 9 * 2,
        "╟───────┼───────┼───────╫───────┼───────┼───────╫───────┼───────┼───────╢\n║",
        *[None] * 9 * 2,
        "╟───────┼───────┼───────╫───────┼───────┼───────╫───────┼───────┼───────╢\n║",
        *[None] * 9 * 2,
        "╠═══════╪═══════╪═══════╬═══════╪═══════╪═══════╬═══════╪═══════╪═══════╣\n║",
        *[None] * 9 * 2,
        "╟───────┼───────┼───────╫───────┼───────┼───────╫───────┼───────┼───────╢\n║",
        *[None] * 9 * 2,
        "╟───────┼───────┼───────╫───────┼───────┼───────╫───────┼───────┼───────╢\n║",
        *[None] * 9 * 2,
        "╚═══════╧═══════╧═══════╩═══════╧═══════╧═══════╩═══════╧═══════╧═══════╝",
    ]

    # Assumes the indices are in order...
    for (idx,), cell in np.ndenumerate(board):
        output_idx = (
            1  # start
            + idx % 9  # per column
            + (idx // 9) * (18 + 1)  # per row
        )
        sep = "│" if idx % 3 != 2 else "║"
        top = []
        bottom = []
        for (i,), v in np.ndenumerate(options):
            arr = top if i < 5 else bottom
            if v & cell:
                if cell & COLLAPSED:
                    arr += f"\033[42m{i}\033[0m"
                else:
                    arr += str(i)
            else:
                arr += " "

        output[output_idx] = " {} {sep}{end}".format(
            "".join(top),
            sep=sep,
            end="\n║" if idx % 9 == 8 else ""
        )
        output[output_idx + 9] = " {} {sep}{end}".format(
            "".join(bottom),
            sep=sep,
            end="\n" if idx % 9 == 8 else "",
        )

    return "".join(output)


def main(initial_board: Optional[Board] = None, *, seed=None, debug=False, quiet=False):
    t = time.perf_counter_ns()
    result = solve(initial_board, seed=seed, debug=debug, quiet=quiet)
    if quiet:
        return
    print("\033[2J")  # clear the screen
    t = time.perf_counter_ns() - t
    print(fmt_board(result.board))
    print(f"Took {t/1e6:.3f} ms with {result.num_backtracks} backtracks")


if __name__ == "__main__":
    main()
