"""
1. Generate board
2. Randomly pick one cell with the lowest entropy, or finish if everything's collapsed.
3. Collapse the cell to a single digit. If multiple where available, store the non-explored branch on the stack.
4. Propagate the current state change against all constraints. If a constraint was broken, pop the branch stack.
5. Goto 2
"""
import time
from itertools import chain
import signal

import numpy as np
import numpy.ma as ma
from numpy.typing import NDArray

signal.signal(signal.SIGUSR1, lambda *args: breakpoint())

rng = np.random.default_rng()
options = np.array([1 << x for x in range(0, 10)], dtype=np.uint16)

COLLAPSED = 1 << 15

board_size = 81

index_board = np.arange(board_size)

unique_constraints = np.array([
    *index_board.reshape((9, 9)),
    *index_board.reshape((9, 9)).T,
    *chain.from_iterable(index_board.reshape((27, 3))[i::3].reshape((3, 9)) for i in range(3))
])


@np.vectorize
def entropy(x: int):
    return x.bit_count()


def solve():
    print("\033[2J")
    board_stack = np.empty(board_size * 8, dtype=np.uint16)

    board_stack[:81] = sum(options)  # Set the initial board to one with all options
    stack_idx = 0

    # "pop" from the board stack
    board = board_stack[stack_idx * board_size: (stack_idx + 1) * board_size].copy()
    stack_idx -= 1
    num_backtracks = 0
    while True:
        non_collapsed = ma.masked_greater_equal(board, COLLAPSED, copy=False)
        if np.all(non_collapsed.mask):
            for constraint in unique_constraints:
                assert np.all(np.unique(board[constraint], return_counts=True)[1] == 1)

            print("\033[2J")
            return board, num_backtracks

        entropy_arr = entropy(non_collapsed)
        lowest_entropy_indices = np.flatnonzero(entropy_arr == entropy_arr.min())
        cell_idx = rng.choice(lowest_entropy_indices)
        cell = board[cell_idx]

        if entropy_arr[cell_idx] > 1:
            # Randomly select one of the available digits of this cell
            digit = options[rng.choice(np.flatnonzero(options & cell))]

            if (stack_idx + 1) * board_size >= board_stack.shape[0]:  # Do we need to resize?
                board_stack.resize((board_stack.shape[0] * 2,))  # Double for amortized constant time

            # "push" onto the board_stack
            stack_idx += 1
            branch = board_stack[stack_idx * board_size: (stack_idx + 1) * board_size] = board
            # Remove this branch from its options
            branch[cell_idx] ^= digit

            # Collapse to the chosen digit in this branch
            board[cell_idx] = digit | COLLAPSED
        else:
            stack_idx -= 1
            # Mark as collapsed
            board[cell_idx] |= COLLAPSED

        for constraint in unique_constraints[np.isin(unique_constraints, cell_idx).any(axis=1)]:
            mask = ma.masked_less(board[constraint], COLLAPSED, copy=False)
            if np.all(mask.mask):
                continue
            collapsed = ma.sum(mask ^ COLLAPSED, dtype=np.uint16)
            board[constraint[mask.mask]] &= ~collapsed
            try:
                assert np.all(board[constraint] & ~COLLAPSED > 0)
            except AssertionError:
                # "pop" from the board stack
                assert stack_idx >= 0
                board = board_stack[stack_idx * board_size: (stack_idx + 1) * board_size].copy()
                stack_idx -= 1
                num_backtracks += 1
            print("\033[1;1H", fmt_board(board), sep="", end="", flush=True)


def fmt_board(board: NDArray):
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


def main():
    t = time.perf_counter_ns()
    board, num_backtracks = solve()
    t = time.perf_counter_ns() - t
    print(fmt_board(board))
    print(f"Took {t/1e6:.3f} ms with {num_backtracks} backtracks")


if __name__ == "__main__":
    main()
