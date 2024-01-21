import argparse
import random
import time
from abc import ABC, abstractmethod
from bisect import bisect
from operator import itemgetter
from typing import Callable, Optional, TypeAlias


class BrokenConstraintError(Exception):
    pass


class Cell:
    def __init__(self, idx: int):
        self.options: int = 0b1111111111
        self.idx = idx
        self.constraints: list["Constraint"] = []

    def add_constraint(self, constraint: "Constraint"):
        self.constraints.append(constraint)
        constraint.add(self)

    @property
    def row(self):
        return self.idx // 9 + 1

    @property
    def col(self):
        return self.idx % 9 + 1

    @property
    def box(self):
        return self.idx // 27 * 3 + (self.idx % 9) // 3 + 1

    @property
    def entropy(self):
        return self.options.bit_count() - 1

    def __lt__(self, other):
        return self.entropy < other.entropy

    def __hash__(self):
        return self.idx

    def __repr__(self):
        return f"r{self.row}c{self.col}b{self.box}"


CellIdx: TypeAlias = int


class Constraint(ABC):
    def __init__(self, name: str):
        self.name = name
        self.cell_names: set[str] = set()
        self.cell_indices: list[CellIdx] = []

    def add(self, cell: Cell):
        self.cell_names.add(repr(cell))
        # Use indirection to simplify backtracking
        self.cell_indices.append(cell.idx)

    @property
    def extract_cells(self) -> Callable[[list[Cell]], list[Cell]]:
        return itemgetter(*self.cell_indices)

    @abstractmethod
    def constrain(self, board: list[Cell]) -> list[Cell]:
        ...

    def __repr__(self):
        return f"{self.__class__}{self.name}<{','.join(sorted(self.cell_names))}>"


class Unique(Constraint):
    def constrain(self, board: list[Cell]) -> list[Cell]:
        constrained_digits = 0
        constrained_cells = []
        for cell in sorted(self.extract_cells(board)):
            if cell.entropy == 0:
                constrained_digits |= cell.options
            else:
                prev_entropy = cell.entropy
                cell.options &= ~constrained_digits
                if cell.entropy < prev_entropy:
                    constrained_cells.append(cell)
            if cell.entropy < 0:
                raise BrokenConstraintError(f"{cell} has no remaining options")

        return constrained_cells


def solve(verbose: bool = True):
    board = []
    rows = {i: Unique(f"Row#{i}") for i in range(1, 10)}
    cols = {i: Unique(f"Col#{i}") for i in range(1, 10)}
    boxs = {i: Unique(f"Box#{i}") for i in range(1, 10)}

    for i in range(9 * 9):
        cell = Cell(i)
        board.append(cell)
        cell.add_constraint(rows[cell.row])
        cell.add_constraint(cols[cell.col])
        cell.add_constraint(boxs[cell.box])

    remaining_cells = board.copy()
    branches: list[tuple[tuple[int], tuple[int]]] = []
    number_of_backtracks = 0
    t_start = time.monotonic_ns()
    while True:
        try:
            while remaining_cells:
                remaining_cells.sort()
                # Pick the lowest entropy cell
                last_idx_in_entropy_group = bisect(remaining_cells, remaining_cells[0]) - 1
                if last_idx_in_entropy_group > 0:
                    # Don't think we need to add a branch here, shouldn't matter, right?
                    cell_idx = random.randint(0, last_idx_in_entropy_group)
                else:
                    cell_idx = 0

                cell = remaining_cells.pop(cell_idx)

                # Collapse
                if cell.entropy > 0:
                    # add branch
                    option = random.choice(
                        tuple(
                            filter(
                                None,
                                (
                                    cell.options & (1 << x)
                                    for x in range(cell.options.bit_length())
                                ),
                            )
                        )
                    )
                    # Remove what we chose in this branch
                    cell.options &= ~option
                    branches.append((tuple(c.options for c in board), tuple(c.idx for c in remaining_cells)))
                    cell.options = option

                constraint_propagation_stack: list[Cell] = [cell]
                while constraint_propagation_stack:
                    cell = constraint_propagation_stack.pop()
                    # Update constraints
                    for constraint in cell.constraints:
                        constraint_propagation_stack.extend(constraint.constrain(board))
        except BrokenConstraintError:
            if not branches:
                raise
            board_options, remaining_cells_indices = branches.pop()
            for idx, o in enumerate(board_options):
                board[idx].options = o
            remaining_cells = [board[idx] for idx in remaining_cells_indices]
            number_of_backtracks += 1
            if verbose:
                print(
                    number_of_backtracks if number_of_backtracks % 100 == 0 else ".",
                    sep="",
                    end="",
                )
        else:
            time_taken = time.monotonic_ns() - t_start
            if verbose:
                print("\n", fmt_board(board))
                print(
                    f"^ took {time_taken / 10**6:.2f} ms, with {number_of_backtracks} backtracks"
                )
            break


class ProgArgs(argparse.Namespace):
    seed: Optional[str] = None
    forever: bool = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str)
    parser.add_argument("-f", "--forever", action="store_true")
    args = parser.parse_args(namespace=ProgArgs())
    if args.seed is not None:
        random.seed(args.seed)

    solve()
    while args.forever:
        solve()


def fmt_board(board: list[Cell]):
    output = []
    for cell in board:
        idx = cell.idx
        if idx % 27 == 0:
            output += "—" * 31, "\n"

        if idx % 9 == 0:
            output += "|"

        output += " ", str(cell.options.bit_length() - 1), " "

        if idx % 3 == 2:
            output += "|"

        if idx % 9 == 8:
            output += "\n"

    output += "—" * 31
    return "".join(output)


if __name__ == "__main__":
    main()
