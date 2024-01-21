import argparse
import random
from abc import ABC, abstractmethod
from bisect import bisect
from operator import itemgetter
from typing import Optional, TypeAlias, Callable


class BrokenConstraintError(Exception):
    pass


class Cell:
    def __init__(self, idx: int):
        self.options = {*range(1, 10)}
        self.idx = idx
        self.y = idx // 9
        self.row = idx // 9 + 1
        self.col = idx % 9 + 1
        self.box = idx // 27 * 3 + (idx % 9) // 3 + 1
        self.constraints: list["Constraint"] = []

    def add_constraint(self, constraint: "Constraint"):
        self.constraints.append(constraint)
        constraint.add(self)

    @property
    def entropy(self):
        return len(self.options) - 1

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
    def constrain(self, board: list[Cell]):
        ...

    def __repr__(self):
        return f"{self.__class__}{self.name}<{','.join(sorted(self.cell_names))}>"


class Unique(Constraint):
    def constrain(self, board: list[Cell]):
        constrained_digits = set()
        for cell in sorted(self.extract_cells(board)):
            if cell.entropy == 0:
                constrained_digits.update(cell.options)
            else:
                cell.options.difference_update(constrained_digits)
                if cell.entropy == 0:
                    constrained_digits.update(cell.options)
            if cell.entropy < 0:
                raise BrokenConstraintError(f"{cell} has no remaining options")


class ProgArgs(argparse.Namespace):
    seed: Optional[str] = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str)
    args = parser.parse_args(namespace=ProgArgs())
    if args.seed is not None:
        random.seed(args.seed)

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

    remaining_cells = set(board)

    while remaining_cells:
        by_entropy = sorted(remaining_cells)
        # Pick the lowest entropy cell
        cell: Cell = random.choice(by_entropy[: bisect(by_entropy, by_entropy[0])])
        remaining_cells.remove(cell)
        assert cell.entropy >= 0
        # Collapse
        option = {random.choice(list(cell.options))}
        cell.options = option

        # Update constraints
        for constraint in cell.constraints:
            constraint.constrain(board)

    print(fmt_board(board))


def fmt_board(board: list[Cell]):
    output = []
    for cell in board:
        idx = cell.idx
        if idx % 27 == 0:
            output += "—" * 31, "\n"

        if idx % 9 == 0:
            output += "|"

        digit = cell.options.pop()
        output += " ", str(digit), " "

        if idx % 3 == 2:
            output += "|"

        if idx % 9 == 8:
            output += "\n"

    output += "—" * 31, "\n"
    return "".join(output)


if __name__ == "__main__":
    main()
