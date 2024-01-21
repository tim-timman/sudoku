import random
from bisect import bisect
from collections import defaultdict

random.seed("bork")


class Cell:
    def __init__(self, idx: int):
        self.options = {*range(1, 10)}
        self.idx = idx
        self.y = idx // 9
        self.row = idx // 9 + 1
        self.col = idx % 9 + 1
        self.box = idx // 27 * 3 + (idx % 9) // 3 + 1
        self.constraints = []

    def add_constraint(self, constraint: list["Cell"]):
        self.constraints.append(constraint)

    @property
    def entropy(self):
        return len(self.options) - 1

    def __lt__(self, other):
        return self.entropy < other.entropy

    def __hash__(self):
        return self.idx

    def __repr__(self):
        return f"r{self.row}c{self.col}b{self.box}"


board = []
rows = defaultdict(list)
cols = defaultdict(list)
boxs = defaultdict(list)


for i in range(9 * 9):
    cell = Cell(i)
    board.append(cell)
    for c in (rows[cell.row], cols[cell.col], boxs[cell.box]):
        c.append(cell)
        cell.constraints.append(c)


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
        for cell in constraint:
            if cell not in remaining_cells:
                continue
            cell.options.difference_update(option)


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


print(fmt_board(board))
