import itertools
import time

import numpy as np
from cpmpy import *

_ = None
given = np.array([[_, _, _, _, 0, _, _, _, _],
                  [_, _, 0, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, 5],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, 5, _, _, _, 4],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, 3]])


def to_boolean_grid(g):
    a = np.zeros(*g.shape, 10)
    for i, j in np.ndindex(*g.shape):
        if (digit := g[i, j]) is not _:
            a[i, j, digit] = 1
    return a


def to_digit_grid(b):
    a = np.ndarray(b.shape[:2], dtype=int)
    for i, j, k in np.ndindex(*b.shape):
        if b[i, j, k] == 1:
            a[i, j] = k
    return a


grid = boolvar(shape=(*given.shape, 10), name="grid")

model = Model(
    # a cell can only contain one digit
    [sum(cell) == 1 for cell in grid.reshape(-1, 10)],

    # digits in rows must all be different
    [sum(digit) <= 1 for digit in np.transpose(grid, (0, 2, 1)).reshape(-1, 9)],

    # digits in columns must all be different
    [sum(digit) <= 1 for digit in np.transpose(grid, (1, 2, 0)).reshape(-1, 9)],

    # digits in boxes must all be different
    [sum(digit) <= 1 for i, j in itertools.product(range(0, 9, 3), repeat=2) for digit in grid[i:i+3, j:j+3].T],

    # given digits
    [grid[i, j, given[i, j]] == 1 for i, j in np.ndindex(*given.shape) if given[i, j] is not None],

    # 5 zeroes in the grid
    sum(grid.transpose(2, 0, 1).reshape(10, -1)[0]) == 5,

    # no zeroes on the positive diagonal
    [grid[i, i, 0] == 0 for i in range(0, 9)],

    # a zero in the grid must index another zero by the digits missing in row/column
    [(grid[i, j, 0] == 1)
     .implies(grid[44 - sum(grid[i, :, k] * k for k in range(1, 10)), 44 - sum(grid[:, j, k] * k for k in range(1, 10)), 0] == 1)
     for i, j in np.ndindex(9, 9)],

    # # a box with a zero must be missing the same digit as in the zero's column
    # [(grid[i, j, 0] == 1)
    #  .implies(
    #     sum(grid[i // 3 * 3: i // 3 * 3 + 3, j // 3 * 3:j // 3 * 3 + 3].transpose(2, 0, 1).reshape(10, -1)[k] * k for k in range(1, 10))
    #     ==
    #     sum(grid[:, j].transpose(1, 0).reshape(10, -1)[k] * k for k in range(1, 10)))
    #  for i, j in np.ndindex(9, 9)],
)

t1 = time.perf_counter_ns()
if model.solve():
    print(to_digit_grid(grid.value()))
else:
    print("No solution found")

print(model.status())
t = time.perf_counter_ns() - t1

units = ["µs", "ms", "s"]
unit = "ns"
while t > 1e3:
    try:
        unit = units.pop(0)
    except IndexError:
        break
    t = t / 1e3

print(f"Took: {t:.2f} {unit}\n")
