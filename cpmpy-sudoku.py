import itertools

import numpy as np
from cpmpy import *

_ = None
given = np.array([[_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _]])


grid = intvar(0, 9, shape=given.shape, name="grid")

model = Model(
    # digits in rows must all be different
    [AllDifferent(row) for row in grid],
    # digits in columns must all be different
    [AllDifferent(col) for col in grid.T],
    # digits in boxes must all be different
    [AllDifferent(grid[i:i+3, j:j+3])
     for i, j in itertools.product(range(0, 9, 3), repeat=2)],
    # given digits
    grid[given != _] == given[given != _],
    # 5 zeroes in the grid
    Count(grid, 0) == 5,
    # no zeroes on the positive diagonal
    [grid[i, i] != 0 for i in range(0, 9)],
    # a zero in the grid must index another zero by the digits missing in row/column
    [(grid[i, j] == 0)
     .implies(grid[44 - sum(grid[i]), 44 - sum(grid.T[j])] == 0)
     for i, j in np.ndindex(*grid.shape)],
)

if model.solve():
    print(grid.value())
else:
    print("No solution found")
