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

# # a zero in a box implies the box is missing the same digit as the column
# # @Note: this constraint takes forever to compute...
# for i in range(0, 9, 3):
#     for j in range(9):
#         br = i // 3 * 3
#         bc = j // 3 * 3
#         box_mask = np.zeros((3, 3))
#         box_mask[:, j % 3] = 1
#         box = np.ma.masked_array(grid[br:br+3, bc:bc+3], mask=box_mask).compressed()
#
#         column_mask = np.zeros((9,))
#         column_mask[br:br+3] = 1
#         column = np.ma.masked_array(grid.T[j], mask=column_mask).compressed()
#         model += (Count(grid[i:i+3, j], 0) == 1).implies(sum(box) == sum(column))

t1 = time.perf_counter_ns()
if model.solve():
    print(grid.value())
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
