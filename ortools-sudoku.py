import math
import time

from ortools.sat.python import cp_model
import numpy as np

_ = None
BOARD = [[_, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _]]


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, grid_vars):
        super().__init__()
        self.grid_vars = grid_vars
        self.prev_time = time.perf_counter_ns()

    def on_solution_callback(self):
        print(f"Took: {(time.perf_counter_ns() - self.prev_time) / 1e3:.2f} µs\n")
        for row in self.grid_vars:
            print([d for col in row for d, v in enumerate(col) if self.Value(v)])
        self.prev_time = time.perf_counter_ns()
        self.StopSearch()


def main(board=None):
    if board is None:
        board = BOARD

    model = cp_model.CpModel()
    size = 9
    digits = 10

    # construct the variables for the grid
    grid_vars = np.array([[[model.NewBoolVar(f"r{i+1}c{j+1}d{k}")
                            for k in range(digits)]
                           for j in range(size)]
                          for i in range(size)])
    grid_vars_t = grid_vars.transpose(1, 0, 2)

    # Cells may only contain one of the digits
    for i, j in np.ndindex(size, size):
        model.AddExactlyOne(grid_vars[i, j])

    for i, k in np.ndindex(size, digits):
        # at most one of each digit in every row
        model.AddAtMostOne(*grid_vars[i, :, k])
        # at most one of each digit in every column
        model.AddAtMostOne(*grid_vars_t[i, :, k])

    sqrt_size = int(math.sqrt(size))
    # at most one of each digit in every box
    for i, j in np.ndindex(sqrt_size, sqrt_size):
        box = grid_vars[i*3:(i*3)+sqrt_size, j*3:(j*3)+sqrt_size]
        for digit in range(digits):
            model.AddAtMostOne(*box[:, :, digit].flat)

    # exactly 5 zeros in the grid
    model.Add(sum(grid_vars[:, :, 0].flat) == 5)

    # the zeroes may not be on the positive diagonal
    for i in range(size):
        model.Add(grid_vars[i, i, 0] == 0)

    # intermediate variables (channeling constraints)
    intermediate_zeroes = np.ndarray((size, size), dtype=cp_model.IntVar)
    intermediate_missing_row_digits = np.ndarray((size, size), dtype=cp_model.IntVar)
    intermediate_missing_col_digits = np.ndarray((size, size), dtype=cp_model.IntVar)
    for i, j in np.ndindex(size, size):
        intermediate_zeroes[i, j] = zero_var = model.NewBoolVar(f"r{i+1}r{j+1}==0")
        zero_expr = grid_vars[i, j, 0]
        model.Add(zero_expr == 1).OnlyEnforceIf(zero_var)
        model.Add(zero_expr != 1).OnlyEnforceIf(zero_var.Not())

        intermediate_missing_row_digits[i, j] = row_var = model.NewBoolVar(f"r{i+1}_missing_{j}")
        row_expr = sum(grid_vars[i, :, j+1])
        model.Add(row_expr == 0).OnlyEnforceIf(row_var)
        model.Add(row_expr != 0).OnlyEnforceIf(row_var.Not())

        intermediate_missing_col_digits[i, j] = col_var = model.NewBoolVar(f"c{i+1}_missing_{j}")
        col_expr = sum(grid_vars_t[i, :, j+1])
        model.Add(col_expr == 0).OnlyEnforceIf(col_var)
        model.Add(col_expr != 0).OnlyEnforceIf(col_var.Not())

    # a zero may only be placed if it indexes another zero by digits replaced in row/column
    for i, j, r, c in np.ndindex(size, size, size, size):
        model.Add(intermediate_zeroes[r, c] == 1).OnlyEnforceIf(intermediate_zeroes[i, j],
                                                                intermediate_missing_row_digits[i, r],
                                                                intermediate_missing_col_digits[j, c])

    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter(grid_vars)

    solver.parameters.enumerate_all_solutions = True
    solver.parameters.cp_model_presolve = True
    solver.Solve(model, solution_printer)


if __name__ == '__main__':
    main()
