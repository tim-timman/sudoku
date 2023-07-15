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

    # vars to be able to constrain the value of a row/column
    missing_row_vals = [model.NewIntVar(0, digits, f"not_in_r{i+1}") for i in range(size)]
    missing_col_vals = [model.NewIntVar(0, digits, f"not_in_c{i+1}") for i in range(size)]

    for i in range(size):
        row_var = missing_row_vals[i]
        col_var = missing_col_vals[i]
        model.Add(row_var == 45 - sum(sum(grid_vars[i, :, k]) * k for k in range(digits)))
        model.Add(col_var == 45 - sum(sum(grid_vars_t[i, :, k]) * k for k in range(digits)))

    missing_row_bools = np.array([[model.NewBoolVar(f"r{i+1}={k}")
                                   for k in range(digits)]
                                  for i in range(size)])
    missing_col_bools = np.array([[model.NewBoolVar(f"c{i + 1}={k}")
                                   for k in range(digits)]
                                  for i in range(size)])

    for i, k in np.ndindex(size, digits):
        row_bool = missing_row_bools[i, k]
        model.Add(missing_row_vals[i] == k).OnlyEnforceIf(row_bool)
        model.Add(missing_row_vals[i] != k).OnlyEnforceIf(row_bool.Not())

        col_bool = missing_col_bools[i, k]
        model.Add(missing_col_vals[i] == k).OnlyEnforceIf(col_bool)
        model.Add(missing_col_vals[i] != k).OnlyEnforceIf(col_bool.Not())

    for i in range(size):
        model.Add(sum(missing_row_bools[i]) == 1)
        model.Add(sum(missing_col_bools[i]) == 1)

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

    for i, j in np.ndindex(size, size):
        model.Add(grid_vars[i, j, 0] == 1).OnlyEnforceIf(
            sum(grid_vars[k, l, 0] == 1
                for k in range(1, digits)
                for l in range(1, digits)
                if missing_row_bools[i, k] == 1
                and missing_col_bools[j, l] == 1
                ) >= 1,
        )

    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter(grid_vars)

    solver.parameters.enumerate_all_solutions = True
    solver.parameters.cp_model_presolve = True
    solver.Solve(model, solution_printer)


if __name__ == '__main__':
    main()
