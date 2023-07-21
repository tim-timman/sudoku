import itertools
import math
import time

from ortools.sat.python import cp_model
import numpy as np

_ = None
BOARD = np.array([[_, _, _, _, 0, _, _, _, _],
                  [_, _, 0, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, 5],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, 5, _, _, _, 4],
                  [_, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, 3]])


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, grid_vars):
        super().__init__()
        self.grid_vars = grid_vars
        self.offset = 1 if grid_vars.shape[2] == 9 else 0
        self.prev_time = time.perf_counter_ns()

    def on_solution_callback(self):
        t = time.perf_counter_ns() - self.prev_time
        units = ["µs", "ms", "s"]
        unit = "ns"
        while t > 1e3:
            try:
                unit = units.pop(0)
            except IndexError:
                break
            t = t / 1e3

        print(f"Took: {t:.2f} {unit}\n")
        for row in self.grid_vars:
            print([d + self.offset for col in row for d, v in enumerate(col) if self.Value(v)])
        self.prev_time = time.perf_counter_ns()
        self.StopSearch()


USE_KILLER = True
USE_KNIGHTS_MOVE = False
USE_ZERO = True
USE_EVENS = True


def main(board=None):
    if board is None:
        board = BOARD

    model = cp_model.CpModel()
    size = board.shape[0]
    offset = 0 if USE_ZERO else 1
    digits = 10 if USE_ZERO else 9

    # construct the variables for the grid
    grid_vars = np.array([[[model.NewBoolVar(f"r{i+1}c{j+1}d{k+offset}")
                            for k in range(digits)]
                           for j in range(size)]
                          for i in range(size)])
    grid_vars_t = grid_vars.transpose(1, 0, 2)

    for i, j in np.ndindex(size, size):
        given = board[i, j]
        if given is not None:
            model.Add(grid_vars[i, j, given] == 1)

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

    if USE_ZERO:
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

        # the digit missing in a column with a zero must be the same as in the zero's box
        # @Note: this constraint makes it take forever to compute.
        column_box_constraint = False
        if column_box_constraint:
            intermediate_missing_box_digits = np.ndarray((size, size), dtype=cp_model.IntVar)
            intermediate_box_col_missing_same = np.ndarray((size, size), dtype=cp_model.IntVar)

            for i, j in np.ndindex(size, size):
                intermediate_missing_box_digits[i, j] = box_var = model.NewBoolVar(f"b{i + 1}_missing_{j}")
                brow = i // 3 * 3
                bcol = i % 3 * 3
                box_expr = sum(grid_vars[brow:brow + 3, bcol:bcol + 3, j + 1].flat)
                model.Add(box_expr == 0).OnlyEnforceIf(box_var)
                model.Add(box_expr != 0).OnlyEnforceIf(box_var.Not())

            for i, j in np.ndindex(size, size):
                bcol = i % 3 * 3
                if bcol <= j < bcol + 3:
                    intermediate_box_col_missing_same[i, j] = box_col_var = model.NewBoolVar(f"b{i + 1} == c{j + 1}")
                    box_sum = sum(intermediate_missing_box_digits[i, k] * (k + 1) for k in range(9))
                    col_sum = sum(intermediate_missing_col_digits[j, k] * (k + 1) for k in range(9))
                    model.Add(box_sum == col_sum).OnlyEnforceIf(box_col_var)
                    model.Add(box_sum != col_sum).OnlyEnforceIf(box_col_var.Not())

            for i, j in np.ndindex(size, size):
                box_idx = i // 3 * 3 + (j // 3) % 3
                model.Add(intermediate_box_col_missing_same[box_idx, j] == 1).OnlyEnforceIf(intermediate_zeroes[i, j])

    if USE_EVENS:
        # row/column coordinates
        evens = np.array([(1, 2),
                          (2, 1),
                          (3, 1),
                          (2, 3),
                          (3, 3),
                          (4, 2),

                          (2, 4),
                          (2, 5),
                          (2, 6),
        ]) - 1
        for i, j in evens:
            model.Add(sum(grid_vars[i, j, k] for k in range(digits) if k % 2 == 0) == 1)

    if USE_KNIGHTS_MOVE:
        knights_digits = np.array(range(digits))
        intermediates = np.ndarray((size, size, digits), dtype=cp_model.IntVar)
        for i, j, k in np.ndindex(*intermediates.shape):
            intermediates[i, j, k] = var = model.NewBoolVar(f"r{i+1}c{j+1}=={k+offset}")
            var_expr = grid_vars[i, j, k]
            model.Add(var_expr == 1).OnlyEnforceIf(var)
            model.Add(var_expr != 1).OnlyEnforceIf(var.Not())

        # knights move
        wraparound = False

        intermediate_digit_in_knights_move = np.ndarray((size, size, digits), dtype=cp_model.IntVar)
        knight_offsets = list((x, y) for x, y in itertools.product(list(range(-2, 3)), repeat=2) if abs(x) + abs(y) == 3)
        for i, j, k in np.ndindex(size, size, digits):
            intermediate_digit_in_knights_move[i, j, k] = var = model.NewBoolVar(f"knights_of_r{i+1}c{j+1}_is_{k+offset}")
            var_expr = sum(intermediates[(i + x) % size, (j + y) % size, k]
                           for x, y in knight_offsets
                           if wraparound or (0 <= (i + x) < size and 0 <= (j + y) < size))
            model.Add(var_expr >= 1).OnlyEnforceIf(var)
            model.Add(var_expr < 1).OnlyEnforceIf(var.Not())

        for i, j in np.ndindex(size, size):
            for k in knights_digits:
                model.Add(intermediates[i, j, k] != 1)\
                    .OnlyEnforceIf(intermediate_digit_in_knights_move[i, j, k])

    if USE_KILLER:
        killers = [
            (grid_vars[0, 3:6, :], 7),
        ]
        for idx, (cage, total) in enumerate(killers):
            # digits may not repeat in a cage
            cage_digits = cage.reshape(-1, digits).T
            for d in cage_digits:
                model.AddAtMostOne(d)

            if total is not None:
                # digits in cage must sum to given total
                sums = sum(i * sum(k) for i, k in enumerate(cage_digits))
                model.Add(sums == total)

    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter(grid_vars)

    solver.parameters.enumerate_all_solutions = True
    solver.parameters.cp_model_presolve = True
    solver.Solve(model, solution_printer)
    print(solver.StatusName())


if __name__ == '__main__':
    main()
