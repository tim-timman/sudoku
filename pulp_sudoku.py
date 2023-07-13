import pulp

# All rows, columns and values within a Sudoku take values from 1 to 9
VALS = ROWS = COLS = range(1, 10)

# The boxes list is created, with the row and column index of each square in each box
Boxes = [
    [(3 * i + k + 1, 3 * j + l + 1) for k in range(3) for l in range(3)]
    for i in range(3)
    for j in range(3)
]

prob = pulp.LpProblem("sudoku")
prob.setSolver(type(pulp.LpSolverDefault)(msg=False))  # We want a quiet output

# The decision variables are created
choices = pulp.LpVariable.dicts("grid_vars", (VALS, ROWS, COLS), cat="Binary")

# We do not define an objective function since none is needed

# A constraint ensuring that only one value can be in each square is created
for r in ROWS:
    for c in COLS:
        prob += pulp.lpSum([choices[v][r][c] for v in VALS]) == 1

# The row, column and box constraints are added for each value
for v in VALS:
    for r in ROWS:
        prob += pulp.lpSum([choices[v][r][c] for c in COLS]) == 1

    for c in COLS:
        prob += pulp.lpSum([choices[v][r][c] for r in ROWS]) == 1

    for b in Boxes:
        prob += pulp.lpSum([choices[v][r][c] for (r, c) in b]) == 1

# The starting numbers are entered as constraints
input_data = [
    (5, 1, 1), (6, 2, 1), (8, 4, 1),
    (4, 5, 1), (7, 6, 1), (3, 1, 2),
    (9, 3, 2), (6, 7, 2), (8, 3, 3),
    (1, 2, 4), (8, 5, 4), (4, 8, 4),
    (7, 1, 5), (9, 2, 5), (6, 4, 5),
    (2, 6, 5), (1, 8, 5), (8, 9, 5),
    (5, 2, 6), (3, 5, 6), (9, 8, 6),
    (2, 7, 7), (6, 3, 8), (8, 7, 8),
    (7, 9, 8), (3, 4, 9),
    # Since the previous Sudoku contains only one unique solution, we remove some numers from the board to obtain a
    # Sudoku with multiple solutions
    #     (1, 5, 9),
    #     (6, 6, 9),
    #     (5, 8, 9)
]

for v, r, c in input_data:
    var = choices[v][r][c]
    var.setInitialValue(1)
    var.fixValue()


def solve():
    prob.solve()


def print_solutions():
    global prob
    while True:
        prob.solve()
        # The status of the solution is printed to the screen
        print("Status:", pulp.LpStatus[prob.status])
        # The solution is printed if it was deemed "optimal" i.e met the constraints
        if pulp.LpStatus[prob.status] == "Optimal":
            # The solution is written to the sudokuout.txt file
            for r in ROWS:
                if r in [1, 4, 7]:
                    print("+-------+-------+-------+")
                for c in COLS:
                    for v in VALS:
                        if pulp.value(choices[v][r][c]) == 1:
                            if c in [1, 4, 7]:
                                print("| ", end="")
                            print(str(v) + " ", end="")
                            if c == 9:
                                print("|")
            print("+-------+-------+-------+\n")
            print(f"Took {prob.solutionTime:.2f} seconds")
            # The constraint is added that the same solution cannot be returned again
            prob += (
                pulp.lpSum(
                    [
                        choices[v][r][c]
                        for v in VALS
                        for r in ROWS
                        for c in COLS
                        if pulp.value(choices[v][r][c]) == 1
                    ]
                )
                <= 80
            )
        # If a new optimal solution cannot be found, we end the program
        else:
            break


if __name__ == "__main__":
    print_solutions()
