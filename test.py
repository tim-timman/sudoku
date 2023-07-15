from ortools.sat.python import cp_model

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables, n_vars, n_values):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__nvars = n_vars
        self.__nvalues = n_values
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        for i in range(self.__nvars):
            for j in range(self.__nvalues):
                if self.BooleanValue(self.__variables[i, j]):
                    print('{} '.format(j), end=' ')
        print()
        self.StopSearch()

    def solution_count(self):
        return self.__solution_count
        # [END print_solution]

model = cp_model.CpModel()

num_positions = 6
num_digits = 10
sum_target = 15

positions = {}
for i in range(num_positions):
    for j in range(num_digits):
        positions[i, j] = model.NewBoolVar('var_{}_eq_{}'.format(i, j))

# Var consistency

for i in range(num_positions):
    model.Add(sum(positions[i, j] for j in range(num_digits)) == 1)

count_equal_to = []
for j in range(num_digits):
    var = model.NewIntVar(0, num_positions, 'equal_to_{}'.format(j))
    count_equal_to.append(var)
    model.Add(var == sum(positions[i, j] for i in range(num_positions)))

sum_is_3 = []
for j in range(num_digits):
    lit = model.NewBoolVar('sum_of_{}_is_3'.format(j))
    model.Add(count_equal_to[j] == 3).OnlyEnforceIf(lit)
    model.Add(count_equal_to[j] != 3).OnlyEnforceIf(lit.Not())
    sum_is_3.append(lit)

model.Add(sum(sum_is_3) == 1)

model.Add(sum(count_equal_to[j] * j for j in range(num_digits)) == sum_target)

solver = cp_model.CpSolver()
cb = SolutionPrinter(positions, num_positions, num_digits)
solver.SearchForAllSolutions(model, cb)
