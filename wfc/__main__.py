import numpy as np

from . import solver

board_template = """
    I   I   I   |   I   *   *   |   *   *   * 
    *   *   I   |   J   J   J   |   J   J   * 
    *   Ie  *   |   J   J   J   |   J   J   * 
    ––––––––––––––––––––––––––––––––––––––––––
    Ie  I   Ie  |   IA  A   A   |   e   *   e
    e   *   e   |   *   *   *   |   *   *   * 
    *   e   *   |   B   B   B   |   *   *   * 
    ––––––––––––––––––––––––––––––––––––––––––
    *   *   *   |   *   O   *   |   *   *   * 
    *   *   *   |   *   K   K   |   K   *   * 
    *   *   *   |   *   K   K   |   K   *   * 
"""


# @TODO: parse from visual and simple setting of constraints
# constraints = {
#     "A": Sum(9) & Unique(),
#     "B": Sum(8) & Unique(),
#     "e": Even(),
#     "O": OneOf(*range(1, 10)),
#     "*": OneOf(*range(0, 10)),
#     "I": AllOf(*range(0, 10)),
#     "J": AllOf(*range(0, 10)),
#     "K": Unique(),
# }


if __name__ == "__main__":
    board: solver.Board = np.full((), sum(solver.options), solver.board_dt)
    solver.main(board)
