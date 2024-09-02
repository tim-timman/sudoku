import re
import signal
from typing import Iterable

import numpy as np

from . import solver
from .constraints import ConstraintABC, Default, Even, MustContainUnique, NumberOfZeros, Options, Cage, Unique, COLLAPSED

blank_board_template = """\
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
"""

board_template = """
    4   8   3   |   7   *   *   |   *   *   * 
    *   *   1   |   J   J   J   |   J   J   * 
    *   6   *   |   J   J   J   |   J   J   * 
    ------------------------------------------
    2e  9   0e  |   5A  A   A   |   e   *   e
    e   *   e   |   *   *   *   |   *   *   * 
    *   e   *   |   B   B   B   |   *   *   * 
    ------------------------------------------
    *   *   *   |   *   O   *   |   *   *   * 
    *   *   *   |   *   K   K   |   K   *   * 
    *   *   *   |   *   K   K   |   K   *   3 
"""

board_template = """
    4   8   3   |   7   *   *   |   *   *   * 
    *   *   1   |   9u  5u  8u  |   2u  6u  * 
    *   6   *   |   0u  4u  3u  |   7u  1u  * 
    ------------------------------------------
    2   9   0   |   5   A   A   |   e   *   e
    e   *   e   |   *   *   *   |   *   *   * 
    *   e   *   |   B   B   B   |   *   *   * 
    ------------------------------------------
    *   *   *   |   a   z   *   |   *   *   * 
    *   *   *   |   8   z   y   |   y   *   * 
    *   *   *   |   a   *   y   |   y   x   x 
"""
# board_template = blank_board_template
constraints_map = {
    **{str(i): Options(i) for i in range(10)},
    "A": Options(1, 3),
    "B": Cage(total=8),
    "O": NumberOfZeros(5),
    "u": Unique(),
    "x": Cage(total=9),
    "y": Cage(total=18),
    "z": Options(2,6),
    "a": Options(1, 3),
    "e": Even(),
    "*": Default(),
}


def parse_board(template: str) -> solver.Board:
    cells = list(filter(None, re.split(r"[ |\n-]", template)))
    if len(cells) != 81:
        print(f"Incorrect number of cells. Expected 81, got {len(cells)}")
        exit(1)

    constraints_to_apply: list[ConstraintABC] = list()
    for idx, tokens in enumerate(cells):
        for token in tokens:
            constraint = constraints_map[token]
            for c in constraint if isinstance(constraint, Iterable) else (constraint,):
                if c not in constraints_to_apply:
                    constraints_to_apply.append(c)
                c.collect(idx)

    solver_board: solver.Board = np.full((), sum(solver.options), solver.board_dt)

    for c in constraints_to_apply:
        c.apply(solver_board)

    return solver_board


class Retry(Exception):
    pass


def alarm(*_):
    raise Retry


def debug_break(*_):
    signal.alarm(0)
    breakpoint()


signal.signal(signal.SIGUSR1, debug_break)
signal.signal(signal.SIGALRM, alarm)

board = parse_board(board_template)

while True:
    signal.alarm(10)
    try:
        # @FIXME: Something's bugged with "regular" constraints see seed=5595725474079879758229220073552776793
        #  Probably the Cage constraint...
        solver.main(board) #, debug=True)
    except Retry:
        print("restarting search...")
        continue
    break
