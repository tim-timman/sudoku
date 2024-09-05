import re
import signal
from typing import Iterable

import numpy as np

from . import solver
from .types import NoSolutions
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
    *   *   1   |   9   5   8   |   2   6   * 
    *   6   *   |   0   4   3   |   7   1   * 
    ------------------------------------------
    2   9   0   |   5   3   1   |   e   *   e
    e   *   e   |   4   *   *   |   *   *   * 
    *   4   *   |   *   0   *   |   *   *   *
    ------------------------------------------
    *   *   b   |   *   *   *   |   *   *   * 
    *   *   b   |   z   z   *   |   *   *   * 
    3   0   b   |   z   z   y   |   y   y   *
"""
# board_template = blank_board_template
constraints_map = {
    **{str(i): Options(i) for i in range(10)},
    "O": NumberOfZeros(5),
    "b": Default(), # Options(2, 4, 6),
    "d": Options(3,4),
    "y": Default(), # Cage(total=12),
    "z": Default(), #Cage(total=24),
    "e": Even(),
    "*": Default(),
    "c": MustContainUnique(3),
    "g": MustContainUnique(0),
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

i = 0
while True:
    signal.alarm(10)
    try:
        # @FIXME: Something's bugged with "regular" constraints see seed=5595725474079879758229220073552776793
        #  Probably the Cage constraint... nope, buggy even without Cage constraints, maybe not that seed though.
        solver.main(board) #, debug=True)
    except Retry:
        print("restarting search...")
        continue
    except (NoSolutions, AssertionError):
        # Workaround the bugs... just try a couple of extra times (because there probably are solutions) :P
        #  reset the invalid boards on "NoSolutions" because they may be invalid :sweat-smile:
        solver.invalid_boards_idx = -1
        i += 1
        if i <= 25:
            continue
    break
