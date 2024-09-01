import re
import signal
from typing import Iterable

import numpy as np

from . import solver
from .constraints import ConstraintABC, Default, Even, NumberOfZeros, Options, Cage, Unique, COLLAPSED

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
    *   *   1   |   *   *   *   |   *   *   * 
    *   6   *   |   *   *   *   |   *   *   * 
    ------------------------------------------
    2   9   0   |   5   A   A   |   e   *   e
    e   *   e   |   *   *   *   |   *   *   * 
    *   e   *   |   B   B   B   |   *   *   * 
    ------------------------------------------
    *   *   *   |   *   *   *   |   *   *   * 
    *   *   *   |   *   *   *   |   *   *   * 
    *   *   *   |   *   *   *   |   *   *   * 
"""
board_template = blank_board_template
constraints_map = {
    **{str(i): Options(i) for i in range(10)},
    "A": Options(1,3),
    "B": Cage(total=8),
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


def solve():
    solver.main(board, seed=35354952237464344499561046517502448174) #, quiet=True)
    solver.main(board, seed=32044001236707542926087009531794486745) #, quiet=True)


while True:
    signal.alarm(10)
    try:
        # solve()
        solver.main(board)
    except Retry:
        print("restarting search...")
        continue
    break
