import re
import signal
from typing import Iterable

import numpy as np

from . import solver
from .constraints import ConstraintABC, Default, Even, Options, Cage, Unique, COLLAPSED

board_template = """
    4   8   3   |   7   *   *   |   *   *   * 
    *   *   1   |   J   J   J   |   J   J   * 
    *   6   *   |   J   J   J   |   J   J   * 
    ------------------------------------------
    2e  9   0e  |   5A  A   A   |   e   *   e
    e   *   e   |   *   *   *   |   *   *   * 
    *   e   *   |   B   B   B   |   *   *   * 
    ------------------------------------------
    *   *   *   |   *   5   *   |   *   *   * 
    *   *   *   |   *   K   K   |   K   *   * 
    *   *   *   |   *   K   K   |   K   *   3 
"""

constraints_map = {
    **{str(i): Options(i) for i in range(10)},
    "A": Options(1,3,5),
    "B": Cage(total=8),
    "e": Even(),
    # "O": Options(*range(1, 10)),
    "*": Default(),
    "I": Unique(),
    "J": Unique(),
    "K": Unique(),
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


signal.signal(signal.SIGUSR1, lambda *_: breakpoint())
signal.signal(signal.SIGALRM, alarm)

board = parse_board(board_template)


def solve():
    solver.main(board, seed=333994325917206120681495443532316063120, quiet=True)


while True:
    signal.alarm(2)
    try:
        solver.main(board)
    except Retry:
        print("restarting search...")
        continue
    break
