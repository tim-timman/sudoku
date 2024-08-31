import itertools
from abc import ABC, abstractmethod
from itertools import chain, combinations
from typing import override, Optional

import numpy as np
import numpy.ma as ma

from .types import (
    index_board,
    options,
    options_sum,
    COLLAPSED,
    Board,
    CellIndex,
    Constraint,
    BrokenConstraintsError,
    invalid_board_dt,
)

# Each array contains board indices that must all contain unique digits.
# Note: Keep the row, column, box constraints in 1st, 2nd, and 3rd positions respectively.
# Hint: Extract the specific row, column, box as follows:
#   row, col, box = standard_constraints[np.isin(standard_constraints, cell_idx).any(axis=1)]
standard_constraints = np.array([
    # Sudoku row constraints
    *index_board.reshape((9, 9)),
    # Sudoku column constraints
    *index_board.reshape((9, 9)).T,
    # Sudoku box constraints
    *chain.from_iterable(index_board.reshape((27, 3))[i::3].reshape((3, 9)) for i in range(3))
])

use_column_with_zero_equals_box_constraint = True
use_row_with_zero_equals_box_constraint = False
use_zero_indexing_constraint = False


def apply_standard_constraints(board: Board, idx: CellIndex):
    for constraint in standard_constraints[np.isin(standard_constraints, idx).any(axis=1)]:
        mask = ma.masked_less(board[constraint], COLLAPSED, copy=False)

        collapsed = ma.sum(mask ^ COLLAPSED)
        board[constraint[mask.mask]] &= ~collapsed

        # If any cell has no (zero) options left.
        if np.any(board[constraint] & ~COLLAPSED == 0):
            raise BrokenConstraintsError


def apply_zero_constraints(board: Board, _: CellIndex):
    """Special global zero constraint

    A column with a zero in it must match the digits of the zero's box.
    """

    collapsed_zero_indices = np.flatnonzero(board == options[0] | COLLAPSED)

    # # At most 5 zeros!
    # if collapsed_zero_indices.size == 5:
    #     board[board < COLLAPSED] &= ~options[0]
    #     if np.any(board & ~COLLAPSED == 0):
    #         raise BrokenConstraintsError

    for cell_idx in collapsed_zero_indices:
        row, col, box = constraints = standard_constraints[np.isin(standard_constraints, cell_idx).any(axis=1)]
        # Find the symmetric difference between box and column;
        #  these must be match.
        if use_column_with_zero_equals_box_constraint:
            box_column_intersection = np.intersect1d(col, box, assume_unique=True)
            box_part = np.setdiff1d(box, box_column_intersection, assume_unique=True)
            column_part = np.setdiff1d(col, box_column_intersection, assume_unique=True)
            box_options = np.bitwise_or.reduce(board[box_part]) | COLLAPSED

            column_options = np.bitwise_or.reduce(board[column_part]) | COLLAPSED
            board[box_part] &= column_options
            board[column_part] &= box_options

        if use_row_with_zero_equals_box_constraint:
            # Find the symmetric difference between box and row;
            #  these must be match.
            box_row_intersection = np.intersect1d(row, box, assume_unique=True)
            box_part = np.setdiff1d(box, box_row_intersection, assume_unique=True)
            row_part = np.setdiff1d(row, box_row_intersection, assume_unique=True)

            box_options = np.bitwise_or.reduce(board[box_part]) | COLLAPSED
            row_options = np.bitwise_or.reduce(board[row_part]) | COLLAPSED

            board[box_part] &= row_options
            board[row_part] &= box_options

        if (
            (use_row_with_zero_equals_box_constraint or use_column_with_zero_equals_box_constraint)
            and np.any(board[constraints] & ~COLLAPSED == 0)
        ):
            raise BrokenConstraintsError

        if use_zero_indexing_constraint:
            # Zero indexing rule:
            #   For col == box: Proven (by exhaustive search) - DOESN'T WORK :(
            # continue
            current_board = np.zeros((), dtype=invalid_board_dt)
            for (idx,), val in ma.ndenumerate(ma.masked_less(board, COLLAPSED, copy=False)):
                current_board[idx // 2] = int(np.log2(val ^ COLLAPSED)) << (idx % 2) * 4

            ma_row_collapsed = ma.masked_less(board[row], COLLAPSED)
            ma_col_collapsed = ma.masked_less(board[col], COLLAPSED)

            row_options = (ma.bitwise_or.reduce(ma_row_collapsed ^ COLLAPSED) ^ options_sum) & ~options[0]
            col_options = (ma.bitwise_or.reduce(ma_col_collapsed ^ COLLAPSED) ^ options_sum) & ~options[0]

            if np.any(ma_row_collapsed.mask) and np.bitwise_count(row_alt := row_options & np.bitwise_or.reduce(board[row[ma_row_collapsed.mask]])) == 1:
                # Special case where one option is strictly missing among remaining alternatives.
                row_indices = (int(np.log2(row_alt) - 1),)
            else:
                row_indices = tuple(map(int, np.log2(options[options & row_options > 0]) - 1))

            if np.any(ma_col_collapsed.mask) and np.bitwise_count(col_alt := col_options & np.bitwise_or.reduce(board[col[ma_col_collapsed.mask]])) == 1:
                # Special case where one option is strictly missing among remaining alternatives.
                col_indices = (int(np.log2(col_alt) - 1),)
            else:
                col_indices = tuple(map(int, np.log2(options[options & row_options > 0]) - 1))

            assert -1 not in (*col_indices, *row_indices)
            if len(row_indices) == 1 and len(col_indices) == 1:
                # Position is completely forced, must be a zero, else constraint is broken.
                idx = row_indices[0] * 9 + col_indices[0]
                if board[idx] & options[0]:
                    board[idx] &= options[0] | COLLAPSED
                else:
                    raise BrokenConstraintsError

            row_forced = row_options
            col_forced = col_options
            for ridx, cidx in itertools.product(row_indices, col_indices):
                # Aggregate valid zero indexed positions, to reduce possibility space.
                idx = ridx * 9 + cidx
                if board[idx] & options[0]:
                    # If the indexed cell can contain zero, it doesn't force the given row/col digits
                    row_forced &= ~options[ridx + 1] | COLLAPSED
                    col_forced &= ~options[cidx + 1] | COLLAPSED

            # Remove any of the alternatives that may not be present.
            if row_forced:
                board[row[ma_row_collapsed.mask]] &= ~row_forced | COLLAPSED
            if col_forced:
                board[col[ma_col_collapsed.mask]] &= ~col_forced | COLLAPSED

            if np.any(board[[row, col]] & ~COLLAPSED == 0):
                raise BrokenConstraintsError

            next_board = np.zeros((), dtype=invalid_board_dt)
            for (idx,), val in ma.ndenumerate(ma.masked_less(board, COLLAPSED, copy=False)):
                next_board[idx // 2] = int(np.log2(val ^ COLLAPSED)) << (idx % 2) * 4

            assert (current_board & next_board == current_board).all()


# Note: order matters
constraints_list: list[Constraint] = [
    apply_standard_constraints,
    apply_zero_constraints,
]


def assert_board_validity(board: Board):
    # Unique set constraints
    for constraint in standard_constraints:
        assert np.all(np.unique(board[constraint], return_counts=True)[1] == 1)

    # Zero constraint
    for cell_idx in np.flatnonzero(board == options[0] | COLLAPSED):
        row, col, box = standard_constraints[np.isin(standard_constraints, cell_idx).any(axis=1)]
        if use_column_with_zero_equals_box_constraint:
            assert np.intersect1d(board[col], board[box]).shape[0] == col.shape[0]
        if use_row_with_zero_equals_box_constraint:
            assert np.intersect1d(board[row], board[box]).shape[0] == row.shape[0]
        if use_zero_indexing_constraint:
            raise NotImplementedError()

    # @TODO: extra constraints...


class ConstraintABC(ABC):
    def __init__(self):
        self.cell_indices: set[int] = set()

    def collect(self, cell_idx: int):
        self.cell_indices.add(cell_idx)

    @abstractmethod
    def apply(self, board: Board): ...


class Default(ConstraintABC):
    def collect(self, cell_idx: int):
        pass

    def apply(self, board: Board):
        pass


class Options(ConstraintABC):
    def __init__(self, *digits: int):
        super().__init__()
        assert all(0 <= o <= 9 for o in digits)
        self.value = sum(options[o] for o in set(digits))

    def apply(self, board: Board):
        board[list(self.cell_indices)] &= self.value


def Even():
    return Options(*range(0, 10, 2))


def Odd():
    return Options(*range(1, 10, 2))


class Unique(ConstraintABC):
    @override
    def collect(self, cell_idx: int):
        super().collect(cell_idx)
        if len(self.cell_indices) > 10:
            raise ValueError(f"too many cells part of same {self.__class__.__name__} constraint; impossible")

    def constraint(self, board: Board, idx: CellIndex):
        cell_indices = self._cell_indices
        if not np.isin(cell_indices, idx).any():
            return
        ma_collapsed: ma.MaskedArray = ma.masked_less(board[cell_indices], COLLAPSED, copy=False)
        assert ma_collapsed.count() == ma.unique(ma_collapsed).count()
        collapsed = ma.sum(ma_collapsed ^ COLLAPSED)
        board[cell_indices[ma_collapsed.mask]] &= ~collapsed

        # If any cell has no (zero) options left.
        if np.any(board[cell_indices] & ~COLLAPSED == 0):
            raise BrokenConstraintsError

    def apply(self, board: Board):
        self._cell_indices = np.array(list(self.cell_indices), dtype=np.uint16)
        constraints_list.insert(-1, self.constraint)


class Cage(Unique):
    def __init__(self, total: Optional[int] = None):
        super().__init__()
        self.total = total
        self.options = None

    def constraint(self, board: Board, idx: CellIndex):
        cell_indices = self._cell_indices
        if not np.isin(cell_indices, idx).any():
            return

        ma_collapsed: ma.MaskedArray = ma.masked_less(board[cell_indices], COLLAPSED, copy=False)
        assert ma_collapsed.count() == ma.unique(ma_collapsed).count()
        collapsed = ma.sum(ma_collapsed ^ COLLAPSED)
        board[cell_indices[ma_collapsed.mask]] &= ~collapsed

        # If any cell has no (zero) options left.
        if np.any(board[cell_indices] & ~COLLAPSED == 0):
            raise BrokenConstraintsError

        if self.total is None:
            return

        sum_options = np.bitwise_or.reduce(self.options[self.options & collapsed > 0] & ~collapsed)
        if not sum_options:
            raise BrokenConstraintsError

        # @Note: this doesn't fully remove all options. Options invalid because
        #  of the combinations of cell options adding to the sum are not removed.
        board[cell_indices[ma_collapsed.mask]] &= sum_options
        if not np.any(self.options & np.bitwise_or.reduce(board[cell_indices]) == self.options):
            raise BrokenConstraintsError

    def apply(self, board: Board):
        super().apply(board)
        if self.total is None:
            return

        num = len(self.cell_indices)
        v = tuple(range(10))
        if not (sum(v[:num]) <= self.total <= sum(v[-num:])):
            raise ValueError(f"the sum {self.total} is impossible in a cage with {num} cells")

        self.options = np.array([np.sum(options.take(x)) for x in combinations(v, num) if sum(x) == self.total], dtype=np.uint16)
        board[self._cell_indices] &= np.bitwise_or.reduce(self.options)
