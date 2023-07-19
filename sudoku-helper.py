#!/usr/bin/env python3.11
import atexit
import argparse
import functools
from collections import defaultdict
from contextlib import suppress
from functools import reduce, partial
from functools import total_ordering
import itertools
from itertools import chain, combinations, repeat, count, zip_longest
from pprint import pp
import operator
from operator import itemgetter, methodcaller
import re
from textwrap import dedent, indent
from typing import Callable, Iterable, Literal, get_args, Iterator

import rich.color
from rich import print
from rich.color import Color
from rich.table import Table


@total_ordering
class SudokuSet(frozenset):
    _selves = {}

    @functools.cache
    def __new__(cls, *args, **kwargs):
        _hash = sum((1 << 9) >> x for x in args[0]) if args else 0
        if self := cls._selves.get(_hash, None):
            return self
        self = super(SudokuSet, cls).__new__(cls, *args, **kwargs)
        self._hash = _hash
        cls._selves[_hash] = self
        self._str = sub_color("".join([f" {v} " for v in sorted(args[0])])) if args else ""
        return self

    def __hash__(self):
        # Switch order for easy comparison
        return self._hash

    def __str__(self):
        return self._str

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplemented
        return self.__hash__() > other.__hash__()


colors = {str(k): f"bold grey3 on {v}" for k, v in zip(count(1), [
    "#005F73",
    "#0A9396",
    "#94D2BD",
    "#E9D8A6",
    "#EE9B00",
    "#CA6702",
    "#BB3E03",
    "#AE2012",
    "#9B2226",
])}


def sub_color(text):
    return re.sub(r"\s(\d)\s", lambda s: f"[{colors[s[1]]}]{s[0]}[/]", text)


class Args(argparse.Namespace):
    subparser: str | None
    sums: list[int]
    num_digits: list[int]
    digit_set: list[int]
    include_digits: list[tuple[tuple[int, int], list[int]]]
    exclude_digits: list[tuple[tuple[int, int], list[int]]]
    num_odd: int | None
    num_even: int | None
    mask: list[tuple[tuple[int, int], list[int]]]
    nop: bool


Alternatives = Iterable[list[SudokuSet[int]]]


def default_nop(fn):
    fn.__nop__ = True
    return fn


def merge(a, b) -> Alternatives:
    a = list(a)
    b = list(b)
    if not b:
        return a
    return [[*x, *y] for x in a for y in b
            if x or y]


def exclusive(a: Alternatives, b: Alternatives, args: Args) -> Alternatives:
    """keep alternatives where the joint set are all unique"""
    rows = list(merge(a, b))
    masks = create_index_masks(args.mask, len(rows[0]))

    def sets_disjoint(row):
        sets = build_sets_from_index_mask(masks, row)
        return bool(reduce(methodoperator("isdisjoint"), sets))

    return (x for x in rows if sets_disjoint(x))


def same_sum(a: Alternatives, b: Alternatives) -> Alternatives:
    """keep alternatives where the sum of a's is the same as b's"""
    temp = [(sum(chain(*x)), x) for x in b]
    return ([*x, *y] for x in a for sy, y in temp
            if x != y and sum(chain(*x)) == sy)


def get_number_mask(args_mask) -> Iterator[int]:
    return chain.from_iterable(m[1] for m in chain(args_mask))


@default_nop
def drop(a: Alternatives, b: Alternatives, args: Args) -> Alternatives:
    """drop columns according to mask"""
    mask = get_number_mask(args.mask)
    return zip(*itertools.compress(zip(*merge(a, b)), chain(mask, repeat(1))))


@default_nop
def permute(a: Alternatives, b: Alternatives, args: Args) -> Alternatives:
    """permute columns using mask to denote new index"""
    mask = get_number_mask(args.mask)
    m = 0
    ret = defaultdict(list)
    for val in zip(*merge(a, b)):
        m = next(mask, m)
        ret[m].insert(m, val)

    return zip(*chain.from_iterable(x[1] for x in sorted(ret.items())))


def create_index_masks(args_mask, length):
    num_sets = list(range(length))
    if not args_mask:
        mask = (num_sets[:-1], num_sets[-1:])
    else:
        mask_dict = defaultdict(list)
        for val in get_number_mask(args_mask):
            try:
                idx = num_sets.pop(0)
            except IndexError:
                break
            if val == -1:
                continue
            mask_dict[val].append(idx)
        mask = (*mask_dict.values(), num_sets)
    mask = list(filter(lambda x: x, mask))
    return mask


def build_sets_from_index_mask(masks, row):
    return list((reduce(operator.or_, (row[i] for i in m)) for m in masks))  # union


def methodoperator(name):
    def caller(lhs, rhs):
        return methodcaller(name, rhs)(lhs)
    return caller


def overlap(a: Alternatives, b: Alternatives, args: Args) -> Alternatives:
    """\
    mark columns as sets by index and number and filter if they don't overlap

    ex. 3 columns. ... -- OVER -m 101  will filter rows where the set of the
    1st and 3rd column doesn't have an overlap with the 2nd
    """
    rows = list(merge(a, b))
    masks = create_index_masks(args.mask, len(rows[0]))

    def sets_overlap(row):
        sets = build_sets_from_index_mask(masks, row)
        return bool(reduce(methodoperator("intersection"), sets))

    return (x for x in rows if sets_overlap(x))


def add(a: Alternatives, b: Alternatives) -> Alternatives:
    """just add a column"""
    return merge(a, b)


operators: dict[str, Callable[[Alternatives, Alternatives], Alternatives]] = {
    "X": exclusive,
    "SUM": same_sum,
    "DROP": drop,
    "OVER": overlap,
    "PER": permute,
    "ADD": add,
}


class SetAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, [*getattr(namespace, self.dest, []), *values])


def set_type(string: str):
    match = re.match(r"(?:(\d(?!\d))?-?(\d)?:)?([\dx]+)$", string)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"{string!r} invalid syntax, must be [d-d:]XXXX "
            f"ex. 1-2:12345 => include 1 or 2 of set {{1,2,3,4,5}}")
    digits = [int(x) if x.isdigit() else -1 for x in match[3]]
    start = match[1]
    end = match[2]

    if not start and not end:
        start = end = len(digits)
    if not start:
        start = 0
    if not end:
        end = len(digits)
    return (int(start), int(end)), digits


notes = []


def defer_note(depth, msg=None):
    if not notes:
        atexit.register(defer_note, -1)

    if depth >= 0:
        notes.append((depth, f"\n  [red]Note: [/]{msg}\n"))
        return

    d, note = notes.pop()
    while True:
        print(note)
        with suppress(IndexError):
            temp, note = notes.pop()
            if temp < d:
                return
        return


def main(prev_alternatives: Alternatives = None, raw_args: list[str] = None, depth=0) -> Alternatives:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sums", type=int, nargs="*", metavar="SUM",
                        help="sums to match of given digits (DEFAULT: %(default)s)")
    parser.add_argument("-n", "--num-digits", type=int, nargs="*", metavar="NUM",
                        action="extend",
                        help="number of digits to match (DEFAULT: %(default)s)")
    parser.add_argument("--nop", action=argparse.BooleanOptionalAction, default=False,
                        help="nop adding digits. default depends on command")
    parser.add_argument("-g", "--digit-set", type=int, nargs="*", metavar="DIGIT",
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help="starting options (DEFAULT: %(default)s)")
    parser.add_argument("-i", "--include-digits", type=set_type, nargs="*", metavar="DIGIT",
                        action=SetAction, default=[],
                        help="digits that must be included, may be repeated to specify that "
                             "at least one of the sets must be included (DEFAULT: %(default)s)")
    parser.add_argument("-x", "--exclude-digits", type=set_type, nargs="*", metavar="DIGIT",
                        action=SetAction, default=[],
                        help="digits the must be excluded, may be repeated to specify that "
                             "at least one of the sets must be excluded (DEFAULT: %(default)s)")
    parser.add_argument("-o", "--num-odd", type=int, metavar="NUM",
                        help="the minimum number of odd number to match")
    parser.add_argument("-e", "--num-even", type=int, metavar="NUM",
                        help="the minimum number of even number to match")
    parser.add_argument("-m", "--mask", type=set_type, nargs="*", action=SetAction, default=[],
                        help="a mask for various commands (DEFAULT: %(default)s)")

    parser.add_argument("--", dest="bork", choices=operators.keys(), nargs="*",
                        default=argparse.SUPPRESS,
                        help=f"chain a new set of constraints for match against. "
                             f"eg. `{parser.prog} -n 3 -s 17 -- X -n 3 -s 7`")

    parser.usage = parser.format_usage() + "\nCOMMANDS\n" + "\n".join([f' => {k}:\n{indent(dedent(v.__doc__), "  ")}'
                                                                       for k, v in operators.items()])

    arg_overrides = {}
    op_func = None
    if raw_args is not None:
        if raw_args:
            if (sep := raw_args.pop(0)) != "--":
                parser.error(f"wrong separator: {sep!r}")
            if not raw_args:
                defer_note(depth - 1, "missing chain operator")
                return prev_alternatives
            if (op_func := operators.get(op := raw_args.pop(0).upper(), None)) is None:
                parser.error(f"invalid chain operator: {op!r}")
            if hasattr(op_func, "__nop__"):
                arg_overrides["nop"] = True
        else:
            return prev_alternatives

    args: Args
    rest: list[str]
    args, rest = parser.parse_known_args(raw_args, namespace=Args(**arg_overrides))

    options = set(args.digit_set) if not args.nop else set()

    sums: defaultdict[int, set[SudokuSet[int]]] = defaultdict(set)

    for i in range(1, len(options) + 1):
        for combination in combinations(options, i):
            sums[sum(combination)].add(SudokuSet(combination))

    if args.sums:
        tmp = (v for s in args.sums for v in sums[s])
    else:
        tmp = (v for _, vs in sums.items() for v in vs)

    if args.num_digits is not None:
        n = args.num_digits
        tmp = (x for x in tmp if len(x) in n)

    if args.include_digits:
        tmp = (x for x in tmp if any(start <= len(x.intersection(d)) <= end
                                     for (start, end), d in args.include_digits))
    if args.exclude_digits:
        tmp = (x for x in tmp if all(start <= len(d) - len(x.intersection(d)) <= end
                                     for (start, end), d in args.exclude_digits))

    if args.num_even:
        tmp = (x for x in tmp if sum(map(lambda y: y % 2 == 0, x)) >= args.num_even)

    if args.num_odd:
        tmp = (x for x in tmp if sum(map(lambda y: y % 2 == 1, x)) >= args.num_odd)

    alternatives = list([x] for x in tmp)

    if op_func:
        kwargs = {}
        if name := next((k for k, v in op_func.__annotations__.items() if v == Args), None):
            kwargs[name] = args

        if name := next((k for k, v in op_func.__annotations__.items() if v == list[str]), None):
            kwargs[name] = rest

        alternatives = op_func(prev_alternatives, alternatives, **kwargs)
        if arg_overrides.get("nop", None):
            defer_note(depth, f"Nop was set by [b]{op}[/] command. [i]Use --no-nop to override.[/]")

        # if alternatives:
        #     alternatives = list(alternatives)
        #     for a in alternatives:
        #         print(" ".join(map(str, a)))

    if rest:
        alternatives = main(alternatives, rest, depth + 1)

    if depth > 0:
        return alternatives

    prepared = sorted((sum(chain(*a)), tuple(map(sum, a)), a) for a in alternatives)
    seen = set()
    common = set(options)
    table = None
    for total_sum, sep_sum, a in prepared:
        if table is None:
            table = Table("Total", "Set sums", *(f"Set {i}" for i in range(1, len(a) + 1)))

        digits = reduce(lambda acc, val: acc | val, chain(a))
        seen.update(digits)
        common.intersection_update(digits)
        table.add_row(str(total_sum), str(sep_sum), *map(str, a))

    print(table)
    print(f"\n  Aggregated digits: {SudokuSet(tuple(seen))}")
    print(  f"   Digits in common: {SudokuSet(tuple(common))}")

    print(f"\n  Number matching results: {len(prepared)}")


if __name__ == "__main__":
    main()
