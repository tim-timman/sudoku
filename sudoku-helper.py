#!/usr/bin/env python3.11
import argparse
import re
from collections import defaultdict
from functools import reduce, partial
from functools import total_ordering
import itertools
from itertools import chain, combinations
from pprint import pp
from typing import Callable, Iterable, Literal, get_args


@total_ordering
class SudokuSet(set):
    def __hash__(self):
        # Switch order for easy comparison
        return sum((1 << 9) >> x for x in self)

    def __repr__(self):
        return str(sorted(self))

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplemented
        return self.__hash__() > other.__hash__()


class Args(argparse.Namespace):
    subparser: str | None
    sums: list[int]
    num_digits: list[int]
    digit_set: list[int]
    include_digits: list[list[int]]
    exclude_digits: list[list[int]]
    num_odd: int | None
    num_even: int | None
    mask: list[Literal[0, 1]]


Alternatives = Iterable[list[SudokuSet[int]]]


def exclusive(a: Alternatives, b: Alternatives) -> Alternatives:
    """
    keep alternatives where the joint set of a's and b's are all unique
    """
    a = list(a)
    b = list(b)
    return ([*x, *y] for x in a for y in b
            if set(chain(*x)).isdisjoint(chain(*y)))


def same_sum(a: Alternatives, b: Alternatives) -> Alternatives:
    """
    keep alternatives where the sum of a's is the same as b's
    """
    temp = [(sum(chain(*x)), x) for x in b]
    return ([*x, *y] for x in a for sy, y in temp
            if x != y and sum(chain(*x)) == sy)


def drop(a: Alternatives, b: Alternatives, args: Args) -> Alternatives:
    return itertools.compress([list(a), list(b)], args.mask)


operators: dict[str, Callable[[Alternatives, Alternatives], Alternatives]] = {
    "X": exclusive,
    "SUM": same_sum,
    "DROP": drop,
}


class SetAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, [*getattr(namespace, self.dest, []), *values])


def set_type(string: str):
    match = re.match(r"(?:(\d(?!\d))?-?(\d)?:)?([1-9]+)$", string)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"{string!r} invalid syntax, must be [d-d:]XXXX "
            f"ex. 1-2:12345 => include 1 or 2 of set {{1,2,3,4,5}}")
    digits = set(map(int, match[3]))
    start = match[1]
    end = match[2]

    if not start and not end:
        start = end = len(digits)
    if not start:
        start = 0
    if not end:
        end = len(digits)
    return (int(start), int(end)), digits


def main(raw_args: list[str] | None = None) -> Alternatives:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sums", type=int, nargs="*", metavar="SUM",
                        help="sums to match of given digits (DEFAULT: %(default)s)")
    parser.add_argument("-n", "--num-digits", type=int, nargs="*", metavar="NUM",
                        action="extend",
                        help="number of digits to match (DEFAULT: %(default)s)")
    parser.add_argument("--digit-set", type=int, nargs="*", metavar="DIGIT",
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
    parser.add_argument("-m", "--mask", type=int, nargs="*", action="extend", choices=(0, 1), default=[],
                        help="a mask for various commands (DEFAULT: %(default)s)")

    parser.add_argument("--", dest="bork", choices=operators.keys(), nargs="*",
                        default=argparse.SUPPRESS,
                        help=f"chain a new set of constraints for match against. "
                             f"{[f'{k} = {v.__name__}' for k, v in operators.items()]} "
                             f"eg. `{parser.prog} -n 3 -s 17 -- X -n 3 -s 7`")

    args: Args
    rest: list[str]
    args, rest = parser.parse_known_args(raw_args, namespace=Args())

    options = set(args.digit_set)

    sums: defaultdict[int, set[SudokuSet[int]]] = defaultdict(set)

    for i in range(1, len(options)):
        for combination in combinations(options, i):
            sums[sum(combination)].add(SudokuSet(combination))

    if args.sums:
        alternatives = (v for s in args.sums for v in sums[s])
    else:
        alternatives = (v for _, vs in sums.items() for v in vs)

    if args.num_digits is not None:
        n = args.num_digits
        alternatives = (x for x in alternatives if len(x) in n)

    if args.include_digits:
        alternatives = (x for x in alternatives if any(start <= len(x.intersection(d)) <= end
                                                       for (start, end), d in args.include_digits))
    if args.exclude_digits:
        alternatives = (x for x in alternatives if all(start <= len(d) - len(x.intersection(d)) <= end
                                                       for (start, end), d in args.exclude_digits))

    if args.num_even:
        alternatives = (x for x in alternatives if sum(map(lambda y: y % 2 == 0, x)) >= args.num_even)

    if args.num_odd:
        alternatives = (x for x in alternatives if sum(map(lambda y: y % 2 == 1, x)) >= args.num_odd)

    alternatives = ([x] for x in alternatives)

    if rest:
        if (sep := rest.pop(0)) != "--":
            parser.error(f"wrong separator: {sep!r}")
        if not rest:
            parser.error("missing chain operator")
        if (op_func := operators.get(op := rest.pop(0), None)) is None:
            parser.error(f"invalid chain operator: {op!r}")

        rhs = main(rest)
        kwargs = {}
        if name := next((k for k, v in op_func.__annotations__.items() if v == Args), None):
            kwargs[name] = parser.parse_args(rest, namespace=Args())

        if name := next((k for k, v in op_func.__annotations__.items() if v == list[str]), None):
            kwargs[name] = rest

        alternatives = op_func(alternatives, rhs, **kwargs)

    if raw_args is not None:
        return alternatives

    print("")
    alternatives = sorted((sum(chain(*a)), tuple(map(sum, a)), a) for a in alternatives)
    sum_padding, sep_sum_padding = [*map(partial(max, default=0),
                                         zip(*([*map(len, map(str, (x, y)))]
                                               for x, y, _ in alternatives)))] or (0,0)

    for total_sum, sep_sum,  a in alternatives:
        print(f"  𝚺{total_sum:{sum_padding}} {sep_sum!s:{sep_sum_padding}} = {' '.join(map(str, a))}")

    print(f"\n  Number matching results: {len(alternatives)}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
