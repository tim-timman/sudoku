#!/usr/bin/env python3.11
import argparse
from collections import defaultdict
from functools import total_ordering
from itertools import chain, combinations
from typing import Callable, Iterable


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
    options: list[int]
    include_digits: list[int]
    exclude_digits: list[int]


Alternatives = Iterable[list[SudokuSet[int]]]


def exclusive(a: Alternatives, b: Alternatives) -> Alternatives:
    a = list(a)
    b = list(b)
    return ([*x, *y] for x in a for y in b
            if set(chain(*x)).isdisjoint(chain(*y)))


operators: dict[str, Callable[[Alternatives, Alternatives], Alternatives]] = {
    "X": exclusive,
}


def main(raw_args: list[str] | None = None) -> Alternatives:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sums", type=int, nargs="*", metavar="SUM",
                        help="sums to match of given digits (DEFAULT: %(default)s)")
    parser.add_argument("-n", "--num-digits", type=int, nargs="*", metavar="NUM",
                        action="extend",
                        help="number of digits to match (DEFAULT: %(default)s)")
    parser.add_argument("-o", "--options", type=int, nargs="*", metavar="DIGIT",
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help="starting options (DEFAULT: %(default)s)")
    parser.add_argument("-i", "--include-digits", type=int, nargs="*", metavar="DIGIT",
                        action="extend", default=[],
                        help="digits that must be included (DEFAULT: %(default)s)")
    parser.add_argument("-x", "--exclude-digits", type=int, nargs="*", metavar="DIGIT",
                        action="extend", default=[],
                        help="digits the must be excluded (DEFAULT: %(default)s)")

    parser.add_argument("--", dest="bork", choices=operators.keys(), nargs="*",
                        default=argparse.SUPPRESS,
                        help=f"chain a new set of constraints for match against. "
                             f"{[f'{k} = {v.__name__}' for k, v in operators.items()]} "
                             f"eg. `{parser.prog} -n 3 -s 17 -- X -n 3 -s 7`")

    args: Args
    rest: list[str]
    args, rest = parser.parse_known_args(raw_args, namespace=Args())

    options = set(args.options).difference(args.exclude_digits)

    sums: defaultdict[int, set[SudokuSet[int]]] = defaultdict(set)

    for i in options:
        for combination in combinations(options, i):
            sums[sum(combination)].add(SudokuSet(combination))

    if args.sums:
        alternatives = (v for s in args.sums for v in sums[s])
    else:
        alternatives = (v for _, vs in sums.items() for v in vs)

    if args.num_digits is not None:
        n = args.num_digits
        alternatives = (x for x in alternatives if len(x) in n)

    alternatives = ([x] for x in alternatives if x.issuperset(args.include_digits))

    if rest:
        if (sep := rest.pop(0)) != "--":
            parser.error(f"wrong separator: {sep!r}")
        if not rest:
            parser.error("missing chain operator")
        if (op_func := operators.get(op := rest.pop(0), None)) is None:
            parser.error(f"invalid chain operator: {op!r}")

        rhs = main(rest)
        alternatives = op_func(alternatives, rhs)

    if raw_args is not None:
        return alternatives

    alternatives = sorted((sum(chain(*a)), a) for a in alternatives)
    sum_padding = len(str(max((x for x, _ in alternatives), default=0)))

    for s, a in alternatives:
        print(f"  𝚺{s:{sum_padding}} = {' '.join(map(str, a))}")

    print(f"\n  Number matching results: {len(alternatives)}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
