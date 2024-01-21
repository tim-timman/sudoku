#!/usr/bin/env python3
import argparse
import cProfile
import random
import time
from abc import ABC, abstractmethod
from bisect import bisect
from operator import itemgetter
from typing import Callable, Optional, TypeAlias


class BrokenConstraintError(Exception):
    pass


class Cell:
    def __init__(self, idx: int):
        self.options: int = 0b1111111111
        self.entropy = self.options.bit_count() - 1
        self.idx = idx
        self.constraints: list["Constraint"] = []

    def add_constraint(self, constraint: "Constraint"):
        self.constraints.append(constraint)
        constraint.add(self)

    @property
    def row(self):
        return self.idx // 9 + 1

    @property
    def col(self):
        return self.idx % 9 + 1

    @property
    def box(self):
        return self.idx // 27 * 3 + (self.idx % 9) // 3 + 1

    def __lt__(self, other):
        return self.entropy < other.entropy

    def __hash__(self):
        return self.idx

    def __repr__(self):
        return f"r{self.row}c{self.col}b{self.box}"


CellIdx: TypeAlias = int


class Constraint(ABC):
    def __init__(self, name: str):
        self.name = name
        self.cell_names: set[str] = set()
        self.cell_indices: list[CellIdx] = []

    def add(self, cell: Cell):
        self.cell_names.add(repr(cell))
        # Use indirection to simplify backtracking
        self.cell_indices.append(cell.idx)

    @property
    def extract_cells(self) -> Callable[[list[Cell]], list[Cell]]:
        return itemgetter(*self.cell_indices)

    @abstractmethod
    def constrain(self, board: list[Cell]) -> list[Cell]:
        ...

    def __repr__(self):
        return f"{self.__class__}{self.name}<{','.join(sorted(self.cell_names))}>"


class Unique(Constraint):
    def constrain(self, board: list[Cell]) -> list[Cell]:
        constrained_digits = 0
        constrained_cells = []
        for cell in sorted(self.extract_cells(board)):
            if cell.entropy == 0:
                constrained_digits |= cell.options
            else:
                prev_entropy = cell.entropy
                cell.options &= ~constrained_digits
                cell.entropy = cell.options.bit_count() - 1
                if cell.entropy < prev_entropy:
                    constrained_cells.append(cell)
            if cell.entropy < 0:
                raise BrokenConstraintError(f"{cell} has no remaining options")

        return constrained_cells


def solve(quiet: bool = True):
    board = []
    rows = {i: Unique(f"Row#{i}") for i in range(1, 10)}
    cols = {i: Unique(f"Col#{i}") for i in range(1, 10)}
    boxs = {i: Unique(f"Box#{i}") for i in range(1, 10)}

    for i in range(9 * 9):
        cell = Cell(i)
        board.append(cell)
        cell.add_constraint(rows[cell.row])
        cell.add_constraint(cols[cell.col])
        cell.add_constraint(boxs[cell.box])

    remaining_cells = board.copy()
    branches: list[tuple[tuple[int], tuple[int]]] = []
    number_of_backtracks = 0
    t_start = time.monotonic_ns()
    random_state = random.getstate()
    while True:
        try:
            while remaining_cells:
                remaining_cells.sort()
                # Pick the lowest entropy cell
                last_idx_in_entropy_group = bisect(remaining_cells, remaining_cells[0]) - 1
                if last_idx_in_entropy_group > 0:
                    # Don't think we need to add a branch here, shouldn't matter, right?
                    cell_idx = random.randint(0, last_idx_in_entropy_group)
                else:
                    cell_idx = 0

                cell = remaining_cells.pop(cell_idx)

                # Collapse
                if cell.entropy > 0:
                    # add branch
                    option = random.choice(
                        tuple(
                            filter(
                                None,
                                (
                                    cell.options & (1 << x)
                                    for x in range(cell.options.bit_length())
                                ),
                            )
                        )
                    )
                    # Remove what we chose in this branch
                    cell.options &= ~option
                    branches.append((tuple(c.options for c in board), tuple(c.idx for c in remaining_cells)))
                    cell.options = option
                    cell.entropy = cell.options.bit_count() - 1

                constraint_propagation_stack: list[Cell] = [cell]
                while constraint_propagation_stack:
                    cell = constraint_propagation_stack.pop()
                    # Update constraints
                    for constraint in cell.constraints:
                        constraint_propagation_stack.extend(constraint.constrain(board))
        except BrokenConstraintError:
            if not branches:
                raise
            board_options, remaining_cells_indices = branches.pop()
            for idx, o in enumerate(board_options):
                board[idx].options = o
                board[idx].entropy = o.bit_count() - 1
            remaining_cells = [board[idx] for idx in remaining_cells_indices]
            number_of_backtracks += 1
            if not quiet:
                print(
                    number_of_backtracks if number_of_backtracks % 100 == 0 else ".",
                    sep="",
                    end="",
                )
        else:
            time_taken = time.monotonic_ns() - t_start
            if not quiet:
                print("\n", fmt_board(board))
                print(
                    f"^ took {time_taken / 10**6:.2f} ms, with {number_of_backtracks} backtracks"
                )
            break


class ProgArgs(argparse.Namespace):
    seed: Optional[str] = None
    forever: bool = False
    quiet: bool = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str)
    parser.add_argument("-f", "--forever", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(namespace=ProgArgs())
    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.setstate((3, (402751974, 841291915, 976215107, 4288087454, 2255914921, 4191072287, 3587906988, 1140395808, 2595483033, 595100139, 1579460200, 3064630791, 1530120159, 3106635667, 2772770879, 3979187380, 2718817573, 865265898, 2839902382, 3280367565, 802987044, 1619518048, 322981127, 3267837683, 4093625388, 2915965257, 2063893899, 3175348520, 2551341617, 3638044758, 2700928936, 1746685800, 2076653964, 2845590663, 3418873061, 1222580806, 2370886530, 3025401322, 965322728, 3402234163, 1957168662, 2782658694, 3135211087, 2140881931, 3872831964, 2662694149, 1042749232, 1159696401, 3899315018, 777101229, 4255811884, 1517567887, 477101356, 1920695814, 4185257287, 1165638166, 3129224377, 2162559175, 18968023, 3214472212, 1836493031, 4018869408, 2498949507, 3191665222, 2388294672, 3613695322, 1247642853, 2466720669, 1276838851, 903330475, 366725305, 398454160, 4044657976, 950179593, 496073064, 116647181, 1603918641, 677925517, 477420680, 3049541619, 3640807300, 1273589563, 409135063, 1405469769, 3563982275, 335860310, 3470399424, 1997715044, 1058528900, 1446889768, 1545442918, 4082956040, 1800814530, 1771539551, 340501441, 76697654, 1142799249, 1746148254, 1937286245, 1494018992, 3736903779, 2849754545, 2138395172, 2656992754, 984250219, 2878462775, 840181258, 2119214904, 1558273804, 923273366, 2910225682, 3316470212, 1836438338, 2090560348, 922741275, 985383345, 3895930625, 8332153, 1323298607, 1554125433, 84041597, 3006613844, 2144018767, 2468461898, 1466222307, 790929631, 2071887357, 2583423627, 2347170319, 1612316031, 1761803633, 416633697, 3459602195, 147922678, 1263769649, 2112014760, 1328145648, 3289464979, 4085562174, 4246036257, 3314586936, 2056055311, 3346753034, 1686244910, 3150764402, 2565084845, 3274296586, 126534169, 2993280992, 4096416275, 4212581275, 3563661196, 3439721320, 450757243, 3360230288, 1860561488, 3777476097, 2716589740, 369195235, 3352577003, 3133746472, 3143576786, 2838141743, 2915634979, 2527443664, 1349753883, 33210465, 3988405801, 2128872739, 1549087508, 4240307466, 1772185263, 1450255830, 3783886413, 1446768282, 178626739, 2975710016, 1063157454, 4093267345, 106006731, 707198220, 1992467237, 175661839, 1209983542, 1017730679, 910886251, 1691850300, 3777382576, 3291553153, 567466757, 3274565284, 3304844747, 2796457753, 1352564560, 1019829503, 2149067739, 1925201985, 696995980, 1701119526, 3979514349, 4005539892, 2720139307, 2046069793, 4065446032, 1919709644, 3589611463, 4215978368, 2214117875, 611254241, 3586375486, 3947231492, 3490397052, 1313780327, 271257485, 3746723178, 1804487349, 605624514, 3185538028, 3101544497, 195445804, 3702562779, 2144794368, 3589056522, 2026647987, 17130683, 1648460613, 1425779967, 1298791856, 698214025, 3594235906, 1967679848, 2722441781, 3325244242, 4168230214, 818667619, 4260077823, 3198295115, 1449267232, 949569802, 3571091004, 2039672302, 3093873915, 1562363589, 2686654277, 2578579856, 4187945120, 2558878021, 1822998039, 665479938, 3166686833, 801356813, 1017591069, 1393688341, 3767831327, 2115540576, 4240923024, 2712122471, 2049765756, 3072572316, 1494882754, 499161637, 3097907510, 3455635848, 3261534989, 668962134, 221456937, 1612365626, 2001765416, 4157728290, 1152014329, 3760698497, 2551514353, 1206814813, 623864042, 3588631369, 2969367636, 2690649011, 3175355791, 617549232, 1791399800, 1951914961, 135043914, 1199387163, 2320298745, 355416817, 1607547367, 1310791634, 109037769, 900978692, 2083862228, 1832167851, 3817948466, 808039583, 3669536308, 3782484563, 4216425709, 181267033, 3576208981, 819398806, 798160681, 780905647, 1772594444, 2378678238, 3150424615, 887628445, 2518789359, 441597133, 2094920603, 2702121849, 4161192584, 2621381160, 477448810, 4027087039, 4145186068, 912311834, 393644357, 4255037029, 3629726434, 3494987207, 88737156, 305403571, 4207960926, 1385726403, 345176001, 961630538, 4274831690, 711550627, 2284172692, 3205492004, 3132749034, 693054169, 2177968104, 1873578929, 2018932321, 3296465284, 1477154429, 4258021932, 2897470674, 2840886645, 3440969698, 420123194, 1273473509, 991890834, 3733982277, 2782088425, 4154091101, 3493293311, 2991220397, 3123036355, 2088552536, 2175547872, 1881011823, 1149110922, 2512679718, 3060166784, 1965512606, 4187689879, 601886843, 2582324081, 3431935522, 2766732821, 2821411082, 1446557076, 1369446151, 3306597979, 1286221843, 2619436108, 4063068542, 1052057619, 4259580232, 2415498353, 3495266926, 3220600894, 877743207, 4270501696, 3589438074, 875144926, 2470906214, 542359877, 2924244220, 771195778, 3962516209, 1530007632, 1955703718, 1838234299, 1644764622, 3256974184, 2073291222, 3338929607, 2456567386, 3990460643, 3204439217, 450523230, 3462482577, 2973770206, 567187124, 2411529291, 2023179284, 2576911362, 1793600658, 876861603, 3771114908, 2134353808, 3585375074, 2742467397, 2352348040, 122300223, 2104066591, 701920309, 4235119672, 1008943961, 1431005208, 479855308, 3651773588, 1297972313, 1511808919, 1667002020, 2540936994, 1987643554, 2518893802, 1453668750, 3082558429, 2477354751, 4187084762, 2301648021, 3120245472, 3597768483, 1219749728, 3756362065, 2996384689, 788743613, 1405077872, 2777210602, 701588077, 3866570306, 3363533602, 1675298399, 664733247, 3761928608, 4054610552, 739960342, 114422415, 4183602383, 2804513755, 1528256950, 244399964, 3801248361, 1750243931, 4208940513, 1976299828, 2017741641, 1657413056, 3689633971, 747223571, 1608372376, 1757729962, 2451793218, 435798872, 2597515113, 3319264192, 2551136312, 3658613556, 1517462689, 2543161919, 2415104647, 1814581157, 585671818, 94511260, 398383552, 2867997526, 3975441741, 3402838494, 701809086, 2401536256, 2193362778, 2575939616, 2792884940, 3501527806, 66685818, 3102572871, 3101694738, 2084775083, 1138118842, 604698152, 2333150448, 372226217, 2697475749, 1705577359, 3621924310, 1469585582, 1475045935, 3322361487, 640108078, 4183217736, 99221813, 3043629265, 8779118, 4287540552, 194381545, 2920885554, 3771068460, 2004901395, 991705894, 3757145097, 1856655532, 2753922732, 2943665274, 3417761295, 967177467, 274134938, 527814165, 2522255808, 746329706, 1929327238, 3584478849, 3094861469, 1132063826, 2775654618, 1692227933, 2912019375, 493793512, 1933587740, 1542441271, 2524305210, 141841052, 3335312390, 982955690, 1446503337, 1078182144, 1522637597, 2900531007, 506068143, 1891406530, 1630278839, 763837724, 2466372261, 3812829237, 1845828452, 1528298866, 3694684997, 242766393, 1016709372, 3019732156, 4258897821, 2834713848, 2258673636, 3427300793, 3700304646, 3566329857, 2056959801, 3114729538, 2130736381, 3643150224, 948471229, 2507841314, 1661623305, 2242151742, 2956908990, 4199955023, 3424412920, 1230404573, 833044975, 4121101119, 3432263369, 2807529973, 2376110412, 3683958658, 2652954733, 3966471097, 2171041197, 2261010525, 3982218027, 1777184658, 2611264964, 1915748358, 2303578638, 2523679864, 2438835277, 4259951561, 585897249, 86074894, 3156566878, 2373260, 1556737186, 3571046026, 4216935746, 2314970999, 2000567896, 2190628374, 1388546714, 1004401967, 3952955088, 3172037926, 1866197890, 237008511, 394258505, 3014938000, 2620505909, 4067773805, 1670145394, 3480333064, 1705928329, 1034587388, 261215841, 2995256824, 127612900, 3830016689, 2062452330, 1519072925, 973586001, 1270854871, 2658396315, 3020197944, 3098929735, 3617321440, 2939554527, 1969104219, 2587670909, 1506900630, 3948520892, 524600488, 3238410024, 257350142, 3283993784, 592), None))

    with cProfile.Profile() as pr:
        solve(args.quiet)
        pr.dump_stats("prof")

    while args.forever:
        solve(args.quiet)


def fmt_board(board: list[Cell]):
    output = []
    for cell in board:
        idx = cell.idx
        if idx % 27 == 0:
            output += "—" * 31, "\n"

        if idx % 9 == 0:
            output += "|"

        output += " ", str(cell.options.bit_length() - 1), " "

        if idx % 3 == 2:
            output += "|"

        if idx % 9 == 8:
            output += "\n"

    output += "—" * 31
    return "".join(output)


if __name__ == "__main__":
    main()
