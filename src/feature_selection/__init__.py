from src.scoring import calc_rmse, score_compare


class GenericFeatureSelector:

    chromosome_scoring_table = {}

    def __init__(self, num_features: int, *args, **kwargs):

        self.num_features = num_features

    def convert_number_to_bool_array(self, n: int) -> list[bool]:
        """converts an integer to an array of boolean values.
        e.g.
            >>> convert_number_to_bool_array(2, 3)
            [False, True, False]

        Args:
            n (int): chromosome number to convert to bool array
            num_features (int): number of features

        Returns:
            list[bool]: an array of booleans representing whether the
                corresponding feature should be included in the model.
        """

        return [
            True if b == "1" else False
            for b in self.convert_number_to_bool_str(n, self.num_features)
        ]

    def convert_number_to_bool_str(self, n: int) -> str:
        """converts an integer to an array of boolean values.
        e.g.
            >>> convert_number_to_bool_str(2, 3)
            "010"

        Args:
            n (int): chromosome number to convert to bool array
            num_features (int): number of features

        Returns:
            str: a string of '1's and '0's representing whether the
                corresponding feature should be included in the model.
        """

        bool_str = bin(n)[2:].zfill(self.num_features)
        return bool_str

    def print_progress(self, skip_preprint: bool = True) -> None:
        """
        Updates the last line in the console with the current progress of the
        feature selector

        Args:
        """

        # List of chromosomes
        chromosomes = self.chromosome_scoring_table.keys()

        # Best score
        scores = list(self.chromosome_scoring_table.values())
        if scores != []:
            best_score = sorted(
                scores,
                key=lambda s: s[0],
                reverse=True,
            )[
                0
            ][0]
        else:
            best_score = 0

        # Compute total length
        total_num = (2**self.num_features) - 1

        # Determine the divider (e.g. number of chromosomes per block).
        # either 4, 8, or 16
        divider = 4
        if total_num > 512:
            divider = 8
        if total_num > 2048:
            divider = int((total_num / 192) - ((total_num / 192)) % 4 + 4)

        # Number of total block characters (each block has 4 chromosomes)
        total_length = int(
            (total_num // divider) + (1 if total_num % divider != 0 else 0)
        )

        # compute number of rows (12 blocks per row)
        total_rows = int(total_length // 12 + (1 if total_length % 12 != 0 else 0))

        # preprint
        if not skip_preprint:
            print("\n" * (total_rows * 2))

        # Characters and ansi control sequences
        blocks = {
            0: "\u2591",  #  0000 ░
            1: "\u2597",  #  0001 ▗
            2: "\u2596",  #  0010 ▖
            3: "\u2584",  #  0011 ▄
            4: "\u259d",  #  0100 ▝
            5: "\u2590",  #  0101 ▐
            6: "\u259e",  #  0110 ▞
            7: "\u259f",  #  0111 ▟
            8: "\u2598",  #  1000 ▘
            9: "\u259a",  #  1001 ▚
            10: "\u258c",  # 1010 ▌
            11: "\u2599",  # 1011 ▙
            12: "\u2580",  # 1100 ▀
            13: "\u259c",  # 1101 ▜
            14: "\u259b",  # 1110 ▛
            15: "\u2588",  # 1111 █
        }

        # go to first row
        s = "\033[F" * ((total_rows * 2) + 1)

        for row in range(total_rows):

            if row == 0:
                s += "Chromosomes evaluated: "
            elif row == 1:
                s += f"{len(chromosomes):>07} / {total_num:>07}      "
            elif row == 2:
                s += f"Best: {best_score:+2.6f}        "
            else:
                s += "                       "
            for position in range(12):

                block_no = (row * 12) + position
                start_position = block_no * divider
                if start_position <= total_num:
                    quad_str = ""
                    for quad in range(4):
                        quad_start_position = start_position + quad * (divider // 4)
                        if any(
                            [
                                quad_start_position + i in chromosomes
                                for i in range(divider // 4)
                            ]
                        ):
                            quad_str += "1"
                        else:
                            quad_str += "0"
                    block_char = blocks[int(quad_str, 2)]
                    s += block_char + " "
            s += "\n"
            if row == 0:
                s += f"                       0·{str(divider).ljust(2,'·')}{str(2*divider).ljust(2,'·')}·················\n"
            elif row == (total_rows - 1):
                s += (
                    "                       "
                    + "··" * ((total_length % 12) - 1)
                    + str(total_num)
                    + "\n"
                )
            else:
                s += f"                       {str(row*12*divider).ljust(23,'·')}\n"

        print(s)


from .brute_force import BruteForceFeatureSelection
from .sequential_floating import SequentialFloatingSelection
from .genetic import GeneticSelection


def get(feature_selector: str) -> object:
    """Returns the feature selection algorithm

    Args:
        feature_selector (str): name of the feature selection algorithm to get.

    Returns:
        object: Feature Selector object that has methods: `set_scores`, `is_still_running`, and `next`
    """
    fs = None
    if feature_selector == "brute_force":
        fs = BruteForceFeatureSelection
    elif feature_selector == "sequential_floating":
        fs = SequentialFloatingSelection
    elif feature_selector == "genetic":
        fs = GeneticSelection
    return fs
