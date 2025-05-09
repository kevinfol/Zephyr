from itertools import filterfalse
from . import GenericFeatureSelector


class BruteForceFeatureSelection(GenericFeatureSelector):
    """Feature selection algorithm in which each possible combination
    of feature variables is evaluated.
    """

    BATCH_SIZE = 10

    def __init__(self, num_features: int, *args, **kwargs) -> None:
        """Class initializer

        Args:
            num_features (int): number of features to choose from

        Available Keyword Args:
            forced_features (list[int]): a list of indices, where
                each index corresponds to a feature that should
                always be included in the model.

        Usage:
            >>> b = BruteForceFeatureSelection(
                num_features = 5, # five total features to select from
                forced_features = [0, 2] # features 0 and 2 should always be ON
            )
        """

        GenericFeatureSelector.__init__(self, num_features, *args, **kwargs)

        # maximum chromosome (in integer form)
        self.max_chromosome_number = 2**num_features - 1

        # current chromosome
        self.current_chromosome_number = 1

        # Forced features
        self.forced_features = kwargs.get("forced_features", [])
        self.forced_chromosome_number = sum([2 ** (n) for n in self.forced_features])

        # maximum number of chromosomes possible (e.g. after accounting for forced)
        self.max_number_of_chromos = 2 ** (num_features - len(self.forced_features)) - 1

        # Iterator
        self.iterator = filterfalse(
            lambda x: x & self.forced_chromosome_number
            != self.forced_chromosome_number,
            range(self.current_chromosome_number, self.max_chromosome_number),
        )

        # keep track of chromosome scores (unused)
        self.num_features = num_features
        self.scores = [None for _ in range(self.num_features)]

        return

    def is_still_running(self) -> bool:
        """Is the feature selector still in progress?

        Returns:
            bool: returns whether or not the feature selector
                is still evaluating features.
        """

        return self.current_chromosome_number < self.max_chromosome_number - 1

    def set_scores(self, scores: list[float]) -> None:
        """Assigns a skill score to each chromosome in the current batch
        of chromosomes. Not used in this algorithm for anything...

        Args:
            scores (list[float]): list of skill scores (higher is better)
                where each score in the list corresponds to a chromosome in
                the current batch of chromosomes.
        """
        self.scores = scores
        for i, score in enumerate(scores):
            self.chromosome_scoring_table[self.current_chromosomes[i]] = score
        return

    def next(self) -> list[int]:
        """Returns the next batch of chromosomes to evaluate. This selection algorithm
        just counts up from 1 to the maximum possible chromosome number.

        Returns:
            list[int]: a batch of chromosomes to evaluate
        """

        # output list of chromosomes
        self.current_chromosomes = []

        # Create the batch
        batch_count = 0
        while (batch_count < 10) and (self.is_still_running()):
            chromo_number = next(self.iterator)
            self.current_chromosomes.append(chromo_number)
            self.current_chromosome_number = chromo_number
            batch_count += 1

        return self.current_chromosomes
