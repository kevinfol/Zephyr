import time
from itertools import filterfalse
from random import randint, choice
from . import GenericFeatureSelector


class SequentialFloatingSelection(GenericFeatureSelector):
    """_summary_"""

    def __init__(self, num_features: int, *args, **kwargs) -> None:
        """_summary_

        Args:
            num_features (int): number of features to choose from.

        Available Keyword Args:
            forced_features (list[int]): a list of indices, where
                each index corresponds to a feature that should
                always be included in the model.
            stopping_time (int): number of seconds after which
                to stop selecting new chromosomes during this
                search.

        Usage:
            >>> b = SequentialForwardFloatingSelection(
                num_features = 5, # five total features to select from
                forced_features = [0, 2], # features 0 and 2 should always be ON
                stopping_time: 120 # stop after 120 seconds.
            )
        """

        GenericFeatureSelector.__init__(self, num_features, *args, **kwargs)

        # maximum chromosome (in integer form)
        self.num_features = num_features
        self.max_chromosome_number = 2**num_features - 1

        # Forced features
        self.forced_features = kwargs.get("forced_features", [])
        self.forced_chromosome_number = sum([2 ** (n) for n in self.forced_features])

        # maximum number of chromosomes possible (e.g. after accounting for forced)
        self.max_number_of_chromos = 2 ** (num_features - len(self.forced_features)) - 1

        # Clock
        self.stopping_time = kwargs.get("stopping_time", 600)
        self.clock_started = False
        self.clock_start_time = time.time()

        self.starting_chromosome = (
            randint(1, self.max_chromosome_number) | self.forced_chromosome_number
        )

        # non-forced chromosomes
        self.non_forced_features = set(range(self.num_features)).difference(
            set(self.forced_features)
        )

        # keep track of number of chromosomes evaluated
        self.num_evaluated = 0

        # keep track of chromosome scores
        self.scores = [None for _ in range(self.num_features)]

        # Seeded?
        self.is_seeded = False
        self.seeds_used = 0

        return

    def is_still_running(self) -> bool:
        """Is the feature selector still in progress?

        Returns:
            bool: returns whether or not the feature selector
                is still evaluating features.
        """

        # Check that the timer has started
        if not self.clock_started:
            return True

        # Check if time has run out
        if time.time() - self.clock_start_time > self.stopping_time:
            return False

        # Check if we've seen all the possible chromosomes (or 98% of them to catch any weird little bugs.)
        self.num_evaluated = len(self.chromosome_scoring_table.keys())
        if self.num_evaluated >= 0.98 * self.max_number_of_chromos:
            return False

        return True

    def set_scores(self, scores: list[float]) -> None:
        """Assigns a skill score to each chromosome in the current batch
        of chromosomes.

        Args:
            scores (list[float]): list of skill scores (higher is better)
                where each score in the list corresponds to a chromosome in
                the current batch of chromosomes.
        """
        self.scores = scores
        for i, score in enumerate(scores):
            self.chromosome_scoring_table[self.current_chromosomes[i]] = score
        self.set_new_starting_chromosome()
        return

    def preseed(self, chromosomes: list[int]) -> None:
        """If there are a lot of candidate predictors, we may need to start the
        feature selector off in a known good place so that it doesn't iterate
        through a bunch of bad solutions before finding a decent good one.

        Preseeding sets the initial chromosomes for iterative feature selectors
        to a known set of good chromosomes.

        Args:
            chromosomes (list[int]): A list of known decent chromosomes
                to start the iterative selector with.
        """
        self.seeds = [
            chromosome | self.forced_chromosome_number for chromosome in chromosomes
        ]
        self.starting_chromosome = choice(self.seeds)
        self.seeds.pop(self.seeds.index(self.starting_chromosome))
        self.is_seeded = True

    def set_new_starting_chromosome(self) -> None:
        """Uses the current scores to set the new starting chromosome,
        or if the current starting chromosome is the best, randomly generate
        a new starting chromosome.
        """
        sorted_scores = sorted(self.scores, reverse=True)
        best_score_idx = self.scores.index(sorted_scores[0])
        if best_score_idx == 0:
            if self.is_seeded and self.seeds_used < len(self.seeds):
                self.starting_chromosome = choice(self.seeds)
                self.seeds.pop(self.seeds.index(self.starting_chromosome))
                self.seeds_used += 1
            else:
                self.starting_chromosome = (
                    randint(1, self.max_chromosome_number)
                    | self.forced_chromosome_number
                )
        else:
            self.starting_chromosome = self.current_chromosomes[best_score_idx]

        return

    def next(self) -> list[int]:
        """_summary_

        Returns:
            list[int]: _description_
        """

        # Start the clock / first iteration
        if not self.clock_started:
            self.clock_started = True
            self.clock_start_time = time.time()

        # New chromosomes are the current starting chromosome,
        # and:  the starting chromosome with all the bits toggled on
        # or off.
        self.current_chromosomes = [self.starting_chromosome] + [
            self.starting_chromosome ^ (2**n) for n in self.non_forced_features
        ]

        # dont allow any empty chromosomes:
        # replace empty chromosomes with a random chromosome
        while 0 in self.current_chromosomes:
            self.current_chromosomes[self.current_chromosomes.index(0)] = (
                randint(1, self.max_chromosome_number) | self.forced_chromosome_number
            )

        return self.current_chromosomes
