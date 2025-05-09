import time
from itertools import filterfalse
from random import randint, random
from . import GenericFeatureSelector
from . import score_compare


class GeneticSelection(GenericFeatureSelector):

    chromosome_scoring_table = {}

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
            population (int): the genetic algorithm population. Default is 10
            generations (int): number of genetic algo generations. Default is 10
            mutation_rate (float): the mutation rate. Default is 0.05 (5%)
            selection_method (str): The mating-selection routine. Options are:
                'roulette' for roulette wheel selection,
                'tourament' for tournament selection.
                Default is 'roulette'.

        Usage:
            >>> b = GeneticSelection(
                num_features = 5, # five total features to select from
                forced_features = [0, 2], # features 0 and 2 should always be ON
                stopping_time = 120, # stop after 120 seconds.
                population = 12,
                generations = 12,
                selection_method = 'tournament'
            )
        """

        GenericFeatureSelector.__init__(self, num_features, *args, **kwargs)

        # Genetic algorithm specific kwargs
        self.population = kwargs.get("population", 11)
        self.generations = kwargs.get("generations", 12)
        self.mutation_rate = kwargs.get("mutation_rate", 0.09)
        self.selection_method = kwargs.get("selection_method", "roulette")

        self.current_generation = 0

        # Ensure population is a multiple of 2
        self.population = self.population + (self.population % 2)

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

        # generator of all possible combinations
        all_chromosomes = filterfalse(
            lambda x: x & self.forced_chromosome_number
            != self.forced_chromosome_number,
            range(1, self.max_chromosome_number),
        )

        # non-forced chromosomes
        self.non_forced_features = set(range(self.num_features)).difference(
            set(self.forced_features)
        )

        # total number of possible combinations
        self.max_num_chromosomes = sum([1 for _ in all_chromosomes])

    def is_still_running(self) -> bool:
        """Is the feature selector still in progress?

        Returns:
            bool: returns whether or not the feature selector
                is still evaluating features.
        """

        # Check that the timer has started
        if not self.clock_started:
            return True

        # Check if we're done with the final generation
        if self.current_generation >= self.generations:
            return False

        # Check if time has run out
        if time.time() - self.clock_start_time > self.stopping_time:
            return False

        # Check if we've seen all the possible chromosomes
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

        return

    def initialize_population(self) -> None:
        """Create an initial popuplation of potential solutions. The initial population
        is composed of random chromosomes
        """
        self.current_chromosomes = [
            randint(1, self.max_chromosome_number) | self.forced_chromosome_number
            for _ in range(self.population)
        ]

        return

    def selection(self) -> list[int]:
        """Selection algorithm that takes the current pool/population of chromosomes
        and selects 2 parents (Either through roulette wheel, or through
        tournament selection) for recombination.

        Returns:
            list[int]: parents to recombine into new chromosomes
        """

        parents = [None, None]

        # Roulette wheel selection
        if self.selection_method == "roulette":

            probabilities = [0] * self.population

            for scorer in range(len(self.scores[0])):

                scorer_probs = [
                    self.scores[i][scorer] / sum(map(lambda s: s[scorer], self.scores))
                    for i in range(self.population)
                ]
                probabilities = [
                    p + (ps / len(self.scores[0]))
                    for p, ps in zip(probabilities, scorer_probs)
                ]

            cumulative_probs = [
                sum(probabilities[: i + 1]) for i in range(self.population)
            ]

            # Choose the parents
            for i in range(2):
                randn = random()
                for j in range(self.population):
                    if randn < cumulative_probs[j]:
                        parents[i] = self.current_chromosomes[j]
                        break
            return parents

        # Tournament selection
        elif self.selection_method == "tournament":
            tournament_size = 3
            for i in range(2):
                best = None
                winner = None
                for k in range(tournament_size):
                    candidate_idx = randint(0, self.population - 1)
                    if (best == None) or (
                        score_compare(self.scores[candidate_idx], best)
                    ):
                        winner = self.current_chromosomes[candidate_idx]
                        best = self.scores[candidate_idx]
                parents[i] = winner
            return parents

        else:
            raise ValueError(
                'Required selection type must be "tournament" or "roulette".'
            )

    def crossover(self, parent1: int, parent2: int) -> list[int]:
        """Performs crossover between 2 parents to generate 2 new children.
        For each child: a crossover point is chosen at random and the
        child has the genes of the parent1 up until the crossover point and the
        remaining genes come from parent 2

        Args:
            parent1 (int): the first parent chromosome
            parent2 (int): the second parent chromosome.

        Returns:
            list[int]: 2 children chromosomes that are the result of crossover.
        """

        # Children start out as copies of the parents
        child1, child2 = parent1, parent2

        # Check whether we are recombining (there's a small chance for
        # no recombination)
        if random() < 0.9:
            # perform crossover
            crossover_point = randint(1, self.num_features - 2)
            child1_bits = (
                bin(parent1)[2:].zfill(self.num_features)[:crossover_point]
                + bin(parent2)[2:].zfill(self.num_features)[crossover_point:]
            )
            child1 = int(child1_bits, 2)
            child2_bits = (
                bin(parent2)[2:].zfill(self.num_features)[:crossover_point]
                + bin(parent1)[2:].zfill(self.num_features)[crossover_point:]
            )
            child2 = int(child2_bits, 2)

        return [child1, child2]

    def mutation(self, chromosome: int) -> int:
        """Performs mutation on a chromosome. Uses the mutation
        rate defined in the constructor to decide whether to mutation
        bits in the chromosome.

        Args:
            chromosome (int): chromosome to maybe mutate

        Returns:
            int: mutated chromosome
        """
        bitstring = list(bin(chromosome)[2:].zfill(self.num_features))
        for i in range(len(bitstring)):
            if random() < self.mutation_rate:
                # flip bit
                bitstring[i] = "0" if bitstring[i] == "1" else "1"
        return int("".join(bitstring), 2)

    def next(self) -> list[int]:
        """_summary_

        Returns:
            list[int]: _description_
        """

        # initialize chromosomes if generation = 0
        if self.current_generation == 0:
            self.clock_started = True
            self.clock_start_time = time.time()
            self.initialize_population()
            self.current_generation += 1
            return self.current_chromosomes

        # Check for lack of genetic diversity
        if len(list(set(self.current_chromosomes))) < 3:
            self.initialize_population()
            self.current_generation += 1
            return self.current_chromosomes

        # otherwise do genetic algorithm
        # Selection
        selected_parents = [self.selection() for _ in range(self.population // 2)]

        children = []

        for i in range(self.population // 2):

            # Crossover
            parent1, parent2 = selected_parents[i]
            child1, child2 = self.crossover(parent1, parent2)

            # mutation
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)

            children.append(child1)
            children.append(child2)

        self.current_chromosomes = children
        self.current_generation += 1

        # dont allow any empty chromosomes:
        # replace empty chromosomes with a random chromosome
        while 0 in self.current_chromosomes:
            self.current_chromosomes[self.current_chromosomes.index(0)] = (
                randint(1, self.max_chromosome_number) | self.forced_chromosome_number
            )

        return self.current_chromosomes
