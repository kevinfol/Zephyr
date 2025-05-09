import numpy as np


def weighted_hamming_distance(
    bitstring_1: str, bitstring_2: str, score_2: float
) -> float:
    """returns the hamming distance between 2 chromosomes, multiplied by
    the score associated with the second chromosome (usually between 0 and 1).
    In this way, we artificially make it so that higher skill models seem
    to be further away from the first chromosome.

    Args:
        bitstring_1 (str): a chromosome to analyze, e.g. '1011'
        bitstring_2 (str): a chromosome to compare, e.g. '1000'
        score_2 (list(float)): scores associated with bitstring_2

    Returns:
        float: the weighted hamming distance
    """

    distance = 0
    for n in range(len(bitstring_1)):
        if bitstring_1[n] != bitstring_2[n]:
            distance += 1
    return (distance / len(bitstring_1)) * (1.5 * np.mean(score_2))


def to_bitstring(chromosome: int, length: int) -> str:
    """converts and integer into a bitstring.

    Args:
        chromosome (int): integer to convert to bitstring
        length (int): number of bits total

    Returns:
        str: bitstring
    """
    return bin(chromosome)[2:].zfill(length)


def distance_analysis(
    model_and_score_table: dict, num_models_returned: int, num_features: int
) -> list[int]:
    """_summary_

    Args:
        model_and_score_table (dict): a dictionary containing chromosomes as keys, and
            skill scores as values.
        num_models_returned (int): number of models to be returned by this analysis
        num_features (int): number of features in the overall evaluation

    Returns:
        list[int]: a list of chromosomes corresponding to the best unique models
            from the input table.
    """

    # list of chromosomes and another list of scores
    chromosomes = list(model_and_score_table.keys())
    scores = list(map(lambda c: model_and_score_table[c], chromosomes))

    # Create the bitstrings for each chromosome
    bitstrings = [to_bitstring(chromosome, num_features) for chromosome in chromosomes]

    # Create outputs
    output_chromosomes = []
    output_bitstrings = []

    # Iterate and add models to the output
    # The best performing model is added first
    for n in range(num_models_returned):

        if n == 0:

            # store the best performing model in the output
            sorted_scores = sorted(scores, reverse=True)
            best_chromosome_idx = scores.index(
                sorted_scores[0]
            )  # find the index of the best score
            scores.pop(best_chromosome_idx)
            output_chromosomes.append(chromosomes.pop(best_chromosome_idx))
            output_bitstrings.append(bitstrings.pop(best_chromosome_idx))

        else:

            # find the most different / best performing chromosome compared
            # to those already in the output to store.
            distances = np.full(shape=(len(chromosomes)), fill_value=0.0)
            for chromosome_idx in range(len(output_chromosomes)):
                distances += np.array(
                    [
                        weighted_hamming_distance(
                            bitstring_1=output_bitstrings[chromosome_idx],
                            bitstring_2=bitstrings[i],
                            score_2=scores[i],
                        )
                        for i in range(len(bitstrings))
                    ]
                )
            most_distant_idx = np.argmax(distances)
            scores.pop(most_distant_idx)
            output_chromosomes.append(chromosomes.pop(most_distant_idx))
            output_bitstrings.append(bitstrings.pop(most_distant_idx))

    return output_chromosomes
