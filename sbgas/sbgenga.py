import numpy as np
from collections.abc import Iterable

class DiscreteSBGenGA:

    def __init__(self, fitness_function, population_size, genome_size,
                 mutation_prob, recombination_prob, seed_bank_size, dormant_prob,
                 deme_size, alphabet, random_seed = None):
        
        # basic params
        self.fitness = fitness_function
        self.population = np.zeros((population_size, genome_size), dtype=int)
        self.deme_size = deme_size
        self.pmutate = mutation_prob
        self.precomb = recombination_prob

        # seed bank params
        self.seed_bank = np.zeros((seed_bank_size, genome_size))
        self.pdormant = dormant_prob

        # possible states
        self.alphabet = alphabet

        # rng
        if random_seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(random_seed)

    
    def _generation(self, num_winners: int, winners: Iterable[int] = None) -> np.array:

        if winners is None:
            # we will select winners with probability proportional to their
            fitness_values = np.zeros(self.population.shape[0])
            for i, genome in enumerate(self.population):
                fitness_values[i] = self.fitness(genome)
            
            # normalize to be a probability distribution
            win_prob = fitness_values / np.sum(fitness_values)

            winners = self.rng.choice(self.population.shape[0],
                                      size=num_winners,
                                      p=win_prob)
        
        elif num_winners % 2 != 0:
            raise ValueError(
                'num_winners must be even for reproduction, value given {} is odd'.format(num_winners))

        # need to randomly pair the winners for recombination
        winners_perm = winners.copy()
        self.rng.shuffle(winners_perm)

        # we can get pairs by stacking halves of the array after permutation
        half_idx = int(winners.shape[0] / 2)
        pairs = np.vstack((winners_perm[:half_idx], winners_perm[half_idx:]))
        
        # I guess we're just going to accept that we replace only 
        # num_winner / 2 "losers" for now. one per "sexual pair"

        # also assuming fitness values are in [0, 1] does this break mny tests?
        unfitness_values = 1 - fitness_values     
        unfitness_values[winners] = 0  # can't pick winners as losers
        lose_prob =  unfitness_values / np.sum(unfitness_values)

        losers = self.rng.choice(self.population.shape[0],
                                 size = pairs.shape[0],
                                 p=lose_prob)

        
