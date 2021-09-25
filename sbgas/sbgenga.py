import numpy as np
from collections.abc import Iterable

class SBGenGA:

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

        