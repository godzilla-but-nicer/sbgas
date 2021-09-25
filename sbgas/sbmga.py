import numpy as np


class DiscreteSBMGA:
    def __init__(self, fitness_function, pop_size, genome_size, mut_prob,
                 rec_prob, sb_size, dorm_prob, deme_size, alphabet, random_seed=None):
        """
        lets do it babyyyyyy
        """
        # core parameters
        self.fitness = fitness_function
        self.population = np.zeros((pop_size, genome_size))
        # we're allowed not to have a seed bank
        if sb_size > 0:
            self.seed_bank = np.zeros((sb_size, genome_size))
        else:
            self.seed_bank = None
        
        # more core parameters
        self.alphabet = alphabet
        if deme_size == 0 or deme_size > (pop_size - 1):
            self.deme_size = pop_size - 1
        else:
            self.deme_size = deme_size

        # rates
        self.pmutate = mut_prob
        self.pinfect = rec_prob
        self.pdormant = dorm_prob

        # rng
        if not random_seed:
            self.rng = np.random.default_rng(random_seed)
        else:
            self.rng = np.random.default_rng(random_seed)

    
    def _tournament(self, i, j):
        """
        This function will implement one round of binary competition
        """
        fi = self.fitness(self.population[i])
        fj = self.fitness(self.population[j])

        if fi >= fj:
            win = i
            lose = j
        else:
            win = j
            lose = i
        
        # replace loser with a seed bank genome?
        if self.rng.uniform() < self.pdormant and self.seed_bank is not None:
            ri = self.rng.choice(self.seed_bank.shape[0])
            temp = self.population[lose].copy()
            self.population[lose] = self.seed_bank[ri]
            self.seed_bank[ri] = temp

        # the loser gets infected and mutated
        else:
            for gene in range(self.population.shape[1]):
                # infect loser
                if self.rng.uniform() < self.pinfect:
                    self.population[lose, gene] = self.population[win, gene]
                # mutate loser
                elif self.rng.uniform() < self.pmutate:
                    # we dont want to mutate to the same thing
                    mutations = [a for a in self.alphabet if a != self.population[lose, gene]]
                    self.population[lose, gene] = self.rng.choice(mutations)
        
        return win, lose

    
    def _pick_competitors(self, focal_i = None):
        """
        just returns indices for the competion while obeying the deme. Can set the focal individuals index for testing purposes
        """
        if not focal_i:
            i = self.rng.choice(self.population.shape[0])
        else:
            i = focal_i
        offset = self.rng.choice(range(1, self.deme_size + 1)) # would it be cool to weigh this by proximity?
        j = (i + offset) % self.population.shape[0]
        return i, j


    def run_evolution(self, generations, random_start=True):
        """
        This does the whole dang thing
        """

        # initialize population and seed bank
        if random_start:
            self.population = self.rng.choice(self.alphabet, size=self.population.shape)
            if self.seed_bank is not None:
                self.seed_bank = self.rng.choice(self.alphabet, size=self.seed_bank.shape)

        
        # initialize some arrays to track fitnesses
        self.fitness_history = np.zeros((generations, self.population.shape[0]))

        for gen in range(generations):
            # first we will get all of the fitness values
            for indiv in range(self.population.shape[0]):
                fit = self.fitness(self.population[indiv])
                self.fitness_history[gen, indiv] = fit

            # then we can do the tournament
            i, j = self._pick_competitors()
            self._tournament(i, j)
        
        return self.fitness_history
