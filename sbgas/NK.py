import numpy as np

"""
This code was written by eduardo not pat!!!! Someday pat will understand this
but that day is not today!!!!
"""


def ind2gen(index, n):
    '''Return a genotype of length n that encodes the index in binary'''
    # For example, ind2gen(255,8) = [1, 1, 1, 1, 1, 1, 1, 1]
    genotype = np.zeros(n)
    if index >= 2**n:
        print("ind2gen error")
        return genotype
    while n > 0:
        n = n - 1
        if index % 2 == 0:
            genotype[n] = 0
        else:
            genotype[n] = 1
        index = index // 2
    return genotype


def gen2ind(genotype):
    '''Return the index encoded in the genotype'''
    # For example, gen2ind([1,1,1,1,1,1,1,1]) = 255
    i = 0
    index = 0
    mg = len(genotype)
    while i < mg:
        index += genotype[i]*(2**(mg-i-1))
        i += 1
    return int(index)


class Landscape:
    '''Create a tunably rugged landscape with N dimensions and K epistatic interactions'''

    def __init__(self, n, k, random_seed=None):
        self.n = n
        self.maxfit = 0.0
        self.minfit = 1000000
        if not random_seed:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(random_seed)
        # Create random matrix for interactions
        # For example, when N = 5 and K = 1, there are 5 rows each with 4 columns
        # to score each combination of an individual and its neighbor (i.e. 01, 10, 11, 00)
        self.interactions = self.rng.uniform(size=n*(2**(k+1))).reshape(n, 2**(k+1))
        self.fit_table = np.zeros(2**n)
        self.visited_table = np.zeros(2**n)
        self.best = 0
        # Figure out fitness for each possible solution
        for solution in range(2**n):
            fit = 0
            genotype = ind2gen(solution, n)
            # Calculate contribution of each gene in the current solution
            for gene in range(n):
                subgen = []
                for nbr in range(k+1):    # Identify neighbors
                    nbr_ind = (gene+nbr) % n
                    subgen.append(genotype[nbr_ind])
                # Calculate epistatic interactions with each neighbor
                ind = gen2ind(subgen)
                fit += self.interactions[gene][ind]
            self.fit_table[solution] = fit
            if fit > self.maxfit:
                self.maxfit = fit
                self.best = genotype
            if fit < self.minfit:
                self.minfit = fit
        self.fit_table = (self.fit_table - self.minfit) / \
            (self.maxfit-self.minfit)  # Normalize
        self.fit_table = self.fit_table**8    # Scale

    def fitness(self, genotype):
        '''Return the fitness of a solution.'''
        index = gen2ind(genotype)
        return self.fit_table[index]
