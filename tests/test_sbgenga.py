import numpy as np
from sbgas.sbgenga import SBGenGA

# Paramters for test GA
FFUNC = np.sum
POPSIZE = 4
GENESIZE = 3
MUTPROB = 0.05
SBSIZE = 2
DORMPROB = 0.01
DEMESIZE = 3
ALPHABET = [0, 1]
RANDOMSEED = 666
TESTWINNERS = np.array([0, 1])


def test_init():
    ga = SBGenGA(FFUNC, POPSIZE, GENESIZE, MUTPROB SBSIZE, DORMPROB, DEMESIZE, ALPHABET, RANDOMSEED)
    # get a bunch of flags
    pop_shape = ga.population.shape == (POPSIZE, GENESIZE)
    gene_probs = ga.pmutate == MUTPROB
    sb_shape = ga.seed_bank.shape == (SBSIZE, GENESIZE)
    dorm_prob = ga.pdormant == DORMPROB
    deme = ga.deme_size == DEMESIZE
    alph = ga.alphabet == ALPHABET
    # make sure all flags are true
    assert pop_shape and gene_probs and sb_shape and dorm_prob and deme and alph


def test_generation_basic():
    ga = SBGenGA(FFUNC, POPSIZE, GENESIZE, MUTPROB, SBSIZE, DORMPROB, DEMESIZE, ALPHABET, RANDOMSEED)
    # Set two of the genomes to higher fit genotypes
    ga.population[0] = np.array([0, 0, 1])
    ga.population[1] = np.array([0, 0, 1])
    winners = ga._generation(TESTWINNERS)
    assert np.array_equal(winners, np.array([0, 1])) and np.sum(ga.population) > 2


def test_generation_mutation():
    ga = SBGenGA(FFUNC, POPSIZE, GENESIZE, 1, SBSIZE, 0, DEMESIZE, ALPHABET, RANDOMSEED)
    # set our two genomes again
    ga.population[0] = np.array([0, 0, 1])
    ga.population[1] = np.array([0, 0, 1])
    ga._generation(TESTWINNERS)
    assert np.array_equal(ga.population[2], np.ones(3)) and np.array_equal(ga.population[3], np.ones(3))


def test_generation_recombination():
    ga = SBGenGA(FFUNC, POPSIZE, GENESIZE, 0, SBSIZE, 0, DEMESIZE, ALPHABET, RANDOMSEED)
    # set our two genomes again
    ga.population[0] = np.array([0, 0, 1])
    ga.population[1] = np.array([0, 0, 1])
    ga._generation(TESTWINNERS)
    assert np.array_equal(ga.population[2], np.array([0, 0, 1])) and np.array_equal(ga.population[3], np.array([0, 0, 1]))


def test_generation_dormancy():
    ga = SBGenGA(FFUNC, POPSIZE, GENESIZE, 0, SBSIZE, 1, DEMESIZE, ALPHABET, RANDOMSEED)
    # set our two genomes again
    ga.population[0] = np.array([0, 0, 1])
    ga.population[1] = np.array([0, 0, 1])
    ga._generation(TESTWINNERS)
    assert np.array_equal(ga.seed_bank[0], np.array([0, 0, 1])) and np.array_equal(ga.seed_bank[1], np.array([0, 0, 1]))


def test_pick_competitors():
    ga_comp = SBGenGA(FFUNC, POPSIZE, GENESIZE, MUTPROB, SBSIZE, DORMPROB, 1, ALPHABET)
    i, j = ga_comp._pick_competitors(4)
    assert i == 3 and j == 0


def test_run_evolution():
    ga = SBGenGA(FFUNC, POPSIZE, GENESIZE, MUTPROB, SBSIZE, DORMPROB, DEMESIZE, ALPHABET, RANDOMSEED)
    history = ga.run_evolution(50)
    assert np.max(history[-1]) == 3
