import numpy as np
from numpy.core.fromnumeric import shape
from sbgas.sbmga import DiscreteSBMGA

# test parameters
FFUNC = np.sum
POPSIZE = 5
GENESIZE = 3
MUTPROB = 0.01
INFECTPROB = 0.01
SBSIZE = 2
DORMPROB = 0.01
DEMESIZE = 3
ALPHABET = [0, 1]


ga = DiscreteSBMGA(FFUNC, POPSIZE, GENESIZE, MUTPROB, INFECTPROB, SBSIZE, DORMPROB, DEMESIZE, ALPHABET)

def test_init():
    # get a bunch of flags
    pop_shape = ga.population.shape == (POPSIZE, GENESIZE)
    gene_probs = ga.pmutate == MUTPROB and ga.pinfect == INFECTPROB
    sb_shape = ga.seed_bank.shape == (SBSIZE, GENESIZE)
    dorm_prob = ga.pdormant == DORMPROB
    deme = ga.deme_size == DEMESIZE
    alph = ga.alphabet == ALPHABET
    # make sure all flags are true
    assert pop_shape and gene_probs and sb_shape and dorm_prob and deme and alph

def test_tournament_basic():
    # set so genome 1 is better than genome 0
    ga.population[1] = np.array([0, 0, 1])
    winner_index, loser_index = ga._tournament(0, 1)
    assert winner_index == 1 and loser_index == 0

def test_tournament_mutation():
    # pass in a mutation probability of 1, other probs to zero
    ga_mut = DiscreteSBMGA(FFUNC, POPSIZE, GENESIZE, 1, 0, SBSIZE, 0, DEMESIZE, ALPHABET)
    ga_mut._tournament(0, 1)
    assert np.array_equal(ga_mut.population[1], np.ones(GENESIZE))

def test_tournament_infection():
    ga_inf = DiscreteSBMGA(FFUNC, POPSIZE, GENESIZE, 0, 1, SBSIZE, 0, DEMESIZE, ALPHABET)
    ga_inf.population[0] = np.ones(GENESIZE)
    ga_inf._tournament(0, 1)
    assert np.array_equal(ga_inf.population[1], np.ones(GENESIZE))

def test_tournament_dormancy():
    ga_sb = DiscreteSBMGA(FFUNC, POPSIZE, GENESIZE, 0, 0, SBSIZE, 1, DEMESIZE, ALPHABET)
    # set so 0 still wins but we can see the change in the seed bank when 1 enters
    ga_sb.population[0] = np.ones(GENESIZE)
    ga_sb.population[1] = np.array([0, 0, 1])
    ga_sb._tournament(0, 1)
    assert np.array_equal(ga_sb.population[1], np.zeros(GENESIZE)) and np.sum(ga_sb.seed_bank) == 1

def test_pick_competitors():
    ga_comp = DiscreteSBMGA(FFUNC, POPSIZE, GENESIZE, MUTPROB, INFECTPROB, SBSIZE, DORMPROB, 1, ALPHABET)
    i, j = ga_comp._pick_competitors(4)
    assert i == 4 and j == 0

def test_pick_competitors_stochastic():
    i, j = ga._pick_competitors(2)
    assert i == 2 and j in [3, 4, 0]

def test_run_evolution():
    history = ga.run_evolution(500)
    assert np.max(history[-1]) == 3
