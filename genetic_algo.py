import numpy as np
import time
from itertools import combinations  
from scipy import stats

from data import chi2_fn, n_obs, n_ops, χ2_SM

debut = time.time()

SEED = 42
np.random.seed(SEED)
print(f"Using seed = {SEED}")


N_STEPS = 10
N = 6
valid_solutions = {}
delta_chi2_min = 140
rng = np.random.default_rng(SEED)

def genetic_algorithm(population_size=25, n_generations=600, elite_size=2, tournament_size=3, mutation_rate=0.2, crossover_rate = 0.3):
    """
    Genetic algorithm for fixed-cardinality operator selection.

    Parameters
    ----------
    population_size : (int)   Number of candidate subsets per generation.
    n_generations   : (int)   Number of evolutionary generations.
    elite_size      : (int)   Best candidates copied unchanged to the next generation.
    tournament_size : (int)   Number of candidates compared during parent selection.
    crossover_rate  : (float) Probability of crossing over two parents.
    mutation_rate   : (float) Probability of applying a swap mutation to an offspring.
    """

    seen_sets, seen_subsets = set(), set()
    def evaluate(subset):
        chi2 = chi2_fn(tuple(sorted(subset)))
        if χ2_SM - chi2 >= delta_chi2_min and subset not in seen_sets:
            seen_sets.add(subset)

            for subset in list(combinations(subset, N)):
                subset = tuple(sorted(subset))
                if subset not in valid_solutions and subset not in seen_subsets:
                    seen_subsets.add(subset)
                    chi2_new = chi2_fn(subset)
                    if stats.chi2.sf(chi2_new, n_obs-N)>0.1: valid_solutions[subset] = χ2_SM - chi2_new
        return chi2

    def crossover(parent1, parent2):
        # Uniform crossover on binary operator membership.
        child = { i for i in range(n_ops) if rng.random() < (0.5 * (i in set(parent1)) + 0.5 * (i in set(parent2)))}

        # Repair chromosome so it has exactly subset_size operators.
        if len(child) > N_STEPS: child.difference_update(rng.choice(sorted(child), size=len(child) - N_STEPS, replace=False).tolist())
        elif len(child) < N_STEPS: child.update(rng.choice(np.array(sorted(set(range(n_ops)) - child), dtype=int), size=N_STEPS-len(child), replace=False).tolist())

        return tuple(sorted(child))

    def mutate(subset):
        # Swap one selected operator for one unselected operator.
        remove_op, add_op = int(rng.choice(list(subset))), int(rng.choice(list(set(range(n_ops)) - set(subset))))

        child = set(subset)
        child.remove(remove_op); child.add(add_op)

        return tuple(sorted(child))

    # Create initial population of distinct subsets
    population = []
    while len(population) < population_size:
        candidate = tuple(sorted(rng.choice(n_ops, size=N_STEPS, replace=False).tolist())) #random
        if candidate not in population: population.append(candidate)#; seen.add(candidate)

    best_chi2 = float("inf")

    for generation in range(n_generations):
        if generation%10==0:print("generation: ", generation)
        scores = np.array([evaluate(s) for s in population])
        order = np.argsort(scores)
        best_chi2 = min(best_chi2, float(scores[order[0]]))
        
        next_population = [population[i] for i in order[:elite_size]] # Elitism: preserve the best individuals unchanged

        # Generate offspring.
        attempts = 0
        while len(next_population) < population_size:
            attempts += 1
            # If the population space is saturated, duplicates are allowed rather than looping indefinitely.
            if attempts > population_size * 5:
                next_population.append(tuple(sorted(rng.choice(n_ops, size=(N_STEPS, population_size-len(next_population)), replace=False).tolist()))); continue

            # Tournament selection for parents: lower chi2 is better.
            len_pop = len(population)
            parent1 = population[min(rng.choice(len_pop, size=min(tournament_size, len_pop), replace=False), key=lambda i: scores[i])]
            parent2 = population[min(rng.choice(len_pop, size=min(tournament_size, len_pop), replace=False), key=lambda i: scores[i])]
            # child is cross over between 2 parents chosen in current population
            child = crossover(parent1, parent2) if rng.random() < crossover_rate else tuple(sorted(parent1))
            if rng.random() < mutation_rate: child = mutate(child) # mutate one ops with prob mutation_rate

            if child not in next_population: next_population.append(child) # only add a child if not already in the population

        population = next_population # update population

    return {"best_delta_chi2": χ2_SM-best_chi2, "n_valid": len(valid_solutions), "valid_sol": valid_solutions, "n_eval": population_size * n_generations}


print(genetic_algorithm())
input(f"Total computation time :{time.time()-debut} sec")
