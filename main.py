import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import trange

from evolution import Evolution, optimization, mutation, selection, crossover, utils, tsp


if __name__ == '__main__':
    # Read seeds
    seed_file = Path.cwd() / "seeds.txt"
    seeds = utils.read_seeds(seed_file)

    # Read TSP file
    tsp_file = Path.cwd() / "att48_dist.txt"
    tsp_matrix = tsp.read_tsp(tsp_file)

    # Problem and evolution parameters
    runs = 30
    # dimensions = 50

    sigma_search = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    b_search = [2 ** i for i in range(-5, 6)]
    params = {
        "max_gen": 1500,
        "population_size": 200,
        # "domain": [[-5.12, 5.12]] * dimensions,
        # "prob_mutation": 1 / dimensions,
        "domain": [[0, 1]] * len(tsp_matrix),
        "prob_mutation": 1 / len(tsp_matrix),
        "prob_crossover": 0.7,
        "crossover_fn": crossover.arithmetical_cross(alpha=0.4),
        "mutation_fn": mutation.delta_mutation(b=0.5),
        # "mutation_fn": mutation.gauss_mutation(sigma=0.2),
        "parents_fn": selection.tour_sel(3),
        "survivors_fn": selection.elitism(0.1),
        "fitness_fn": tsp.fitness(tsp_matrix),
        "maximisation": False,
    }

    path = Path.cwd() / "tsp"
    path.mkdir(exist_ok=True)

    # Run the evolutionary algorithm
    for b in b_search:
        params.update(mutation_fn=mutation.delta_mutation(b))

        # Create performance metric storage
        avg_by_gen = np.empty((runs, params["max_gen"] + 1))
        best_by_gen = np.empty((runs, params["max_gen"] + 1))
        for run in trange(runs):
            random.seed(seeds[run])
            evolution = Evolution(**params)
            avg_by_gen[run, 0] = evolution.avg_fitness
            best_by_gen[run, 0] = evolution.best_fitness

            for gen in evolution:
                avg_by_gen[run, gen] = evolution.avg_fitness
                best_by_gen[run, gen] = evolution.best_fitness
            best_on_end[run] = evolution.best_fitness

        overall_avg_by_gen = np.mean(avg_by_gen, axis=0)
        overall_best_by_gen = np.mean(best_by_gen, axis=0)
        np.savetxt(path / f"delta_{b:1g}_best", overall_best_by_gen)
        np.savetxt(path / f"delta_{b:1g}_avg", overall_avg_by_gen)

    # Plot performance
    # sns.set_theme()
    # plt.plot(overall_avg_by_gen, label='Average')
    # plt.plot(overall_best_by_gen, label='Best')
    # plt.title(f"TSP ({'maximisation' if params['maximisation'] else 'minimisation'})")
    # plt.xlabel("Generations")
    # plt.ylabel("Fitness")
    # plt.legend()
    # plt.show()
