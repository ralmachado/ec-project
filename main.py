import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import trange

from evolution import Evolution, optimization, mutation, selection, crossover


if __name__ == '__main__':
    # Problem and evolution parameters
    runs = 1
    dimensions = 10
    params = {
        "max_gen": 200,
        "population_size": 200,
        "domain": [[-5.12, 5.12]] * dimensions,
        "prob_mutation": 1 / dimensions,
        "prob_crossover": 0.7,
        "crossover_fn": crossover.arithmetical_cross(alpha=0.2),
        "mutation_fn": mutation.delta_mutation(b=2),
        "parents_fn": selection.tour_sel(3),
        "survivors_fn": selection.elitism(0.1),
        "fitness_fn": optimization.rastrigin,
        "maximisation": False,
    }

    # Create performance metric storage
    avg_by_gen = np.empty((runs, params["max_gen"] + 1))
    best_by_gen = np.empty((runs, params["max_gen"] + 1))

    # Run the evolutionary algorithm
    for run in trange(runs):
        evolution = Evolution(**params)
        avg_by_gen[run, 0] = evolution.avg_fitness
        best_by_gen[run, 0] = evolution.best_fitness

        for gen in evolution:
            avg_by_gen[run, gen] = evolution.avg_fitness
            best_by_gen[run, gen] = evolution.best_fitness

    overall_avg_by_gen = np.mean(avg_by_gen, axis=0)
    overall_best_by_gen = np.mean(best_by_gen, axis=0)

    # Plot performance
    sns.set_theme()
    plt.plot(overall_avg_by_gen, label='Average')
    plt.plot(overall_best_by_gen, label='Best')
    plt.title(f"Rastrigin ({'maximisation' if params['maximisation'] else 'minimisation'})")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()
