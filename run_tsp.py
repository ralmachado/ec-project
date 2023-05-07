import random
from pathlib import Path

import numpy as np
from tqdm import trange

from base_config import params
from evolution import Evolution, utils, tsp, mutation

if __name__ == '__main__':
    # Read seeds
    seed_file = Path.cwd() / "seeds.txt"
    seeds = utils.read_seeds(seed_file)

    # Read TSP file
    tsp_file = Path.cwd() / "att48_dist.txt"
    tsp_matrix = tsp.read_tsp(tsp_file)

    # Set save prefix and ensure directory exists
    path = Path.cwd() / "tsp"
    path.mkdir(exist_ok=True)

    # Problem and evolution parameters
    runs = 30

    # Customize parameters for TSP
    params.update({
        "domain": [[0, 1]] * len(tsp_matrix),
        "prob_mutation": 1 / len(tsp_matrix),
        "fitness_fn": tsp.fitness(tsp_matrix),
    })

    # Mutation parameter search values
    configs = {
        "gauss": [2 ** i for i in range(-5, 6)],
        "delta": [2 ** i for i in range(-5, 6)]
    }

    # Run the evolutionary algorithm
    for _mutation, search in configs.items():
        current_path = path / _mutation
        current_path.mkdir(exist_ok=True)
        for param in search:
            # Skip already finished run
            best_file = current_path / f"{_mutation}_{param:1g}_best.txt"
            avg_file = current_path / f"{_mutation}_{param:1g}_avg.txt"
            boe_file = current_path / f"{_mutation}_{param:1g}_boe.txt"
            if best_file.exists() and avg_file.exists() and boe_file.exists():
                print(f"Skipping {_mutation}_mutation({'b' if _mutation == 'delta' else 'sigma'}={param})...")
                continue

            # Update mutation function with current parameter
            if _mutation == "gauss":
                print(f"gauss_mutation(sigma={param})")
                params.update(mutation_fn=mutation.gauss_mutation(sigma=param))
            elif _mutation == "delta":
                print(f"delta_mutation(b={param})")
                params.update(mutation_fn=mutation.delta_mutation(b=param))

            # Create performance metric storage
            avg_by_gen = np.empty((runs, params["max_gen"] + 1))
            best_by_gen = np.empty((runs, params["max_gen"] + 1))
            best_on_end = np.empty(runs)

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
            np.savetxt(best_file, overall_best_by_gen)
            np.savetxt(avg_file, overall_avg_by_gen)
            np.savetxt(boe_file, best_on_end)
