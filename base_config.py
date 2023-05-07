from evolution import crossover, selection

params = {
    "max_gen": 1500,
    "population_size": 200,
    "prob_crossover": 0.7,
    "crossover_fn": crossover.arithmetical_cross(alpha=0.4),
    "parents_fn": selection.tour_sel(3),
    "survivors_fn": selection.elitism(0.1),
    "maximisation": False,
}

# full_config = {
#     "max_gen": 1500,
#     "population_size": 200,
#     # "domain": [[-5.12, 5.12]] * dimensions,
#     # "prob_mutation": 1 / dimensions,
#     "domain": [[0, 1]] * len(tsp_matrix),
#     "prob_mutation": 1 / len(tsp_matrix),
#     "prob_crossover": 0.7,
#     "crossover_fn": crossover.arithmetical_cross(alpha=0.4),
#     "mutation_fn": mutation.delta_mutation(b=0.5),
#     # "mutation_fn": mutation.gauss_mutation(sigma=0.2),
#     "parents_fn": selection.tour_sel(3),
#     "survivors_fn": selection.elitism(0.1),
#     "fitness_fn": tsp.fitness(tsp_matrix),
#     "maximisation": False,
# }
