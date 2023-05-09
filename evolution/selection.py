import random
from operator import itemgetter


# Tournament Selection
def tour_sel(t_size):
    def tournament(pop, maximisation: bool = False):
        size_pop = len(pop)
        mate_pool = []
        for i in range(size_pop):
            pool = random.sample(pop, t_size)
            pool.sort(key=itemgetter(1), reverse=maximisation)
            mate_pool.append(pool[0])
        return mate_pool

    return tournament


# Survivals: elitism
def elitism(elite):
    def _elitism(parents, offspring, maximisation: bool = False):
        size = len(parents)
        comp_elite = int(size * elite)
        offspring.sort(key=itemgetter(1), reverse=maximisation)
        parents.sort(key=itemgetter(1), reverse=maximisation)
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population

    return _elitism
