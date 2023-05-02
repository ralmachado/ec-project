"""
Simple Evolutionary Algorithm

Provides a modular, iterator-style, Evolution class, allowing for customizable performance
metrics computation as needed in the evolution loop.
"""

import random
from dataclasses import dataclass
from operator import itemgetter
from typing import Callable

from numpy import mean


@dataclass
class Evolution:
    max_gen: int
    population_size: int
    domain: list[list[float]]
    prob_mutation: float
    prob_crossover: float
    crossover_fn: Callable
    mutation_fn: Callable
    parents_fn: Callable
    survivors_fn: Callable
    fitness_fn: Callable
    maximisation: bool = False

    def __post_init__(self):
        # TODO add more population initialization methods if needed
        # Uniform population initialization
        self.population = [[random.uniform(*param) for param in self.domain] for _ in range(self.population_size)]
        self.population = [[individual, self.fitness_fn(individual)] for individual in self.population]
        self.gen = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.gen == self.max_gen:
            raise StopIteration

        self.gen += 1

        # Parent selection and Crossover
        mating_pool = self.parents_fn(self.population, maximisation=self.maximisation)
        parents = []
        for i in range(self.population_size - 1):
            x1 = mating_pool[i]
            x2 = mating_pool[i + 1]
            xover = self.crossover_fn(x1, x2, self.prob_crossover)
            if xover is not None:
                parents.extend([[x, self.fitness_fn(x)] for x in xover])
            else:
                parents.extend([x1, x2])
            if len(parents) == self.population_size:
                break

        # Mutation
        children = []
        for genotype, _ in parents:
            #new_genotype = self.mutation_fn(genotype, self.prob_mutation, self.domain)
            new_genotype = self.mutation_fn(self.gen, genotype, 10, self.max_gen, 0.1, self.prob_mutation)
            children.append([new_genotype, self.fitness_fn(new_genotype)])
        # Survivor selection
        self.population = self.survivors_fn(self.population, children, maximisation=self.maximisation)
        return self.gen

    @property
    def avg_fitness(self):
        """
        Get current population's average fitness
        """
        return mean([fit for _, fit in self.population])

    @property
    def best_fitness(self):
        """
        Get current population's best fitness
        """
        return sorted(self.population, key=itemgetter(1), reverse=self.maximisation)[0][1]
