import random


def uniform_mutation(genotype, prob, domain, **kwargs):
    for i in range(len(genotype)):
        if random.random() < prob:
            genotype[i] = random.uniform(*domain[i])
    return genotype


def gauss_mutation(sigma):
    def _gauss_mutation(genotype, prob, domain, **kwargs):
        for i in range(len(genotype)):
            if random.random() < prob:
                genotype[i] += random.gauss(0, sigma)
                if genotype[i] < domain[i][0]:
                    genotype[i] = domain[i][0]
                if genotype[i] > domain[i][1]:
                    genotype[i] = domain[i][1]
        return genotype

    return _gauss_mutation


# Delta Mutation
def delta_mutation(b):
    def _delta_mutation(genotype, prob_mutation, domain, t, T, **kwargs):
        for i in range(len(genotype)):
            if random.random() < prob_mutation:
                choice = round(random.uniform(0, 1), 0)
                if choice == 0:
                    y = domain[i][1] - genotype[i]
                else:
                    y = genotype[i] - domain[i][0]
                delta = y * random.uniform(0, 1) * (1 - (t / T)) ** b
                if choice == 0:
                    genotype[i] = genotype[i] + delta
                else:
                    genotype[i] = genotype[i] - delta
        return genotype

    return _delta_mutation
