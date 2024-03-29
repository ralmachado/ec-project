import random


def uniform_mutation(genotype, prob, domain, **kwargs):
    for i in range(len(genotype)):
        if random.random() < prob:
            genotype[i] = random.uniform(*domain[i])
    return genotype


def gauss_mutation(sigma):
    def _gauss_mutation(genotype, prob, domain, **kwargs):
        copy = genotype[:]
        for i in range(len(genotype)):
            if random.random() < prob:
                copy[i] += random.gauss(0, sigma)
                if copy[i] < domain[i][0]:
                    copy[i] = domain[i][0]
                elif copy[i] > domain[i][1]:
                    copy[i] = domain[i][1]
        return copy

    return _gauss_mutation


# Delta Mutation
def delta_mutation(b):
    def _delta_mutation(genotype, prob_mutation, domain, t, T, **kwargs):
        copy = genotype[:]
        for i in range(len(copy)):
            if random.random() < prob_mutation:
                choice = round(random.uniform(0, 1), 0)
                if choice == 0:
                    y = domain[i][1] - copy[i]
                else:
                    y = copy[i] - domain[i][0]
                delta = y * random.uniform(0, 1) * (1 - (t / T)) ** b
                if choice == 0:
                    copy[i] = copy[i] + delta
                else:
                    copy[i] = copy[i] - delta
        return copy

    return _delta_mutation
