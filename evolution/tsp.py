from copy import deepcopy
from pathlib import Path


def read_tsp(file: str | Path):
    """Store the matrix distance file as a list of lists. Symmetric distances. With header!"""
    if not isinstance(file, Path):
        file = Path(file)

    with file.open('r') as f:
        mat_dist = []
        # reader and discard the header
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()
        mat_dist.append([eval(elem) for elem in line.split()])
        # read data , line by line
        for line in f:
            dist_line = [eval(elem) for elem in line.split()]
            mat_dist.append(dist_line)
    return mat_dist


def random_keys(genotype):
    aux = deepcopy(genotype)
    idx = deepcopy(genotype)
    idx.sort()
    permutation = []
    for gene in idx:
        permutation.append(aux.index(gene))
        aux[aux.index(gene)] = None
    return permutation


def get_cost(genotype, matrix_dist):
    """ return the list of distances."""
    fen = []
    for i in range(len(genotype) - 1):
        fen.append(matrix_dist[genotype[i]][genotype[i + 1]])
    fen.append(matrix_dist[genotype[-1]][genotype[0]])
    return fen


def fitness(distance_matrix):
    def _fitness(genotype):
        permutation = random_keys(genotype)
        return sum(get_cost(permutation, distance_matrix))

    return _fitness
