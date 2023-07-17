import random

from networkx.generators.random_graphs import fast_gnp_random_graph


def AssertValuesAreClose(value1, value2, tolerance=0.0001):
    if abs(value1 - value2) > tolerance:
        raise AssertionError("Values {} and {} are not close".format(value1, value2))


def create_graph(size, p, seed):
    G = fast_gnp_random_graph(size, p, seed=seed, directed=True)
    return G


def create_weighted_graph(size, p, seed):
    random.seed(seed)
    G = fast_gnp_random_graph(size, p, seed=seed, directed=True)
    for u, v in G.edges():
        G.edges[u, v]["weight"] = random.randint(1, 10)
    return G
