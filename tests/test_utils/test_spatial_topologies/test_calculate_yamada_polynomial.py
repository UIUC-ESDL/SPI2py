import networkx as nx
from cypari import pari
from src.SPI2Py.utils.spatial_topologies.calculate_yamada_polynomial import has_cut_edge, remove_valence_two_vertices, H_poly



def test_has_cut_edge_1():

    G = nx.MultiGraph(nx.barbell_graph(3, 0))

    assert has_cut_edge(G)


def test_has_cut_edge_1():

    G = nx.MultiGraph(nx.barbell_graph(3, 0))
    ekey = G.add_edge(2, 3)

    assert not has_cut_edge(G)


def test_remove_valence_two_vertices():
    G = nx.MultiGraph([(0, 1), (1, 2), (2, 0)])
    C = remove_valence_two_vertices(G)
    assert list(C.edges()) == [(0, 0)]


def test_H_poly_1():

    G = nx.barbell_graph(3, 0)
    assert H_poly(G) == 0


def test_H_poly_2():

    A = pari('A')
    assert H_poly(nx.MultiGraph([(0, 0)])) == (A**2 + A + 1)/A


def test_H_poly_3():
    A = pari('A')
    assert H_poly(nx.MultiGraph([(0, 1), (1, 2), (2, 0)])) == (A**2 + A + 1)/A


# def test_H_poly_4():
#     ...
#
#
# def test_H_poly_5():
#     ...