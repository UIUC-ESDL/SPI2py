# import networkx as nx
# import pickle
# import collections
# from cypari import pari
# import itertools
# import subprocess
# import string
# import io
# import time





# def shadows_via_plantri_by_ascii(num_tri_verts, num_crossings):
#     assert num_tri_verts % 2 == 0
#     vertices = num_tri_verts + num_crossings
#     edges = (3 * num_tri_verts + 4 * num_crossings) // 2
#     faces = 2 - vertices + edges
#     cmd = ['./plantri',
#            '-p -d',  # simple planar maps, but return the dual
#            '-f4',  # maximum valence in the returned dual is <= 4
#            '-c1',  # graph should be 1-connected
#            '-m2',  # no valence 1 vertices = no loops in the dual
#            '-a',  # return in ascii format
#            '-e%d' % edges,
#            '%d' % faces]
#     proc = subprocess.run(' '.join(cmd), shell=True, text=True, capture_output=True)
#     ans = []
#     for line in proc.stdout.splitlines():
#         graph_data = line.split()[1].split(',')
#         counts = collections.Counter(len(v) for v in graph_data)
#         assert counts[3] == num_tri_verts and counts[4] == num_crossings
#         ans.append(graph_data)
#     return ans


# def read_edge_code(stream, size):
#     """
#     Read 1 byte form of edge code
#     """
#     ans = [[]]
#     for _ in range(size):
#         i = int.from_bytes(stream.read(1), 'big')
#         if i < 255:
#             ans[-1].append(i)
#         else:
#             ans.append([])
#     return ans


# def shadows_via_plantri_by_edge_codes(num_tri_verts, num_crossings):
#     assert num_tri_verts % 2 == 0
#     vertices = num_tri_verts + num_crossings
#     edges = (3 * num_tri_verts + 4 * num_crossings) // 2
#     faces = 2 - vertices + edges
#     cmd = ['./plantri',
#            '-p -d',  # simple planar maps, but return the dual
#            '-f4',  # maximum valence in the returned dual is <= 4
#            '-c1',  # graph should be 1-connected
#            '-m2',  # no valence 1 vertices = no loops in the dual
#            '-E',  # return binary edge code format
#            '-e%d' % edges,
#            '%d' % faces]
#     proc = subprocess.run(' '.join(cmd), shell=True, capture_output=True)
#     stdout = io.BytesIO(proc.stdout)
#     assert stdout.read(13) == b'>>edge_code<<'
#     shadows = []
#     while True:
#         b = stdout.read(1)
#         if len(b) == 0:
#             break
#         size = int.from_bytes(b, 'big')
#         assert size != 0
#         shadows.append(read_edge_code(stdout, size))
#
#     return shadows


# class Shadow:
#     def __init__(self, edge_codes):
#         self.edge_codes = edge_codes
#         self.vertices = [edges for edges in edge_codes if len(edges) == 3]
#         self.crossings = [edges for edges in edge_codes if len(edges) == 4]
#         self.num_edges = sum(len(edges) for edges in edge_codes) // 2
#
#     def spatial_graph_diagram(self, signs=None, check=True):
#         num_cross = len(self.crossings)
#         if signs is None:
#             signs = num_cross * [0]
#         else:
#             assert len(signs) == num_cross
#
#         classes = [Edge(i) for i in range(self.num_edges)]
#         for v, edges in enumerate(self.vertices):
#             d = len(edges)
#             V = Vertex(d, 'V%d' % v)
#             classes.append(V)
#             for i, e in enumerate(edges):
#                 E = classes[e]
#                 e = 0 if E.adjacent[0] is None else 1
#                 V[i] = E[e]
#
#         for c, edges in enumerate(self.crossings):
#             C = Crossing('C%d' % c)
#             classes.append(C)
#             for i, e in enumerate(edges):
#                 E = classes[e]
#                 e = 0 if E.adjacent[0] is None else 1
#                 C[(i + signs[c]) % 4] = E[e]
#
#         return SpatialGraphDiagram(classes, check=check)


# def spatial_graph_diagrams_from_shadow(shadow, signs):
#    assert len(signs

# TODO Compile Plantri for Windows
# def spatial_graph_diagrams_fixed_crossings(G, crossings):
#     """
#     Let's start with the theta graph
#
#     >>> T = nx.MultiGraph(3*[(0, 1)])
#     >>> len(list(spatial_graph_diagrams_fixed_crossings(T, 3)))
#     2
#     """
#     assert all(d == 3 for v, d in G.degree)
#     assert all(a != b for a, b in G.edges())
#
#     raw_shadows = shadows_via_plantri_by_edge_codes(G.number_of_nodes(), crossings)
#
#     for raw_shadow in raw_shadows:
#         shadow = Shadow(raw_shadow)
#         diagram = shadow.spatial_graph_diagram(check=False)
#         U = diagram.underlying_graph()
#         if U is not None:
#             if nx.is_isomorphic(G, U):
#                 if not diagram.has_R6():
#                     num_cross = len(shadow.crossings)
#                     if num_cross == 0:
#                         yield diagram
#                     else:
#                         for signs in itertools.product((0, 1), repeat=num_cross - 1):
#                             signs = (0,) + signs
#                             D = shadow.spatial_graph_diagram(signs=signs, check=False)
#                             if not D.has_R2():
#                                 yield D


# def get_coefficients_and_exponents(poly):
#     # Assumes all denominators are only A**n with no coefficient
#     coefficients = poly.numerator().Vec()
#     coeff_len = len(coefficients)
#
#     exponents = []
#     degree = poly.poldegree()
#
#     for _ in range(coeff_len):
#         exponents.append(degree)
#         degree -= 1
#
#     return coefficients, exponents


# def reverse_poly(poly):
#     """
#     >>> reverse_poly(A**-1 + 2)
#     A + 2
#     """
#
#     A = pari('A')
#
#     coeffs, exps = get_coefficients_and_exponents(poly)
#
#     ans = pari(0)
#
#     for c, e in zip(coeffs, exps):
#         ans += c * A ** (-e)
#
#     return ans


# def normalize_poly(poly):
#     """
#     The Yamada polynomial is only defined up to a power of (-A)^n.
#
#     Also, we don't want to distinguish between a spatial graph and its
#     mirror image, which corresponds to interchanging A <-> A^(-1).
#     """
#     _, exps = get_coefficients_and_exponents(poly)
#     a, b = min(exps), max(exps)
#     ans1 = (-A) ** (-a) * poly
#     ans2 = (-A) ** b * reverse_poly(poly)
#     return min([ans1, ans2], key=list)


# def enumerate_yamada_classes(G, max_crossings):
#     examined = 0
#     polys = dict()
#     for crossings in range(0, max_crossings + 1):
#         for D in spatial_graph_diagrams_fixed_crossings(G, crossings):
#             p = D.yamada_polynomial()
#             p = normalize_poly(p)
#             if p not in polys:
#                 polys[p] = D
#             examined += 1
#     return polys, examined


# def to_poly(diagram):
#     p = diagram.yamada_polynomial()
#     p = normalize_poly(p)
#     return p, diagram


# def enumerate_yamada_classes_multicore(G, max_crossings, pool):
#     examined = 0
#     polys = dict()
#     timings = dict()
#     for crossings in range(0, max_crossings + 1):
#         start = time.time()
#         diagrams = spatial_graph_diagrams_fixed_crossings(G, crossings)
#         some_polys = pool.imap_unordered(to_poly, diagrams)
#         for p, D in some_polys:
#             if p not in polys:
#                 polys[p] = D
#             examined += 1
#         timings[crossings] = time.time() - start
#     return polys, timings, examined


# def pickle_yamada(data, filename):
#     with open(filename, 'wb') as file:
#         pickle.dump(data, file)


# def load_yamada(filename):
#     with open(filename, 'rb') as file:
#         return pickle.load(file)