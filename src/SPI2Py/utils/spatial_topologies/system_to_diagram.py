import networkx as nx
from cypari import pari

import string
# Warning snappy
from yamada_polynomials import Edge, Vertex, Crossing, SpatialGraphDiagram, normalize_poly
from snappy.exterior_to_link.link_projection import fig8_points, project_to_diagram

alphabet = list(string.ascii_uppercase)


A, B, C, D = [Vertex(3, L) for L in 'ABCD']
X, Y, Z = [Crossing(L) for L in 'XYZ']

A[0], A[1], A[2] = D[0], B[2], X[2]
B[0], B[1] = C[0], X[3]
C[1], C[2] = D[2], Z[0]
D[1] = Z[1]
X[0], X[1] = Y[3], Y[2]
Y[0], Y[1] = Z[3], Z[2]

D = SpatialGraphDiagram([A, B, C, D, X, Y, Z])

d_yamada = normalize_poly(D.yamada_polynomial())

print("Omega-2 Graph Yamada Polynomial:", d_yamada )






f8 = fig8_points()
d8 = project_to_diagram(f8)
print(d8.PD_code())

crossings = []
pd_codes = d8.PD_code()

for i, pd_code in enumerate(pd_codes):
    print('crossing:', pd_code)

    crossings.append(Crossing(alphabet[i]))





