import networkx as nx
from cypari import pari


# Warning snappy
from yamada_polynomials import Edge, Vertex, SpatialGraphDiagram
from snappy.exterior_to_link.link_projection import fig8_points, project_to_diagram




A, B = Vertex(3, 'A'), Vertex(3, 'B')

E0, E1, E2 = Edge(0), Edge(1), Edge(2)

A[0], A[1], A[2] = E0[0], E1[0], E2[0]

B[0], B[1], B[2] = E0[1], E2[1], E1[1]

D = SpatialGraphDiagram([A, B, E0, E1, E2])

print("Unknotted Theta Graph Yamada Polynomial:", D.yamada_polynomial() )







f8 = fig8_points()
d8 = project_to_diagram(f8)
print(d8.crossing_strands())
# fig8 = lp.fig8_points()



