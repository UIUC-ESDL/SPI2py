import networkx as nx
from cypari import pari


# Warning snappy
from snappy.exterior_to_link.link_projection import fig8_points, project_to_diagram

f8 = fig8_points()

d8 = project_to_diagram(f8)


print(d8.crossing_strands())
# fig8 = lp.fig8_points()



