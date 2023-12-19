import networkx as nx
g = nx.Graph()
g.add_nodes_from([0,1,2,3])
g.add_edges_from([(0,1),(1,2),(2,3),(3,0)])
nx.draw(g)