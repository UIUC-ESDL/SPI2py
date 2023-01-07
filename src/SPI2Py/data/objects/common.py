"""

"""


class Port:

    def __init__(self, node, name, color, location, num_connections):
        self.node = node
        self.name = name
        self.color = color
        self.location = location
        self.num_connections = num_connections

# Define single port information
port_nodes_single = [0]
port_names_single = ['supply']
port_colors_single = ['blue']
port_locations_single = [[1, 2, 1]]
port_number_of_connections_single = [1]


port0 = Port(port_nodes_single[0], port_names_single[0], port_colors_single[0], port_locations_single[0], port_number_of_connections_single[0])