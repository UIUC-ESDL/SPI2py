from src.SPI2Py.data.classes.objects import Port

# Define single port information
node = 'test_component_node_1'
name = 'supply'
color = 'blue'
location = [1, 2, 1]
number_of_connections = 1


def test_create_port():
    test_port = Port(node, name, color, location, number_of_connections)