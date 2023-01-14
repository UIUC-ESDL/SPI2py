from SPI2Py.data.classes.objects import Component
from src.SPI2Py.data.classes.common import Port

# Define component inputs
positions = [[1, 1, 1]]
radii = [1]
color = 'blue'
node = 1
name = 'test component'

# Define single port information
port_nodes_single = [0]
port_names_single = ['supply']
port_colors_single = ['blue']
port_locations_single = [[1, 2, 1]]
port_number_of_connections_single = [1]

# Define multiple port information
port_nodes_multiple = [0, 1]
port_names_multiple = ['supply', 'return']
port_colors_multiple = ['red', 'blue']
port_locations_multiple = [[1, 2, 1], [1, 2, 2]]
port_number_of_connections_multiple = [1, 1]

def test_create_Component_no_ports():

    test_component = Component(positions, radii, color, node, name)

def test_create_Component_one_port():
    test_component = Component(positions, radii, color, node, name,
                                port_nodes = port_nodes_single,
                                port_names = port_names_single, 
                                port_colors = port_colors_single,
                                port_locations = port_locations_single, 
                                port_num_connections = port_number_of_connections_single)

    # Make sure the attribute exists
    assert test_component.ports is not None

    # It should have one port
    assert len(test_component.ports) == 1

    # The item in the list should be of type Port
    assert all(isinstance(port, Port) for port in test_component.ports)


def test_create_Component_multiple_ports():
    test_component = Component(positions, radii, color, node, name,
                                port_nodes = port_nodes_multiple,
                                port_names = port_names_multiple, 
                                port_colors = port_colors_multiple,
                                port_locations = port_locations_multiple, 
                                port_num_connections = port_number_of_connections_multiple)

    # Make sure the attribute exists
    assert test_component.ports is not None

    # Should have two prots
    assert len(test_component.ports) == 2

    # The items in the list should be of type Port
    assert all(isinstance(port, Port) for port in test_component.ports)

