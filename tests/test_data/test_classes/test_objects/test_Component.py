# from SPI2Py.data.classes.objects import Component
#
# # Define component inputs
# positions = [[1, 1, 1]]
# radii = [1]
# color = 'blue'
# node = 1
# name = 'test component'
#
# # Define single port information
# port_names_single = ['supply']
# port_positions_single = [[1, 2, 1]]
# port_radii_single = [1]
#
# # Define multiple port information
# port_names_multiple = ['supply', 'return']
# port_positions_multiple = [[1, 2, 1], [1, 2, 2]]
# port_radii_multiple = [1, 0.5]
#
# def test_create_Component_no_ports():
#
#     test_component = Component(positions, radii, color, node, name)
#
# def test_create_Component_one_port():
#     test_component = Component(positions, radii, color, node, name,
#                                 port_names = port_names_single,
#                                 port_positions = port_positions_single,
#                                 port_radii = port_radii_single)
#
#     # Make sure the attribute exists
#     # assert test_component.ports is not None
#
#     # It should have one port
#     assert len(test_component.ports_names) == 1
#
#     # # The item in the list should be of type Port
#     # assert all(isinstance(port, Port) for port in test_component.ports)
#
#
# def test_create_Component_multiple_ports():
#     test_component = Component(positions, radii, color, node, name,
#                                 port_names = port_names_multiple,
#                                 port_positions = port_positions_multiple,
#                                 port_radii = port_radii_multiple)
#
#     # # Make sure the attribute exists
#     # assert test_component.ports is not None
#
#     # # Should have two prots
#     # assert len(test_component.ports) == 2
#
#     # # The items in the list should be of type Port
#     # assert all(isinstance(port, Port) for port in test_component.ports)
#
