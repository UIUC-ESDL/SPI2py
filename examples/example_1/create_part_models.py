"""

"""

from SPI2py.group_model.component_geometry import generate_rectangular_prisms


generate_rectangular_prisms('part_models/control_valve_1',
                            [[0, 0, 0]],
                            [[6, 2, 2]])

generate_rectangular_prisms('part_models/actuator_1',
                            [[0, 0, 0], [0, 0, 1.5], [0, 0, 3], [1, 1, 3.5]],
                            [[3, 3, 1.5], [3, 3, 1.5], [3, 3, 1.5], [1, 1, 5]])

generate_rectangular_prisms('part_models/component_2',
                            [[0, 0, 0]],
                            [[1, 1, 1]])

generate_rectangular_prisms('part_models/component_3',
                            [[0, 0, 0], [1, 0, 0], [1, 1, 0.5], [1, 1, 3]],
                            [[1, 1, 1], [1, 2, 1], [1, 1, 3], [2, 1, 1]])

generate_rectangular_prisms('part_models/structure_1',
                            [[0, 0, 0]],
                            [[2, 2, 0.5]])
