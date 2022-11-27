"""

"""

import yaml
import json
from datetime import datetime

from .utils.optimizers.gradient_based_optimization import optimize
from .utils.visualization.visualization import generate_gif
from .utils.spherical_decomposition.prismatic_shapes import generate_rectangular_prisms
from .utils.classes.objects import Component, InterconnectSegment, Structure
from .utils.classes.organizational import SpatialConfiguration
from .utils.spatial_topologies.force_directed_layouts import generate_random_layout


class SPI2:
    """
    The SPI2 Class provides the user with a means to interact with the API...
    """

    def __init__(self):
        self.config = None
        self.inputs = None

        self.components = None
        self.interconnect_nodes = None
        self.interconnects = None
        self.structures = None

        self.layout = None

        self.result = None
        self.design_vector_log = None

        self.outputs = None

    def add_input_file(self, input_filepath):

        with open(input_filepath, 'r') as f:
            self.inputs = yaml.safe_load(f)


    def add_configuration_file(self, config_filepath):

        with open(config_filepath, 'r') as f:
            self.config = yaml.safe_load(f)


    def create_objects_from_input(self, inputs):
        """

        :param inputs:
        :return:
        """


        # Create Components
        self.components = []
        for component in inputs['components']:
            node = component
            name = inputs['components'][component]['name']
            color = inputs['components'][component]['color']
            origins = inputs['components'][component]['origins']
            dimensions = inputs['components'][component]['dimensions']

            positions, radii = generate_rectangular_prisms(origins, dimensions)
            self.components.append(Component(positions, radii, color, node, name))



        # Create Interconnects

        self.interconnects = []
        for interconnect in inputs['interconnects']:

            component_1 = self.components[inputs['interconnects'][interconnect]['component 1']]
            component_2 = self.components[inputs['interconnects'][interconnect]['component 2']]

            diameter = inputs['interconnects'][interconnect]['diameter']
            color = inputs['interconnects'][interconnect]['color']

            self.interconnects.append(InterconnectSegment(component_1, component_2, diameter, color))



        # Create Interconnect Nodes
        self.interconnect_nodes = []
        # for interconnect






        # Create Structures
        self.structures = []
        for structure in inputs['structures']:
            name = inputs['structures'][structure]['name']
            color = inputs['structures'][structure]['color']
            origins = inputs['structures'][structure]['origins']
            dimensions = inputs['structures'][structure]['dimensions']

            positions, radii = generate_rectangular_prisms(origins, dimensions)
            self.structures.append(Structure(positions, radii, color, name))




    def generate_layout(self, layout_generation_method, include_interconnect_nodes=False):
        """

        Parameters
        ----------
        layout_generation_method
        include_interconnect_nodes: If False then initialize interconnects as straight lines between
        the components after generating their layout. If True then include InterconnectNodes in the layout generation
        method (i.e., interconnects won't necessarily connect components in a straight line).

        Returns
        -------

        """

        # TODO Implement ability to include or exclude interconnect nodes from layout generation methods
        if include_interconnect_nodes:
            pass
        else:
            pass

        self.create_objects_from_input(self.inputs)

        self.layout = SpatialConfiguration(self.components, self.interconnect_nodes, self.interconnects, self.structures)

        # TODO implement different layout generation methods

        if layout_generation_method == 'manual':
            # TODO Implement functionality to manually define starting points
            pass

        elif layout_generation_method == 'force directed':
            initial_layout_design_vector = generate_random_layout(self.layout)
            self.layout.set_positions(initial_layout_design_vector)

        else:
            print('Sorry, no other layout generation methods are implemented yet')

    def optimize_spatial_configuration(self):
        self.result, self.design_vector_log = optimize(self.layout)

        # Generate GIF
        if self.config['Visualization']['Output GIF'] is True:
            generate_gif(self.layout, self.design_vector_log, 1, self.config['Outputs']['Folderpath'])

    def write_output(self, output_filepath):

        # Create a timestamp
        now = datetime.now()
        now_formatted = now.strftime("%d/%m/%Y %H:%M:%S")

        # TODO Create a prompt to ask user for comments on the results

        # Create the output dictionary
        self.outputs = {'Placeholder': 1,
                   '': 1,
                   'Date and time': now_formatted,
                   '': 1,
                   'Comments': 'Placeholder'}


        with open(output_filepath, 'w') as f:
            json.dump(self.outputs, f)
