import logging
import os

import yaml

from .geometry.spherical_decomposition import generate_rectangular_prisms
from src.SPI2py.objects import Component, Port, Interconnect, Structure
from src.SPI2py.systems import System

class Data:
    """
    Data class for interacting with the SPI2py API.
    """

    def __init__(self,
                 directory:  str):

        self.directory  = directory

        # Initialize default configuration
        self._entry_point_directory = os.path.dirname(__file__) + '/'
        self.config                 = self.read_config_file('config.yaml')

        self.logger_name            = self.directory + "logger.log"

        # Initialize the logger
        self.initialize_logger()

        self.system = self.create_system()

        # Create objects from the input file


    def read_config_file(self, config_filepath):
        config_filepath = self._entry_point_directory + config_filepath
        with open(config_filepath, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def read_input_file(self, input_filename):
        input_filepath = self.directory + input_filename
        with open(input_filepath, 'r') as f:
            inputs = yaml.safe_load(f)
        return inputs

    def add_component(self,
                      name: str,
                      color: str,
                      movement_class: str,
                      shapes: list):


        """
        Add a component to the system.

        :param name:
        :param color:
        :param movement_class:
        :param shapes:
        :return:
        """

        origins = []
        dimensions = []
        for shape in shapes:
            origins.append(shape['origin'])
            dimensions.append(shape['dimensions'])

        positions, radii = generate_rectangular_prisms(origins, dimensions)

        component = Component(name, positions, radii, color, movement_class=movement_class)

        # Update the system
        self.system.components.append(component)

    def add_port(self, component_name, port_name, color, radius,reference_point_offset, movement_class):
        """
        Add a port to the system.

        :param
        """
        port = Port(component_name, port_name, color, radius,reference_point_offset, movement_class=movement_class)
        self.system.ports.append(port)

    def add_interconnect(self, name, component_1, component_1_port, component_2, component_2_port, radius, color, number_of_bends):
        """
        Add an interconnect to the system.

        """
        interconnect = Interconnect(name, component_1, component_1_port, component_2, component_2_port, radius, color, number_of_bends)

        self.system.interconnects.append(interconnect)
        self.system.interconnect_segments.extend(interconnect.segments)
        self.system.interconnect_nodes.extend(interconnect.interconnect_nodes)

    def add_structure(self, name, color, movement_class, shapes):
        """
        Add a structure to the system.

        """

        origins = []
        dimensions = []
        for shape in shapes:
            origins.append(shape['origin'])
            dimensions.append(shape['dimensions'])

        positions, radii = generate_rectangular_prisms(origins, dimensions)

        structure = Structure(name, positions, radii, color, movement_class)

        self.system.structures.append(structure)

    def initialize_logger(self):
        logging.basicConfig(filename=self.logger_name, encoding='utf-8', level=logging.INFO, filemode='w')

    def print_log(self):
        with open(self.logger_name) as f:
            print(f.read())

    def create_system(self):
        """
        Create the objects from the input files.

        :return:
        """
        # TODO Replace all with interconnects
        system = System(self.config)

        return system