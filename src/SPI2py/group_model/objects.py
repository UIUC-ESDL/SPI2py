"""
TODO Can I remove movement classes if I just use degrees of freedom and reference axes?

"""

import tomli

from src.SPI2py.group_model.component_geometry.spherical_decomposition_methods.finite_sphere_method import read_xyzr_file
from src.SPI2py.group_model.component_kinematics.spatial_transformations import affine_transformation
from src.SPI2py.group_model.component_kinematics.bounding_volumes import bounding_box
from src.SPI2py.group_model.component_kinematics.distance_calculations import signed_distances
from src.SPI2py.group_model.utilities import kreisselmeier_steinhauser

import torch
from typing import Sequence
import pyvista as pv


class Component:

    def __init__(self,
                 name: str,
                 color: str = None,
                 degrees_of_freedom: Sequence[str] = ('x', 'y', 'z', 'rx', 'ry', 'rz'),
                 filepath=None,
                 ports: Sequence[dict] = ()):

        # Assign the inputs
        self.name = name
        self.color = color
        self.degrees_of_freedom = degrees_of_freedom
        self.filepath = filepath
        self.ports = ports

        # Extract the positions and radii of the spheres from the xyzr file
        self.positions, self.radii = read_xyzr_file(filepath)

        # Initialize the ports
        self.port_indices = {}
        if self.ports is not None:
            for port in self.ports:
                self.port_indices[port['name']] = len(self.positions - 1)
                self.positions = torch.vstack((self.positions, torch.tensor(port['origin'], dtype=torch.float64)))
                self.radii = torch.cat((self.radii, torch.tensor([port['radius']], dtype=torch.float64)))

        # Default transformation vectors
        self.translation = torch.tensor([0, 0, 0], dtype=torch.float64)
        self.rotation = torch.tensor([0, 0, 0], dtype=torch.float64)
        self.scale = torch.tensor([1, 1, 1], dtype=torch.float64)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @property
    def reference_position(self):
        """
        Returns the reference position of the object.
        """

        x_mean = torch.mean(self.positions[:, 0])
        y_mean = torch.mean(self.positions[:, 1])
        z_mean = torch.mean(self.positions[:, 2])

        return torch.tensor([x_mean, y_mean, z_mean], dtype=torch.float64)

    @property
    def design_vector(self):

        design_vector = []

        if 'x' in self.degrees_of_freedom:
            design_vector.append(self.translation[0])
        if 'y' in self.degrees_of_freedom:
            design_vector.append(self.translation[1])
        if 'z' in self.degrees_of_freedom:
            design_vector.append(self.translation[2])

        if 'rx' in self.degrees_of_freedom:
            design_vector.append(self.rotation[0])
        if 'ry' in self.degrees_of_freedom:
            design_vector.append(self.rotation[1])
        if 'rz' in self.degrees_of_freedom:
            design_vector.append(self.rotation[2])

        if 'sx' in self.degrees_of_freedom:
            design_vector.append(self.scale[0])
        if 'sy' in self.degrees_of_freedom:
            design_vector.append(self.scale[1])
        if 'sz' in self.degrees_of_freedom:
            design_vector.append(self.scale[2])

        return torch.tensor(design_vector)

    @property
    def object_dict(self):
        return {self.__repr__(): {'positions': self.positions,
                                  'radii': self.radii,
                                  'port_indices': self.port_indices}}

    def decompose_design_vector(self, design_vector: torch.tensor) -> dict:
        """
        Takes a 1D design vector and decomposes it into a dictionary of design variables.
        """

        if len(design_vector) != len(self.degrees_of_freedom):
            raise ValueError('The specified design vector must be the same length as the degrees of freedom.')

        design_vector_dict = {}

        for i, dof in enumerate(self.degrees_of_freedom):
            design_vector_dict[dof] = design_vector[i]

        return design_vector_dict

    def assemble_transformation_vectors(self, design_vector_dict):

        translation = torch.zeros((3, 1), dtype=torch.float64)
        rotation = torch.zeros((3, 1), dtype=torch.float64)

        if 'x' in self.degrees_of_freedom:
            translation[0] = design_vector_dict['x']
        if 'y' in self.degrees_of_freedom:
            translation[1] = design_vector_dict['y']
        if 'z' in self.degrees_of_freedom:
            translation[2] = design_vector_dict['z']

        if 'rx' in self.degrees_of_freedom:
            rotation[0] = design_vector_dict['rx']
        if 'ry' in self.degrees_of_freedom:
            rotation[1] = design_vector_dict['ry']
        if 'rz' in self.degrees_of_freedom:
            rotation[2] = design_vector_dict['rz']

        return translation, rotation

    def calculate_positions(self, design_vector):
        """
        Calculates the positions of the object's spheres.
        """

        design_vector_dict = self.decompose_design_vector(design_vector)

        translation, rotation = self.assemble_transformation_vectors(design_vector_dict)

        new_positions = affine_transformation(self.reference_position.reshape(-1,1),
                                              self.positions.T,
                                              translation,
                                              rotation).T

        object_dict = {self.__repr__(): {'positions': new_positions,
                                         'radii': self.radii,
                                         'port_indices': self.port_indices}}

        return object_dict

    def set_default_positions(self, translation, rotation):
        """
        Calculates the positions of the object's spheres.
        """

        new_positions = affine_transformation(self.reference_position.reshape(-1,1),
                                              self.positions.T,
                                              translation,
                                              rotation).T

        self.positions = new_positions
        self.translation = translation
        self.rotation = rotation


    def set_positions(self, objects_dict: dict):
        """
        Update positions of object spheres given a design vector

        :param objects_dict:
        :param design_vector:
        :return:
        """

        self.positions = objects_dict[self.__repr__()]['positions']
        self.radii     = objects_dict[self.__repr__()]['radii']


class Interconnect:
    """
    Interconnects are made of one or more non-zero-length segments and connect two components.

    TODO Add a class of components for interconnect dividers (e.g., pipe tee for a three-way split)

    When an interconnect is initialized it does not contain spatial information.

    In the SPI2 class the user specifies which layout generation method to use, and that method tells
    the Interconnect InterconnectNodes what their positions are.

    For now, I will assume that interconnect nodes will start along a straight line between components A
    and B. In the near future they may be included in the layout generation method. The to-do is tracked
    in spatial_configuration.py.

    # placeholder
        component_1 = [i for i in self.components if repr(i) == self.component_1][0]
        component_2 = [i for i in self.components if repr(i) == self.component_2][0]
    """

    def __init__(self,
                 name,
                 component_1,
                 component_1_port,
                 component_2,
                 component_2_port,
                 radius,
                 color='black',
                 linear_spline_segments=1,
                 degrees_of_freedom=()):

        self.name = name
        self.component_1 = component_1
        self.component_1_port = component_1_port
        self.component_2 = component_2
        self.component_2_port = component_2_port
        self.radius = radius
        self.color = color
        self.linear_spline_segments = linear_spline_segments
        self.degrees_of_freedom = degrees_of_freedom

        self.number_of_bends = linear_spline_segments-1
        self.spheres_per_segment = 25

        self.positions = torch.empty((self.spheres_per_segment * self.linear_spline_segments - 6, 3), dtype=torch.float64)
        self.radii = torch.empty((self.spheres_per_segment * self.linear_spline_segments - 4, 1), dtype=torch.float64)

        # Default design variables
        self.waypoint_positions = torch.zeros((self.number_of_bends, 3), dtype=torch.float64)

    def __repr__(self):
        return self.name

    @property
    def design_vector(self):
        # TODO Consider not all DOF being used
        return self.waypoint_positions.flatten()

    def calculate_positions(self, design_vector, objects_dict):

        design_vector = design_vector.reshape((self.number_of_bends, 3))

        object_dict = {}

        port_index_1 = objects_dict[self.component_1]['port_indices'][self.component_1_port]
        port_index_2 = objects_dict[self.component_2]['port_indices'][self.component_2_port]

        pos_1 = objects_dict[self.component_1]['positions'][port_index_1]
        pos_2 = objects_dict[self.component_2]['positions'][port_index_2]

        node_positions = torch.vstack((pos_1, design_vector.reshape(-1, 3), pos_2))

        start_arr = node_positions[0:-1]
        stop_arr = node_positions[1:None]

        diff_arr = stop_arr - start_arr
        n = self.spheres_per_segment
        increment = diff_arr / n

        points = torch.zeros((self.spheres_per_segment * self.linear_spline_segments, 3), dtype=torch.float64)
        points[0] = start_arr[0]
        points[-1] = stop_arr[-1]

        for i in range(self.linear_spline_segments):
            points[i*n:(i+1)*n] = start_arr[i] + increment[i] * torch.arange(1, n+1).reshape(-1, 1)

        # Remove start and stop points
        points = points[2:-2]

        radii = self.radius * torch.ones(len(points))

        object_dict[str(self)] = {'type': 'interconnect', 'positions': points, 'radii': radii}

        return object_dict

    def set_positions(self, objects_dict):
        self.positions = objects_dict[str(self)]['positions']
        self.radii = objects_dict[str(self)]['radii']

    def set_default_positions(self, waypoints, objects_dict):
        object_dict = self.calculate_positions(waypoints, objects_dict)
        self.set_positions(object_dict)


class System:
    def __init__(self, input_file):

        self.input_file = input_file
        self.input = self.read_input_file()


        self.components = self.create_components()
        self.interconnects = self.create_conductors()
        self.objects = self.components + self.interconnects

        objective = self.input['problem']['objective']
        self.set_objective(objective)

    def read_input_file(self):

        with open(self.input_file, mode="rb") as fp:
            input = tomli.load(fp)

        return input

    def create_components(self):
        components_inputs = self.input['components']

        components = []
        for component_inputs in components_inputs.items():
            name = component_inputs[0]
            component_inputs = component_inputs[1]
            color = component_inputs['color']
            degrees_of_freedom = component_inputs['degrees_of_freedom']
            filepath = component_inputs['filepath']
            ports = component_inputs['ports']
            components.append(
                Component(name=name, color=color, degrees_of_freedom=degrees_of_freedom, filepath=filepath,
                          ports=ports))

        return components

    def create_conductors(self):
        conductors_inputs = self.input['conductors']

        conductors = []
        for conductor_inputs in conductors_inputs.items():
            name = conductor_inputs[0]
            conductor_inputs = conductor_inputs[1]
            component_1 = conductor_inputs['component_1']
            component_1_port = conductor_inputs['component_1_port']
            component_2 = conductor_inputs['component_2']
            component_2_port = conductor_inputs['component_2_port']
            radius = conductor_inputs['radius']
            color = conductor_inputs['color']
            linear_spline_segments = conductor_inputs['linear_spline_segments']
            degrees_of_freedom = conductor_inputs['degrees_of_freedom']
            conductors.append(Interconnect(name=name, component_1=component_1, component_1_port=component_1_port,
                                           component_2=component_2, component_2_port=component_2_port, radius=radius,
                                           color=color, linear_spline_segments=linear_spline_segments,
                                           degrees_of_freedom=degrees_of_freedom))

        return conductors

    @property
    def design_vector_size(self):

        size = 0

        for obj in self.objects:
            size += obj.design_vector_size

        return size

    def decompose_design_vector(self, design_vector):

        # Get the size of the design vector for each design vector object
        design_vector_sizes = []
        for obj in self.objects:
            design_vector_size = len(obj.design_vector)
            design_vector_sizes.append(design_vector_size)

        # Index values
        start, stop = 0, 0
        design_vectors = []

        for i, size in enumerate(design_vector_sizes):
            # Increment stop index

            stop += design_vector_sizes[i]

            design_vector_i = design_vector[start:stop]
            design_vectors.append(design_vector_i)

            # Increment start index
            start = stop

        return design_vectors


    def set_objective(self, objective: str):

        """
        Add an objective to the design study.

        :param objective: The objective function to be added.
        :param options: The options for the objective function.
        """

        # SELECT THE OBJECTIVE FUNCTION HANDLE

        if objective == 'bounding box volume':
            _objective_function = bounding_box
        else:
            raise NotImplementedError

        def objective_function(positions):
            return _objective_function(positions)

        self.objective = objective_function

    def plot(self):
        """
        Plot the model at a given state.
        """

        # Create the plot objects
        objects = []
        colors = []
        for obj in self.objects:
            for position, radius in zip(obj.positions, obj.radii):
                objects.append(pv.Sphere(radius=radius, center=position))
                colors.append(obj.color)

        # Plot the objects
        p = pv.Plotter(window_size=[1000, 1000])

        for obj, color in zip(objects, colors):
            p.add_mesh(obj, color=color)

        p.view_vector((5.0, 2, 3))
        p.add_floor('-z', lighting=True, color='tan', pad=1.0)
        p.enable_shadows()
        p.show_axes()
        p.show()