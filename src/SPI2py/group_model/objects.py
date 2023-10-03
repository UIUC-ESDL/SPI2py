"""
TODO Can I remove movement classes if I just use degrees of freedom and reference axes?

"""

import tomli
from itertools import combinations, product
import torch
from torch import sin, cos
from typing import Sequence
import pyvista as pv

from src.SPI2py.group_model.component_geometry.spherical_decomposition_methods.finite_sphere_method import \
    read_xyzr_file
from src.SPI2py.group_model.component_kinematics.spatial_transformations import apply_homogenous_transformation
from src.SPI2py.group_model.component_kinematics.bounding_volumes import bounding_box
from src.SPI2py.group_model.component_kinematics.distance_calculations import signed_distances
from src.SPI2py.group_model.utilities import kreisselmeier_steinhauser

from src.SPI2py.group_model.component_kinematics.bounding_volumes import bounding_box
from src.SPI2py.group_model.utilities import kreisselmeier_steinhauser

class Body:
    pass

class RigidBody:
    pass

class DeformableBody:
    pass

class LinearSpline:
    pass

class Component(RigidBody):

    # TODO Preprocess: Calculate centroids & principal axes, transform components to origin, etc.

    def __init__(self,
                 name: str,
                 color: str = None,
                 dof: Sequence[str] = ('x', 'y', 'z', 'rx', 'ry', 'rz'),
                 filepath=None,
                 ports: Sequence[dict] = ()):

        # Assign the inputs
        self.name = name
        self.color = color
        self.dof = dof
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

        self.num_spheres = len(self.positions)

        self.design_vector_indices = {dof_i: i for i, dof_i in enumerate(self.dof)}
        self.design_vector_size = len(self.design_vector_indices)

    def __repr__(self):
        return self.name

    @property
    def centroid(self):
        """
        Returns the centroid of the object. Assumes no overlapping spheres.
        """

        v_i = ((4 / 3) * torch.pi * self.radii ** 3).view(-1, 1)
        v_total = torch.sum(v_i)

        centroid = torch.sum(self.positions * v_i, 0) / v_total

        return centroid

    def principal_axes(self):
        pass

    def align_component(self):
        pass

    def assemble_transformation_vectors(self, vector, check_dof=True):

        translation = torch.zeros((3, 1), dtype=torch.float64)
        rotation = torch.zeros((3, 1), dtype=torch.float64)

        if not check_dof:
            translation[0] = vector[0]
            translation[1] = vector[1]
            translation[2] = vector[2]
            rotation[0] = vector[3]
            rotation[1] = vector[4]
            rotation[2] = vector[5]
        else:
            if 'x' in self.dof:
                translation[0] = vector[self.design_vector_indices['x']]
            if 'y' in self.dof:
                translation[1] = vector[self.design_vector_indices['y']]
            if 'z' in self.dof:
                translation[2] = vector[self.design_vector_indices['z']]
            if 'rx' in self.dof:
                rotation[0] = vector[self.design_vector_indices['rx']]
            if 'ry' in self.dof:
                rotation[1] = vector[self.design_vector_indices['ry']]
            if 'rz' in self.dof:
                rotation[2] = vector[self.design_vector_indices['rz']]

        return translation, rotation

    def assemble_homogenous_transformation_matrix(self, vector, check_dof=True):

        translation, rotation = self.assemble_transformation_vectors(vector, check_dof)

        # Initialize the transformation matrix
        t = torch.eye(4, dtype=torch.float64)

        # Insert the translation vector
        t[:3, [3]] = translation

        # Unpack the rotation angles (Euler)
        a = rotation[0]  # alpha
        b = rotation[1]  # beta
        g = rotation[2]  # gamma

        # Calculate rotation matrix (R = R_z(gamma) @ R_y(beta) @ R_x(alpha))
        r = torch.cat(
            (cos(b) * cos(g), sin(a) * sin(b) * cos(g) - cos(a) * sin(g), cos(a) * sin(b) * cos(g) + sin(a) * sin(g),
             cos(b) * sin(g), sin(a) * sin(b) * sin(g) + cos(a) * cos(g), cos(a) * sin(b) * sin(g) - sin(a) * cos(g),
             -sin(b),         sin(a) * cos(b),                            cos(a) * cos(b))).view(3, 3)

        # Insert the rotation matrix
        t[:3, :3] = r

        return t

    def calculate_positions(self, design_vector):
        """
        Calculates the positions of the object's spheres.
        """

        t = self.assemble_homogenous_transformation_matrix(design_vector)

        new_positions = apply_homogenous_transformation(self.centroid.reshape(-1, 1),
                                                  self.positions.T,
                                                  t).T

        object_dict = {self.__repr__(): {'positions': new_positions,
                                         'radii': self.radii,
                                         'port_indices': self.port_indices}}

        return object_dict

    def set_default_positions(self, translation, rotation):
        """
        Calculates the positions of the object's spheres.
        """

        vector = torch.cat((translation, rotation))

        t = self.assemble_homogenous_transformation_matrix(vector, check_dof=False)

        new_positions = apply_homogenous_transformation(self.centroid.reshape(-1, 1),
                                                  self.positions.T,
                                                  t).T

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
        self.radii = objects_dict[self.__repr__()]['radii']


class Interconnect:

    def __init__(self,
                 name,
                 radius,
                 color='black',
                 num_segments=1,
                 num_spheres_per_segment=25):

        self.name = name
        self.color = color

        self.num_segments = num_segments
        self.num_spheres_per_segment = num_spheres_per_segment
        self.radius = radius

        self.num_nodes = num_segments + 1
        self.num_control_points = num_segments - 1
        self.num_spheres = num_segments * num_spheres_per_segment

        # The first and last indices of each segment
        self.control_point_indices = []

        # Initialize positions and radii tensors
        self.positions = torch.zeros((num_spheres_per_segment * num_segments, 3), dtype=torch.float64)
        self.radii = radius * torch.ones((num_spheres_per_segment * num_segments, 1), dtype=torch.float64)

        # Determine the sphere indices for collocation constraints
        # segments = torch.arange(1, num_segments)
        # collocation_indices_start = segments * torch.tensor([num_spheres_per_segment]) - 1
        # collocation_indices_stop = segments * torch.tensor([num_spheres_per_segment])
        # self.collocation_constraint_indices = torch.vstack((collocation_indices_start, collocation_indices_stop)).T

    def __repr__(self):
        return self.name

    @property
    def design_vector_size(self):
        # TODO Enable different degrees of freedom

        num_dof = 3 * self.num_nodes

        return num_dof

    def calculate_segment_positions(self, start_position, stop_position):

        x_start = start_position[0]
        y_start = start_position[1]
        z_start = start_position[2]

        x_stop = stop_position[0]
        y_stop = stop_position[1]
        z_stop = stop_position[2]

        x = torch.linspace(x_start, x_stop, self.num_spheres_per_segment)
        y = torch.linspace(y_start, y_stop, self.num_spheres_per_segment)
        z = torch.linspace(z_start, z_stop, self.num_spheres_per_segment)

        positions = torch.vstack((x, y, z)).T

        return positions

    def calculate_positions(self, design_vector):

        delta_node_positions = design_vector.reshape((-1, 3))

        start_positions = delta_node_positions[0:-1]
        stop_positions = delta_node_positions[1:None]

        positions = torch.empty((0, 3), dtype=torch.float64)
        for start_position, stop_position in zip(start_positions, stop_positions):
            segment_positions = self.calculate_segment_positions(start_position, stop_position)
            positions = torch.vstack((positions, segment_positions))

        positions = self.positions + positions

        object_dict= {str(self): {'type': 'interconnect', 'positions': positions, 'radii': self.radii}}

        return object_dict


    def set_positions(self, objects_dict):
        self.positions = objects_dict[str(self)]['positions']


    def set_default_positions(self, waypoints):
        object_dict = self.calculate_positions(waypoints)
        self.set_positions(object_dict)


class Domain:
    pass

class System:
    def __init__(self, input_file):

        self.input_file = input_file
        self.input = self.read_input_file()

        self.components = self.create_components()
        self.interconnects, self.collocation_constraint_indices = self.create_conductors()
        self.objects = self.components + self.interconnects

        self.component_component_pairs = list(combinations(self.components, 2))
        self.component_interconnect_pairs = list(product(self.components, self.interconnects))
        self.interconnect_interconnect_pairs = list(combinations(self.interconnects, 2))

        objective = self.input['problem']['objective']
        self.set_objective(objective)

        # TODO Convert local indexing to global indexing...

        # TODO Component-Interconnect collocation constraints

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
                Component(name=name, color=color, dof=degrees_of_freedom, filepath=filepath,
                          ports=ports))

        return components

    def create_conductors(self):

        # TODO Define collocation constraints

        conductors_inputs = self.input['conductors']

        conductors = []
        collocation_constraint_indices = []
        for conductor_inputs in conductors_inputs.items():
            name = conductor_inputs[0]
            conductor_inputs = conductor_inputs[1]

            component_1 = conductor_inputs['component_1']
            component_1_port = conductor_inputs['component_1_port']
            component_2 = conductor_inputs['component_2']
            component_2_port = conductor_inputs['component_2_port']

            component_1_index = self.components.index([i for i in self.components if repr(i) == component_1][0])
            component_2_index = self.components.index([i for i in self.components if repr(i) == component_2][0])
            component_1_port_index = self.components[component_1_index].port_indices[component_1_port]
            component_2_port_index = self.components[component_2_index].port_indices[component_2_port]
            collocation_constraint_1 = [component_1_port_index, 0]
            collocation_constraint_2 = [component_2_port_index, -1]
            collocation_constraint_indices.append(collocation_constraint_1)
            collocation_constraint_indices.append(collocation_constraint_2)

            radius = conductor_inputs['radius']
            color = conductor_inputs['color']
            num_segments = conductor_inputs['num_segments']


            conductors.append(Interconnect(name=name,
                                           radius=radius,
                                           color=color,
                                           num_segments=num_segments))

        return conductors, collocation_constraint_indices

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
            design_vector_size = obj.design_vector_size
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

    def assemble_transformation_matrices(self):
        pass

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

    def calculate_positions(self, design_vector, limit_spheres=False):

        design_vectors = self.decompose_design_vector(design_vector)
        design_vectors_components = design_vectors[:len(self.components)]
        design_vectors_interconnects = design_vectors[len(self.components):]

        objects_dict = {}

        for component, design_vector in zip(self.components, design_vectors_components):
            object_dict = component.calculate_positions(design_vector=design_vector)
            objects_dict = {**objects_dict, **object_dict}

        for interconnect, design_vector in zip(self.interconnects, design_vectors_interconnects):
            object_dict = interconnect.calculate_positions(design_vector=design_vector)
            objects_dict = {**objects_dict, **object_dict}

        return objects_dict

    def set_default_positions(self, default_positions_dict):

        # Map components first
        for component in self.components:
            translation, rotation = list(default_positions_dict[component.__repr__()].values())
            translation = torch.tensor(translation, dtype=torch.float64).reshape(3, 1)
            rotation = torch.tensor(rotation, dtype=torch.float64).reshape(3, 1)
            component.set_default_positions(translation, rotation)

        # Now interconnects
        for interconnect in self.interconnects:
            waypoints = list(default_positions_dict[interconnect.__repr__()].values())
            waypoints = torch.tensor(waypoints, dtype=torch.float64)
            interconnect.set_default_positions(waypoints)

    def set_positions(self, objects_dict):
        """set_positions Sets the positions of the objects in the layout.

        Takes a flattened design vector and sets the positions of the objects in the layout.

        :param positions_dict: A dictionary of positions for each object in the layout.
        :type positions_dict: dict
        """

        for obj in self.objects:
            obj.set_positions(objects_dict)

    def collision_detection(self, x, object_pair):

        # Calculate the positions of all spheres in layout given design vector x
        positions_dict = self.calculate_positions(x, limit_spheres=True)

        signed_distance_vals = signed_distances(positions_dict, object_pair)
        max_signed_distance = kreisselmeier_steinhauser(signed_distance_vals)

        return max_signed_distance

    def calculate_objective(self, x):

        positions_dict = self.calculate_positions(x)

        positions_array = torch.vstack([positions_dict[key]['positions'] for key in positions_dict.keys()])

        objective = self.objective(positions_array)

        return objective

    def calculate_constraints(self, x):

        g_components_components = self.collision_detection(x, self.component_component_pairs).reshape(1, 1)
        g_components_interconnects = self.collision_detection(x, self.component_interconnect_pairs).reshape(1, 1)
        g_interconnects_interconnects = self.collision_detection(x, self.interconnect_interconnect_pairs).reshape(1, 1)

        g = torch.cat((g_components_components, g_components_interconnects, g_interconnects_interconnects))

        return g

    def plot(self):
        """
        Plot the model at a given state.
        """

        # Create the plot objects
        objects = []
        colors = []
        for obj in self.objects:

            spheres = []
            for position, radius in zip(obj.positions, obj.radii):
                spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))

            merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
            # merged_clipped = merged.clip(normal='z')
            merged_slice = merged.slice(normal=[0, 0, 1])

            objects.append(merged_slice)
            colors.append(obj.color)

        # Plot the objects
        p = pv.Plotter(window_size=[1000, 1000])

        for obj, color in zip(objects, colors):
            p.add_mesh(obj, color=color)

        # p.view_isometric()
        p.view_xy()
        # p.show_axes()
        # p.show_bounds(color='black')
        p.background_color = 'white'
        p.show()

