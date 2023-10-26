"""
TODO Can I remove movement classes if I just use degrees of freedom and reference axes?

"""

import torch
from torch import sin, cos
from typing import Sequence

from src.SPI2py.group_model.component_geometry.spherical_decomposition_methods.finite_sphere_method import \
    read_xyzr_file
from src.SPI2py.group_model.component_kinematics.distance_calculations import centroid
from src.SPI2py.group_model.component_kinematics.spatial_transformations import assemble_transformation_matrix, apply_transformation_matrix


class RigidBody:
    """
    Maximal disjoint sphere decomposition
    Assumes no spheres overlap
    """
    def __init__(self, input_file, prescale=1):

        self.positions, self.radii = read_xyzr_file(input_file)

        self.positions = self.positions * prescale
        self.radii = self.radii * prescale

        self.volume = torch.sum((4/3)*torch.pi*self.radii**3)

        # translation

    @property
    def num_spheres(self):
        return len(self.positions)

    @property
    def centroid(self):
        return centroid(self.positions, self.radii)



class Component(RigidBody):

    def __init__(self, inputs):

        self.name = inputs[0]
        inputs_dict = inputs[1]

        self.filepath = inputs_dict['filepath']
        self.dof = inputs_dict['dof']

        if 'ports' in inputs_dict.keys():
            self.ports = inputs_dict['ports']
        else:
            self.ports = None

        self.color = inputs_dict['color']

        if 'prescale' in inputs_dict.keys():
            self.prescale = inputs_dict['prescale']
        else:
            self.prescale = 1

        # Extract the positions and radii of the spheres from the xyzr file
        # self.positions, self.radii = read_xyzr_file(filepath)
        RigidBody.__init__(self, self.filepath, prescale=self.prescale)

        # Initialize the ports
        self.port_indices = {}
        if self.ports is not None:
            for port in self.ports:
                self.port_indices[port['name']] = len(self.positions - 1)
                self.positions = torch.vstack((self.positions, torch.tensor(port['origin'], dtype=torch.float64)))
                self.radii = torch.cat((self.radii, torch.tensor([port['radius']], dtype=torch.float64)))

        self.design_vector_indices = {dof_i: i for i, dof_i in enumerate(self.dof)}
        self.design_vector_size = len(self.design_vector_indices)

    def __repr__(self):
        return self.name

    def calculate_positions(self, translation, rotation):
        """
        Calculates the positions of the object's spheres.
        """

        t = assemble_transformation_matrix(translation, rotation)

        new_positions = apply_transformation_matrix(self.centroid.reshape(-1, 1),
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

        t = assemble_transformation_matrix(translation, rotation)

        new_positions = apply_transformation_matrix(self.centroid.reshape(-1, 1),
                                                    self.positions.T,
                                                    t).T

        self.positions = new_positions


    def set_positions(self, objects_dict: dict):
        """
        Update positions of object spheres given a design vector

        :param objects_dict:
        :param design_vector:
        :return:
        """

        self.positions = objects_dict[self.__repr__()]['positions']
        self.radii = objects_dict[self.__repr__()]['radii']


class Domain(RigidBody):
    def __init__(self, name, filepath, color):

        self.name = name
        self.filepath = filepath
        self.color = color

        RigidBody.__init__(self, filepath)

    def __repr__(self):
        return self.name


# name,
#                  radius,
#                  color='black',
#                  num_segments=1,
#                  num_spheres_per_segment=25

class Interconnect:

    def __init__(self, inputs, num_segments, num_spheres_per_segment):

        self.name = inputs[0]
        inputs_dict = inputs[1]
        self.color = inputs_dict['color']

        self.num_segments = num_segments
        self.num_spheres_per_segment = num_spheres_per_segment


        self.radius = inputs_dict['radius']

        self.num_nodes = self.num_segments + 1
        self.num_control_points = self.num_segments - 1
        self.num_spheres = self.num_segments * self.num_spheres_per_segment

        # The first and last indices of each segment
        self.control_point_indices = []

        # Initialize positions and radii tensors
        self.positions = torch.zeros((self.num_spheres_per_segment * self.num_segments, 3), dtype=torch.float64)
        self.radii = self.radius * torch.ones((self.num_spheres_per_segment * self.num_segments, 1), dtype=torch.float64)

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

        n = self.num_spheres_per_segment

        delta_xyz = stop_position - start_position

        delta_xyz_n = delta_xyz / n

        sphere_positions = start_position + torch.arange(0, n).reshape(-1, 1) * delta_xyz_n.reshape(1, -1)

        return sphere_positions

    def calculate_positions(self, design_vector):

        delta_node_positions = design_vector.reshape((-1, 3))

        start_positions = delta_node_positions[0:-1]
        stop_positions = delta_node_positions[1:None]

        positions = torch.empty((0, 3), dtype=torch.float64)
        for start_position, stop_position in zip(start_positions, stop_positions):
            segment_positions = self.calculate_segment_positions(start_position, stop_position)
            positions = torch.vstack((positions, segment_positions))

        positions = self.positions + positions

        object_dict = {str(self): {'type': 'interconnect', 'positions': positions, 'radii': self.radii}}

        return object_dict


    def set_positions(self, objects_dict):
        self.positions = objects_dict[str(self)]['positions']


    def set_default_positions(self, waypoints):
        object_dict = self.calculate_positions(waypoints)
        self.set_positions(object_dict)




