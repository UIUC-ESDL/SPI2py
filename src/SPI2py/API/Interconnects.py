import torch


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
