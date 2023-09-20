import torch

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
        points[i * n:(i + 1) * n] = start_arr[i] + increment[i] * torch.arange(1, n + 1).reshape(-1, 1)

    # Remove start and stop points
    points = points[2:-2]

    radii = self.radius * torch.ones(len(points))

    object_dict[str(self)] = {'type': 'interconnect', 'positions': points, 'radii': radii}

    return object_dict



class LinearSpline:
    def __init__(self, num_segments, num_spheres_per_segment, radius):

        self.num_segments = num_segments
        self.num_spheres_per_segment = num_spheres_per_segment
        self.radius = radius

        self.num_nodes = num_segments + 1
        self.num_collocation_points = num_segments - 1
        self.num_spheres = num_segments * num_spheres_per_segment

        # Initialize positions and radii tensors
        self.positions = torch.zeros((num_spheres_per_segment * num_segments, 3), dtype=torch.float64)
        self.radii = radius * torch.ones((num_spheres_per_segment * num_segments, 1), dtype=torch.float64)

        # Determine the sphere indices for collocation constraints
        segments = torch.arange(1, num_segments)
        collocation_indices_start = segments * torch.tensor([num_spheres_per_segment]) - 1
        collocation_indices_stop = segments * torch.tensor([num_spheres_per_segment])
        self.collocation_constraint_indices = torch.vstack((collocation_indices_start, collocation_indices_stop)).T

    @property
    def design_vector_size(self):
        # TODO Enable different degrees of freedom

        num_dof_collocation = 3 * 2 * self.num_collocation_points
        num_dof_start = 3
        num_dof_stop = 3
        num_dof = num_dof_collocation + num_dof_start + num_dof_stop

        return num_dof

    def map_design_vector_to_node_positions(self, design_vector):
        # TODO Enable different degrees of freedom
        # TODO Enable mapping constant terms for fixed dof
        node_positions = design_vector.reshape((-1, 3))
        return node_positions

    def calculate_segment_positions(self, start_position, stop_position):

        x_start = start_position[0]
        y_start = start_position[1]
        z_start = start_position[2]

        x_stop = stop_position[0]
        y_stop = stop_position[1]
        z_stop = stop_position[2]

        x_step = (x_stop - x_start) / (self.num_spheres_per_segment-1)
        y_step = (y_stop - y_start) / (self.num_spheres_per_segment-1)
        z_step = (z_stop - z_start) / (self.num_spheres_per_segment-1)

        x_arange = torch.arange(x_start, x_stop + x_step, x_step)
        y_arange = torch.arange(y_start, y_stop + y_step, y_step)
        z_arange = torch.arange(z_start, z_stop + z_step, z_step)

        positions = torch.vstack((x_arange, y_arange, z_arange)).T

        return positions

    def calculate_positions(self, design_vector):
        # TODO Vectorize calculation

        node_positions = self.map_design_vector_to_node_positions(design_vector)

        start_positions = node_positions.view(-1, 6)[:, [0, 1, 2]]
        stop_positions = node_positions.view(-1, 6)[:, [3, 4, 5]]

        positions = torch.empty((0, 3), dtype=torch.float64)
        for start_position, stop_position in zip(start_positions, stop_positions):
            segment_positions = self.calculate_segment_positions(start_position, stop_position)
            positions = torch.vstack((positions, segment_positions))

        return positions

    def set_positions(self):
        pass

ls = LinearSpline(3, 5, 0.25)

print(len(ls.calculate_positions(torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3], dtype=torch.float64))))

print('Done')