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
    def __init__(self, num_segments, num_spheres_per_segment):

        self.num_segments = num_segments
        self.num_spheres_per_segment = num_spheres_per_segment

        self.num_nodes = num_segments + 1
        self.num_collocation_points = num_segments - 1
        self.num_spheres = num_segments * num_spheres_per_segment

        # Initialize positions and radii tensors
        self.positions = torch.empty((num_spheres_per_segment * num_segments, 3), dtype=torch.float64)
        self.radii = torch.empty((num_spheres_per_segment * num_segments, 1), dtype=torch.float64)

        # Determine the sphere indices for collocation constraints
        segments = torch.arange(num_segments)
        collocation_indices_start = segments * torch.tensor([num_spheres_per_segment]) - 1
        collocation_indices_stop = segments * torch.tensor([num_spheres_per_segment])
        self.collocation_constraint_indices = torch.vstack((collocation_indices_start, collocation_indices_stop)).T

    @property
    def design_vector_size(self):
        # TODO Make dynamic for DOF held constant...

        pass
        # return len(self.num)

    def calculate_positions(self):
        pass

    def set_positions(self):
        pass