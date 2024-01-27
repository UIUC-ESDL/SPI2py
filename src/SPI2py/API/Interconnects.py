import torch
from openmdao.api import ExplicitComponent
from ..models.kinematics.linear_spline_transformations import translate_linear_spline

class Interconnect(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_segments', types=int)
        self.options.declare('num_spheres_per_segment', types=int)
        self.options.declare('radius', types=float)
        self.options.declare('color', types=str)

    def setup(self):

        # Unpack the options
        self.num_segments = self.options['num_segments']
        self.num_spheres_per_segment = self.options['num_spheres_per_segment']
        self.radius = self.options['radius']
        self.color = self.options['color']

        # Define the input shapes
        shape_start_point = (1, 3)
        shape_control_points = (self.num_segments - 1, 3)
        shape_end_point = (1, 3)

        # Define the inputs
        self.add_input('start_point', shape=shape_start_point)
        self.add_input('control_points', shape=shape_control_points)
        self.add_input('end_point', shape=shape_end_point)

        # Calculate the initial intermediate positions
        self.positions = torch.zeros((self.num_spheres_per_segment * self.num_segments, 3), dtype=torch.float64)
        self.radii = self.radius * torch.ones((self.num_spheres_per_segment * self.num_segments, 1), dtype=torch.float64)

        # Define the output shapes
        shape_positions = (self.num_spheres_per_segment * self.num_segments, 3)
        shape_radii = (self.num_spheres_per_segment * self.num_segments, 1)

        # Define the outputs
        self.add_output('positions', shape=shape_positions)
        self.add_output('radii', shape=shape_radii)

    # def setup_partials(self):
    #     pass

    def compute(self, inputs, outputs):

        # Unpack the inputs
        start_point = inputs['start_point']
        control_points = inputs['control_points']
        end_point = inputs['end_point']

        # Convert the inputs to torch tensors
        start_point = torch.tensor(start_point, dtype=torch.float64)
        control_points = torch.tensor(control_points, dtype=torch.float64)
        end_point = torch.tensor(end_point, dtype=torch.float64)

        # Calculate the positions
        translated_positions = translate_linear_spline(self.positions, start_point, control_points, end_point, self.num_spheres_per_segment)

        # Convert the outputs to numpy arrays
        translated_positions = translated_positions.detach().numpy()

        # Set the outputs
        outputs['positions'] = translated_positions
        outputs['radii'] = self.radii

    # def compute_partials(self, inputs, partials):
    #     pass

