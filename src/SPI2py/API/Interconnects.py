import torch
from openmdao.api import ExplicitComponent
from SPI2py.models.kinematics.linear_spline_transformations import translate_linear_spline

class Interconnect(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_segments',type=int)
        self.options.declare('num_spheres_per_segment', type=int)
        self.options.declare('radius', type=float)
        self.options.declare('color', type=str)

    def setup(self):

        # Unpack the options
        self.num_segments = self.options['num_segments']
        self.num_spheres_per_segment = self.options['num_spheres_per_segment']
        self.radius = self.options['radius']
        self.color = self.options['color']

        # Define the inputs
        self.add_input('start_point', shape_by_conn=True)
        self.add_input('control_points', shape_by_conn=True)
        self.add_input('end_point', shape_by_conn=True)

        # Calculate the initial intermediate positions
        self.positions = torch.zeros((self.num_spheres_per_segment * self.num_segments, 3), dtype=torch.float64)
        self.radii = self.radius * torch.ones((self.num_spheres_per_segment * self.num_segments, 1), dtype=torch.float64)

        # Define the outputs
        self.add_output('positions', shape_by_conn=True)
        self.add_output('radii', shape_by_conn=True)

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
        positions = compute_linear_spline_positions(positions, start_point, control_points, end_point, self.num_spheres_per_segment)


    # def compute_partials(self, inputs, partials):
    #     pass

