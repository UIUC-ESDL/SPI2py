import numpy as np
import torch
from torch.autograd.functional import jacobian
from openmdao.api import ExplicitComponent

from ..models.geometry.bounding_box_volume import bounding_box_bounds, bounding_box_volume


class VerticalStackComp(ExplicitComponent):
    """
    An ExplicitComponent that vertically stacks a series of inputs with sizes (n, 3) or (n, 1).
    """

    def initialize(self):
        # Initialize with a list of sizes for each input
        self.options.declare('input_sizes', types=list, desc='List of sizes (n) for each input array')
        self.options.declare('column_size', types=int, desc='Column size, either 1 or 3')

    def setup(self):
        input_sizes = self.options['input_sizes']
        column_size = self.options['column_size']
        total_rows = sum(input_sizes)

        # Define inputs and output
        for i, size in enumerate(input_sizes):
            self.add_input(f'input_{i}', shape=(size, column_size))

        self.add_output('stacked_output', shape=(total_rows, column_size))

        # Declare partials for each input-output pair
        # start_idx = 0
        # for i, size in enumerate(input_sizes):
        #     # rows: Each input contributes size*column_size elements in the output
        #     rows = np.arange(start_idx, start_idx + size * column_size)
        #
        #     # cols: Need to ensure cols aligns with how OpenMDAO flattens the inputs for each partial
        #     # For a single input, OpenMDAO flattens it as [row1_col1, row1_col2, ..., row2_col1, row2_col2, ...]
        #     # Therefore, we need to create a pattern that matches this flattening
        #     if column_size == 1:
        #         # For (n, 1) inputs, it's a direct mapping
        #         cols = np.arange(size)
        #     elif column_size == 3:
        #         # For (n, 3) inputs, repeat each index 3 times to account for each column
        #         cols = np.repeat(np.arange(size), column_size)
        #
        #     self.declare_partials('stacked_output', f'input_{i}', rows=rows, cols=cols, val=1.0)
        #     start_idx += size * column_size

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # Stack inputs vertically
        outputs['stacked_output'] = np.vstack([inputs[f'input_{i}'] for i in range(len(self.options['input_sizes']))])

    def compute_partials(self, inputs, partials):
        # The partial derivatives have been declared in setup as constant 1.0, so no further action is needed.
        pass



class System(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_components', types=int)


    def setup(self):


        self.add_input('transformed_sphere_positions', shape_by_conn=True)
        self.add_input('transformed_sphere_radii', shape_by_conn=True)

        self.add_output('bounding_box_volume', val=1.0)
        # self.add_output('bounding_box_bounds', shape=(6,))
        # self.add_output('constraints', val=torch.zeros(1))

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')
        # self.declare_partials('bounding_box_volume', 'transformed_sphere_positions')
        # self.declare_partials('bounding_box_volume', 'transformed_sphere_radii')

        # for i in range(self.options['num_components']):
        #     self.declare_partials('bounding_box_volume', f'comp_{i}_sphere_positions')
        # self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        # Get the input variables
        sphere_positions = inputs['transformed_sphere_positions']
        sphere_radii = inputs['transformed_sphere_radii']

        # Convert the inputs to torch tensors
        sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64)
        sphere_radii = torch.tensor(sphere_radii, dtype=torch.float64)


        # Calculate the bounding box volume
        # bb_bounds = bounding_box_bounds(sphere_positions, sphere_radii)
        bb_volume = self.compute_bounding_box_volume(sphere_positions, sphere_radii)

        # Convert the outputs to numpy arrays
        # bb_bounds = bb_bounds.detach().numpy()
        bb_volume = bb_volume.detach().numpy()

        # Set the outputs
        # outputs['bounding_box_bounds'] = bb_bounds
        outputs['bounding_box_volume'] = bb_volume
        # outputs['constraints'] = constraints

    def compute_partials(self, inputs, partials):
        pass
        # # Get the input variables
        # sphere_positions = inputs['transformed_sphere_positions']
        # sphere_radii = inputs['transformed_sphere_radii']
        #
        # # Convert the inputs to torch tensors
        # sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64, requires_grad=True)
        # sphere_radii = torch.tensor(sphere_radii, dtype=torch.float64, requires_grad=True)  # TODO Remove True
        #
        # # Calculate the bounding box volume
        # jac_bb_volume = jacobian(self.compute_bounding_box_volume, (sphere_positions, sphere_radii))
        #
        # # Convert the outputs to numpy arrays
        # # jac_bb_volume = jac_bb_volume.detach().numpy()
        # jac_bbv_positions = jac_bb_volume[0].detach().numpy()
        # jac_bbv_radii = jac_bb_volume[1].detach().numpy()
        #
        # # Set the outputs
        # partials['bounding_box_volume', 'transformed_sphere_positions'] = jac_bbv_positions
        # partials['bounding_box_volume', 'transformed_sphere_radii'] = jac_bbv_radii


    @staticmethod
    def compute_bounding_box_volume(sphere_positions, sphere_radii, include_bounds=False):

        bb_bounds = bounding_box_bounds(sphere_positions, sphere_radii)
        bb_volume = bounding_box_volume(bb_bounds)

        if include_bounds:
            return bb_volume, bb_bounds
        else:
            return bb_volume


# class _System:
#     def __init__(self, components, interconnects):
#
#
#         self.components = components
#         self.interconnects = interconnects
#         self.objects = self.components + self.interconnects
#
#
#         self.component_pairs = self.get_component_pairs()
#         self.interconnect_pairs = self.get_interconnect_pairs()
#         self.component_interconnect_pairs = self.get_component_interconnect_pairs()
#
#
#         objective = self.input['problem']['objective']
#         self.set_objective(objective)
#
#         self.translations_shape = (self.num_components, 3)
#         self.rotations_shape = (self.num_components, 3)
#         self.routings_shape = (self.num_interconnects, self.num_nodes, 3)
#
#     # def set_objective(self, objective: str):
#     #
#     #     """
#     #     Add an objective to the design study.
#     #
#     #     :param objective: The objective function to be added.
#     #     :param options: The options for the objective function.
#     #     """
#     #
#     #     # SELECT THE OBJECTIVE FUNCTION HANDLE
#     #
#     #     if objective == 'bounding box volume':
#     #         _objective_function = bounding_box_volume
#     #     else:
#     #         raise NotImplementedError
#     #
#     #     def objective_function(positions):
#     #         return _objective_function(positions)
#     #
#     #     self.objective = objective_function
#     #
#     # def calculate_positions(self, translations, rotations, routings):
#     #
#     #     objects_dict = {}
#     #
#     #     for component, translation, rotation in zip(self.components, translations, rotations):
#     #         object_dict = component.calculate_positions(translation, rotation)
#     #         objects_dict = {**objects_dict, **object_dict}
#     #
#     #     for interconnect, routing in zip(self.interconnects, routings):
#     #         object_dict = interconnect.calculate_positions(routing)
#     #         objects_dict = {**objects_dict, **object_dict}
#     #
#     #     return objects_dict
#
#     def get_component_pairs(self):
#         component_component_pairs = list(combinations(self.components, 2))
#         return component_component_pairs
#
#     def get_interconnect_pairs(self):
#         interconnect_interconnect_pairs = list(combinations(self.interconnects, 2))
#         return interconnect_interconnect_pairs
#
#     def get_component_interconnect_pairs(self):
#         component_interconnect_pairs = list(product(self.components, self.interconnects))
#         return component_interconnect_pairs
#
#     def collision_component_pairs(self, positions_dict):
#         signed_distance_vals = aggregate_signed_distance(positions_dict, self.component_pairs)
#         max_signed_distance = kreisselmeier_steinhauser(signed_distance_vals)
#         return max_signed_distance
#
#     def collision_interconnect_pairs(self, positions_dict):
#         signed_distance_vals = aggregate_signed_distance(positions_dict, self.interconnect_pairs)
#         max_signed_distance = kreisselmeier_steinhauser(signed_distance_vals)
#         return max_signed_distance
#
#     def collision_component_interconnect_pairs(self, positions_dict):
#         # TODO Remove tolerance
#         signed_distance_vals = aggregate_signed_distance(positions_dict, self.component_interconnect_pairs)
#         max_signed_distance = kreisselmeier_steinhauser(signed_distance_vals) - 0.2
#         return max_signed_distance
#
#
#
#     def calculate_constraints(self, translations, rotations, routings):
#
#         positions_dict = self.calculate_positions(translations, rotations, routings)
#
#         # g_component_pairs = self.collision_component_pairs(positions_dict)
#         # g_interconnect_pairs = self.collision_interconnect_pairs(positions_dict)
#         # g_component_interconnect_pairs = self.collision_component_interconnect_pairs(positions_dict)
#         # g = torch.tensor((g_component_pairs, g_interconnect_pairs, g_component_interconnect_pairs))
#
#         # TODO Add other constraints back in
#         g = self.collision_component_pairs(positions_dict)
#
#         return g
#
#