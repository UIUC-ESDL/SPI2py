import numpy as np
import torch
from torch.autograd.functional import jacobian
from openmdao.api import ExplicitComponent

from ..models.geometry.bounding_box_volume import bounding_box_bounds, bounding_box_volume, smooth_bounding_box_bounds, smooth_revised_bounding_box_bounds



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

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        # Get the options
        input_sizes = self.options['input_sizes']

        # Get the input arrays
        input_arrays = ()
        for i in range(len(input_sizes)):
            input_arrays = input_arrays + (inputs[f'input_{i}'],)

        # Convert the input arrays to torch tensors
        input_tensors = ()
        for input_array in input_arrays:
            input_tensors = input_tensors + (torch.tensor(input_array, dtype=torch.float64),)

        # Stack inputs vertically
        stacked_output = self._compute_vstack(*input_tensors)

        # Convert the stacked output to a numpy array
        stacked_output = stacked_output.detach().numpy()

        # Set the output
        outputs['stacked_output'] = stacked_output

    def compute_partials(self, inputs, partials):

        # Get the options
        input_sizes = self.options['input_sizes']

        # Get the input arrays
        input_arrays = ()
        for i in range(len(input_sizes)):
            input_arrays = input_arrays + (inputs[f'input_{i}'],)

        # Convert the input arrays to torch tensors
        input_tensors = ()
        for input_array in input_arrays:
            input_tensors = input_tensors + (torch.tensor(input_array, dtype=torch.float64, requires_grad=True),)

        # Calculate the partial derivatives
        jac_stacked_output = jacobian(self._compute_vstack, input_tensors)

        # Convert the partial derivatives to numpy arrays
        jac_stacked_output_np = []
        for jac in jac_stacked_output:
            jac_stacked_output_np.append(jac.detach().numpy())

        # Set the partial derivatives
        for i in range(len(input_sizes)):
            partials['stacked_output', f'input_{i}'] = jac_stacked_output_np[i]


    @staticmethod
    def _compute_vstack(*args):
        return torch.vstack(args)


class System(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_components', types=int)

    def setup(self):
        self.add_input('transformed_sphere_positions', shape_by_conn=True)
        self.add_input('transformed_sphere_radii', shape_by_conn=True)

        self.add_output('bounding_box_volume', val=1)
        self.add_output('bounding_box_bounds', shape=(6,))

    def setup_partials(self):
        # self.declare_partials('*', '*', method='fd')
        self.declare_partials('bounding_box_volume', 'transformed_sphere_positions')
        self.declare_partials('bounding_box_volume', 'transformed_sphere_radii')

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
        bb_bounds = bounding_box_bounds(sphere_positions, sphere_radii)
        bb_volume = self.compute_bounding_box_volume(sphere_positions, sphere_radii)

        # Convert the outputs to numpy arrays
        bb_bounds = bb_bounds.detach().numpy()
        bb_volume = bb_volume.detach().numpy()

        # Set the outputs
        outputs['bounding_box_bounds'] = bb_bounds
        outputs['bounding_box_volume'] = bb_volume

    def compute_partials(self, inputs, partials):
        # pass
        # Get the input variables
        sphere_positions = inputs['transformed_sphere_positions']
        sphere_radii = inputs['transformed_sphere_radii']

        # Convert the inputs to torch tensors
        sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64, requires_grad=True)
        sphere_radii = torch.tensor(sphere_radii, dtype=torch.float64, requires_grad=True)  # TODO Remove True

        # Calculate the bounding box volume
        jac_bb_volume = jacobian(self.compute_bounding_box_volume, (sphere_positions, sphere_radii))

        # Convert the outputs to numpy arrays
        # jac_bb_volume = jac_bb_volume.detach().numpy()
        jac_bbv_positions = jac_bb_volume[0].detach().numpy()
        jac_bbv_radii = jac_bb_volume[1].detach().numpy()

        # Set the outputs
        partials['bounding_box_volume', 'transformed_sphere_positions'] = jac_bbv_positions
        partials['bounding_box_volume', 'transformed_sphere_radii'] = jac_bbv_radii

    @staticmethod
    def compute_bounding_box_volume(sphere_positions, sphere_radii, include_bounds=False):

        bb_bounds_s = smooth_bounding_box_bounds(sphere_positions, sphere_radii)
        bb_bounds = smooth_revised_bounding_box_bounds(sphere_positions, sphere_radii)
        bb_bounds_n = bounding_box_bounds(sphere_positions, sphere_radii)
        bb_volume = bounding_box_volume(bb_bounds)

        if include_bounds:
            return bb_volume, bb_bounds
        else:
            return bb_volume

    # @staticmethod
    # def compute_bounding_box_volume(sphere_positions_0, sphere_radii_0, sphere_positions_1, sphere_radii_1):
    #
    #     # bb_bounds = smooth_bounding_box_bounds(sphere_positions, sphere_radii)
    #     bb_bounds = bounding_box_bounds(sphere_positions_0, sphere_radii_0, sphere_positions_1, sphere_radii_1)
    #     bb_volume = bounding_box_volume(bb_bounds)
    #
    #
    #     return bb_volume, bb_bounds



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