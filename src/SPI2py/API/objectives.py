import torch
from torch.autograd.functional import jacobian
from openmdao.api import ExplicitComponent

from ..models.geometry.bounding_box_volume import bounding_box_bounds, bounding_box_volume


class BoundingBoxVolume(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_components', types=int)

    def setup(self):
        self.add_input('positions', shape_by_conn=True)
        self.add_input('radii', shape_by_conn=True)

        self.add_output('bounding_box_volume', val=1)
        self.add_output('bounding_box_bounds', shape=(6,))

    def setup_partials(self):
        self.declare_partials('bounding_box_volume', 'positions')
        self.declare_partials('bounding_box_volume', 'radii')

    def compute(self, inputs, outputs):

        # Get the input variables
        positions = inputs['positions']
        radii = inputs['radii']

        # Convert the inputs to torch tensors
        positions = torch.tensor(positions, dtype=torch.float64)
        radii = torch.tensor(radii, dtype=torch.float64)

        # Calculate the bounding box volume
        bb_volume, bb_bounds = self._bounding_box_volume(positions, radii)

        # Convert the outputs to numpy arrays
        bb_bounds = bb_bounds.detach().numpy()
        bb_volume = bb_volume.detach().numpy()

        # Set the outputs
        outputs['bounding_box_bounds'] = bb_bounds
        outputs['bounding_box_volume'] = bb_volume

    def compute_partials(self, inputs, partials):

        # Get the input variables
        sphere_positions = inputs['positions']
        sphere_radii = inputs['radii']

        # Convert the inputs to torch tensors
        sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64, requires_grad=True)
        sphere_radii = torch.tensor(sphere_radii, dtype=torch.float64, requires_grad=True)  # TODO Remove True

        # Calculate the Jacobian
        jac_bb = jacobian(self._bounding_box_volume, (sphere_positions, sphere_radii))

        # Discard the bounding box bounds Jacobian (index 1)
        jac_bbv = jac_bb[0]

        # Convert the outputs to numpy arrays
        jac_bbv_positions = jac_bbv[0].detach().numpy()
        jac_bbv_radii = jac_bbv[1].detach().numpy()

        # Set the outputs
        partials['bounding_box_volume', 'positions'] = jac_bbv_positions
        partials['bounding_box_volume', 'radii'] = jac_bbv_radii

    @staticmethod
    def _bounding_box_volume(positions, radii):

        bb_bounds = bounding_box_bounds(positions, radii)
        bb_volume = bounding_box_volume(bb_bounds)

        return bb_volume, bb_bounds






class CollisionDetection(ExplicitComponent):

    def initialize(self):
        self.options.declare('components', types=list)
        # self.options.declare('interconnects', types=list)

    def setup(self):

        self.components = self.options['components']
        self.interconnects = self.options['interconnects']

        self.component_pairs = self.get_component_pairs()
        self.interconnect_pairs = self.get_interconnect_pairs()
        self.component_interconnect_pairs = self.get_component_interconnect_pairs()

        self.add_input('translations', shape=(len(self.components), 3))
        self.add_input('rotations', shape=(len(self.components), 3))
        self.add_input('routings', shape=(len(self.interconnects), 3))

        self.add_output('g', shape=(3,))

    def setup_partials(self):
        self.declare_partials('g', 'translations')
        self.declare_partials('g', 'rotations')
        self.declare_partials('g', 'routings')

    def compute(self, inputs, outputs):

        translations = inputs['translations']
        rotations = inputs['rotations']
        routings = inputs['routings']

        positions_dict = self.calculate_positions(translations, rotations, routings)

        g_component_pairs = self.collision_component_pairs(positions_dict)
        g_interconnect_pairs = self.collision_interconnect_pairs(positions_dict)
        g_component_interconnect_pairs = self.collision_component_interconnect_pairs(positions_dict)

        g = torch.tensor((g_component_pairs, g_interconnect_pairs, g_component_interconnect_pairs))

        outputs['g'] = g

    def compute_partials(self, inputs, partials):

        translations = inputs['translations']
        rotations = inputs['rotations']
        routings = inputs['routings']

        positions_dict = self.calculate_positions(translations, rotations, routings)

        g_component_pairs = self.collision_component_pairs(positions_dict)
        g_interconnect_pairs = self.collision_interconnect_pairs(positions_dict)
        g_component_interconnect_pairs = self.collision_component_interconnect_pairs(positions_dict)

        g = torch.tensor((g_component_pairs, g_interconnect_pairs, g_component_interconnect_pairs))

        partials['g', 'translations'] = self.calculate_positions

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