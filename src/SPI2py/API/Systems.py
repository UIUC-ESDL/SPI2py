import torch
from torch.autograd.functional import jacobian
from openmdao.api import ExplicitComponent

from ..models.geometry.bounding_box_volume import bounding_box_bounds, bounding_box_volume
from ..models.kinematics.distance_calculations import signed_distances_spheres_spheres

class PairwiseCollisionDetection(ExplicitComponent):

    # def initialize(self):
        # self.options.declare('components', types=list)
        # self.options.declare('interconnects', types=list)

    def setup(self):
        self.add_input('positions_a', shape_by_conn=True)
        self.add_input('radii_a', shape_by_conn=True)

        self.add_input('positions_b', shape_by_conn=True)
        self.add_input('radii_b', shape_by_conn=True)

        # FIXME Hard-coded
        self.add_output('signed_distances', shape=(10, 10))

    def setup_partials(self):
        self.declare_partials('signed_distances', 'positions_a')
        self.declare_partials('signed_distances', 'radii_a')
        self.declare_partials('signed_distances', 'positions_b')
        self.declare_partials('signed_distances', 'radii_b')

    def compute(self, inputs, outputs):

        # Extract the inputs
        positions_a = inputs['positions_a']
        radii_a = inputs['radii_a']
        positions_b = inputs['positions_b']
        radii_b = inputs['radii_b']

        # Convert the inputs to torch tensors
        positions_a = torch.tensor(positions_a, dtype=torch.float64)
        radii_a = torch.tensor(radii_a, dtype=torch.float64)
        positions_b = torch.tensor(positions_b, dtype=torch.float64)
        radii_b = torch.tensor(radii_b, dtype=torch.float64)

        signed_distances = self._compute_signed_distances(positions_a, radii_a, positions_b, radii_b)

        outputs['signed_distances'] = signed_distances

    def compute_partials(self, inputs, partials):

        # Extract the inputs
        positions_a = inputs['positions_a']
        radii_a = inputs['radii_a']
        positions_b = inputs['positions_b']
        radii_b = inputs['radii_b']

        # Convert the inputs to torch tensors
        positions_a = torch.tensor(positions_a, dtype=torch.float64, requires_grad=True)
        radii_a = torch.tensor(radii_a, dtype=torch.float64, requires_grad=True)
        positions_b = torch.tensor(positions_b, dtype=torch.float64, requires_grad=True)
        radii_b = torch.tensor(radii_b, dtype=torch.float64, requires_grad=True)

        # Calculate the partial derivatives
        jac_signed_distances = jacobian(self._compute_signed_distances, (positions_a, radii_a, positions_b, radii_b))

        # Slice the Jacobian
        jac_signed_distances_positions_a = jac_signed_distances[0].detach().numpy()
        jac_signed_distances_radii_a = jac_signed_distances[1].detach().numpy()
        jac_signed_distances_positions_b = jac_signed_distances[2].detach().numpy()
        jac_signed_distances_radii_b = jac_signed_distances[3].detach().numpy()

        # Write the outputs
        partials['signed_distances', 'positions_a'] = jac_signed_distances_positions_a
        partials['signed_distances', 'radii_a'] = jac_signed_distances_radii_a
        partials['signed_distances', 'positions_b'] = jac_signed_distances_positions_b
        partials['signed_distances', 'radii_b'] = jac_signed_distances_radii_b


    @staticmethod
    def _compute_signed_distances(positions_a, radii_a, positions_b, radii_b):
        signed_distances = signed_distances_spheres_spheres(positions_a, radii_a, positions_b, radii_b)
        return signed_distances


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

        bb_bounds = bounding_box_bounds(sphere_positions, sphere_radii)
        bb_volume = bounding_box_volume(bb_bounds)

        if include_bounds:
            return bb_volume, bb_bounds
        else:
            return bb_volume


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