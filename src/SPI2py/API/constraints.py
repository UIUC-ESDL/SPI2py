import torch
from torch.autograd.functional import jacobian
from openmdao.api import ExplicitComponent, Group

from SPI2py.models.kinematics.distance_calculations import signed_distances_spheres_spheres

class CombinatorialCollisionDetection(Group):
    pass

class PairwiseCollisionDetection(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_spheres', types=int, desc='Number of spheres for input a')
        self.options.declare('m_spheres', types=int, desc='Number of spheres for input b')


    def setup(self):
        n = self.options['n_spheres']
        m = self.options['m_spheres']

        self.add_input('positions_a', shape=(n, 3))
        self.add_input('radii_a', shape=(n, 1))

        self.add_input('positions_b', shape=(m, 3))
        self.add_input('radii_b', shape=(m, 1))

        self.add_output('signed_distances', shape=(n, m))

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

        # Calculate the signed distances
        signed_distances = self._compute_signed_distances(positions_a, radii_a, positions_b, radii_b)

        # Convert the outputs to numpy arrays
        signed_distances = signed_distances.detach().numpy()

        # Write the outputs
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