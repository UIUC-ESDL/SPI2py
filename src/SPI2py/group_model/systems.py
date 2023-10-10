from itertools import combinations, product

import pyvista as pv
import tomli
import torch

from src.SPI2py import Component, Interconnect
from src.SPI2py.group_model.component_kinematics.bounding_volumes import bounding_box_volume
from src.SPI2py.group_model.component_kinematics.distance_calculations import signed_distances
from src.SPI2py.group_model.objects import Domain
from src.SPI2py.group_model.utilities import kreisselmeier_steinhauser



class System:
    def __init__(self, input_file):

        self.input_file = input_file
        self.input = self.read_input_file()

        # Create the objects
        self.sphere_indices_component = []
        self.sphere_indices_interconnect = []
        self.components, self.num_components = self.create_components()
        self.interconnects, self.num_interconnects, self.num_nodes, self.collocation_constraint_indices = self.create_conductors()
        self.objects = self.components + self.interconnects

        # TODO Figure out how to handle objects w/ domains...
        # self.domains = self.create_domains()

        self.collision_detection_pairs = self.get_collision_detection_pairs()

        objective = self.input['problem']['objective']
        self.set_objective(objective)

        self.translations_shape = (self.num_components, 3)
        self.rotations_shape = (self.num_components, 3)
        self.routings_shape = (self.num_interconnects, self.num_nodes, 3)

    def read_input_file(self):

        with open(self.input_file, mode="rb") as fp:
            inputs = tomli.load(fp)

        return inputs

    def create_components(self):
        components_inputs = self.input['components']
        components = []
        for component_inputs in components_inputs.items():
            component = Component(component_inputs)
            components.append(component)

        num_components = len(components)
        return components, num_components

    def create_conductors(self):

        conductors_inputs = self.input['conductors']
        conductors = []
        collocation_constraint_indices = []
        num_segments = conductors_inputs['num_segments']
        num_spheres_per_segment = conductors_inputs['num_spheres_per_segment']

        for conductor_inputs in list(conductors_inputs.items())[2:None]:

            component_1 = conductor_inputs[1]['component_1']
            component_1_port = conductor_inputs[1]['component_1_port']
            component_2 = conductor_inputs[1]['component_2']
            component_2_port = conductor_inputs[1]['component_2_port']

            component_1_index = self.components.index([i for i in self.components if repr(i) == component_1][0])
            component_2_index = self.components.index([i for i in self.components if repr(i) == component_2][0])
            component_1_port_index = self.components[component_1_index].port_indices[component_1_port]
            component_2_port_index = self.components[component_2_index].port_indices[component_2_port]
            collocation_constraint_1 = [component_1_port_index, 0]
            collocation_constraint_2 = [component_2_port_index, -1]
            collocation_constraint_indices.append(collocation_constraint_1)
            collocation_constraint_indices.append(collocation_constraint_2)

            conductor = Interconnect(conductor_inputs, num_segments, num_spheres_per_segment)
            conductors.append(conductor)

        num_interconnects = len(conductors)
        num_nodes = num_segments + 1

        return conductors, num_interconnects, num_nodes, collocation_constraint_indices

    def create_domains(self):

        domains_inputs = self.input['domain']

        domains = []
        for domain_inputs in domains_inputs.items():
            name = domain_inputs[0]
            domain_inputs = domain_inputs[1]
            color = domain_inputs['color']
            filepath = domain_inputs['filepath']

            domains.append(Domain(name=name, filepath=filepath, color=color))

        return domains


    def assemble_transformation_matrices(self):
        pass

    def set_objective(self, objective: str):

        """
        Add an objective to the design study.

        :param objective: The objective function to be added.
        :param options: The options for the objective function.
        """

        # SELECT THE OBJECTIVE FUNCTION HANDLE

        if objective == 'bounding box volume':
            _objective_function = bounding_box_volume
        else:
            raise NotImplementedError

        def objective_function(positions):
            return _objective_function(positions)

        self.objective = objective_function

    def calculate_positions(self, translations, rotations, routings):

        objects_dict = {}

        for component, translation, rotation in zip(self.components, translations, rotations):
            object_dict = component.calculate_positions(translation, rotation)
            objects_dict = {**objects_dict, **object_dict}

        for interconnect, routing in zip(self.interconnects, routings):
            object_dict = interconnect.calculate_positions(routing)
            objects_dict = {**objects_dict, **object_dict}

        return objects_dict


    def get_collision_detection_pairs(self):

        collision_detection_pairs = []

        if 'components' in self.input.keys():
            component_component_pairs = list(combinations(self.components, 2))
            collision_detection_pairs.append(component_component_pairs)

        if 'conductors' in self.input.keys():
            interconnect_interconnect_pairs = list(combinations(self.interconnects, 2))
            collision_detection_pairs.append(interconnect_interconnect_pairs)

        if 'components' in self.input.keys() and 'conductors' in self.input.keys():
            component_interconnect_pairs = list(product(self.components, self.interconnects))
            collision_detection_pairs.append(component_interconnect_pairs)

        return collision_detection_pairs


    def collision_detection(self, positions_dict, object_pair):

        signed_distance_vals = signed_distances(positions_dict, object_pair)
        max_signed_distance = kreisselmeier_steinhauser(signed_distance_vals)

        return max_signed_distance

    def calculate_objective(self, translations, rotations, routings):

        positions_dict = self.calculate_positions(translations, rotations, routings)

        positions_array = torch.vstack([positions_dict[key]['positions'] for key in positions_dict.keys()])

        objective = self.objective(positions_array)

        return objective

    def calculate_constraints(self, translations, rotations, routings):

        positions_dict = self.calculate_positions(translations, rotations, routings)

        g = []
        for cd_pair in self.collision_detection_pairs:
            cd_constraint = self.collision_detection(positions_dict, cd_pair)
            g.append(cd_constraint)

        g = torch.tensor(g)

        return g

    def plot(self, translations, rotations, routings):
        """
        Plot the model at a given state.
        """

        positions_dict = self.calculate_positions(translations, rotations, routings)

        # Create the plot objects
        objects = []
        colors = []
        for obj in self.objects:

            positions = positions_dict[str(obj)]['positions']
            radii = positions_dict[str(obj)]['radii']

            spheres = []
            for position, radius in zip(positions, radii):
                spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))

            merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
            # merged_clipped = merged.clip(normal='z')
            # merged_slice = merged.slice(normal=[0, 0, 1])

            objects.append(merged)
            colors.append(obj.color)

        # for domain in self.domains:
        #     spheres = []
        #     for position, radius in zip(domain.positions, domain.radii):
        #         spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))
        #     merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
        #     # merged_clipped = merged.clip(normal='y')
        #     # merged_slice = merged.slice(normal=[0, 0, 1])
        #
        #     objects.append(merged)
        #     colors.append(domain.color)


        # Plot the objects
        p = pv.Plotter(window_size=[1000, 1000])

        for obj, color in zip(objects, colors):
            p.add_mesh(obj, color=color)

        p.view_isometric()
        # p.view_xy()
        p.show_axes()
        p.show_bounds(color='black')
        p.background_color = 'white'
        p.show()
