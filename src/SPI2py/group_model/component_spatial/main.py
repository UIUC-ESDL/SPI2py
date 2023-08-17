import numpy as np

import torch
from torch.autograd.functional import jacobian

import openmdao.api as om

from itertools import combinations, product
from SPI2py.group_model.component_spatial.objects import Component, Interconnect

from SPI2py.group_model.component_spatial.distance_calculations import signed_distances

from SPI2py.group_model.utilities import kreisselmeier_steinhauser

from SPI2py.group_model.component_spatial.bounding_volumes import bounding_box
from SPI2py.group_model.component_spatial.visualization import plot_3d


class SpatialComponent(om.ExplicitComponent):

    def initialize(self):

        self.options.declare('name', types=str)
        self.options.declare('components', types=list)
        self.options.declare('interconnects', types=list)

        self.add_design_var('x')
        self.add_objective('f')
        self.add_constraint('g', upper=0)

    def setup(self):

        name = self.options['name']
        components = self.options['components']
        interconnects = self.options['interconnects']
        self.spatial_interface = SpatialInterface(name=name,
                                                  components=components,
                                                  interconnects=interconnects)

        x_default = self.spatial_interface.design_vector
        # f_default = self.spatial_interface.calculate_objective(x_default)
        # g_default = self.spatial_interface.calculate_constraints(x_default)

        self.add_input('x', val=x_default)
        self.add_output('f', val=1.0)
        self.add_output('g', val=-1.0)
        # self.add_output('g', val=[-1., -1., -1.])


    def setup_partials(self):
        self.declare_partials('f', 'x')
        self.declare_partials('g', 'x')

    def compute(self, inputs, outputs):

        x = inputs['x']

        x = torch.tensor(x, dtype=torch.float64)

        f = self.spatial_interface.calculate_objective(x)
        g = self.spatial_interface.calculate_constraints(x)

        f = f.detach().numpy()
        g = g.detach().numpy()

        outputs['f'] = f
        outputs['g'] = g
    #
    # def compute_partials(self, inputs, partials):
    #
    #     x = inputs['x']
    #
    #     x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    #
    #     jac_f = jacobian(self.spatial_interface.calculate_objective, x)
    #     jac_g = jacobian(self.spatial_interface.calculate_constraints, x)
    #
    #     jac_f = jac_f.detach().numpy()
    #     jac_g = jac_g.detach().numpy()
    #
    #     partials['f', 'x'] = jac_f
    #     partials['g', 'x'] = jac_g



class SpatialInterface:
    """
    Defines the associative (non-spatial) aspects of systems.

    The layout module of SPI2py may map a single System to multiple SpatialConfigurations, performing gradient-based
    optimization on each SpatialConfiguration in parallel.

    This associative information includes things such as which objects to check check for collision between.
    Movement classes and associated constraints dictate which objects have design variables and if those design
    variables are constrained.

    There are four types of objects:
    1. Static:              Objects that cannot move (but must still be mapped to the spatial configuration)
    2. Independent:         Objects that can move independently of each other
    3. Partially Dependent: Objects that are constrained relative to other object(s) but retain some degree of freedom
    4. Fully Dependent:     Objects that are fully constrained to another object (e.g., the port of a component)

    Note: For the time being we will assume that you cannot chain dependencies.
    For example, we may assert that the position of objects B,C, and D may depend directly on A.
    However, we may not assert that the position of object D depends on C, which depends on the position of B.

    TODO Change positions dict from pass-thru edit to merge edit
    TODO Add a method to consistently order all objects based on their dependency chains
    TODO Add a method to check for circular dependencies and overconstrained objects
    TODO Write unit tests to confirm that all property-decorated functions correctly apply filters
    """

    def __init__(self,
                 name: str,
                 components: list = None,
                 interconnects: list = None):

        self.name = name
        self.components = components
        self.interconnects = interconnects
        self.objects = self.components + self.interconnects

        self.component_component_pairs = self.get_component_component_pairs()
        self.component_interconnect_pairs = self.get_component_interconnect_pairs()
        self.interconnect_interconnect_pairs = self.get_interconnect_interconnect_pairs()
        self.object_pairs = self.get_object_pairs()

        self.objective = None
        self.constraints = [self.constraint_collision_components_components,
                            self.constraint_collision_components_interconnects,
                            self.constraint_collision_interconnects_interconnects]

    def __repr__(self):
        return f'System({self.name})'

    def get_component_component_pairs(self):
        return list(combinations(self.components, 2))

    def get_component_interconnect_pairs(self):
        # Create a list of all component-interconnect pairs
        pairs = list(product(self.components, self.interconnects))

        # Remove pairs that contain a component and its own interconnect
        check_pairs = []
        for component, interconnect in pairs:
            if component.__repr__() != interconnect.component_1_name and component.__repr__() != interconnect.component_2_name:
                check_pairs.append((component, interconnect))

        return check_pairs


    def get_interconnect_interconnect_pairs(self):
        # Create a list of all interconnect pairs
        pairs = list(combinations(self.interconnects, 2))
        return pairs

    def get_object_pairs(self):
        object_pairs = []
        object_pairs += [self.component_component_pairs]
        object_pairs += [self.component_interconnect_pairs]
        object_pairs += [self.interconnect_interconnect_pairs]
        return object_pairs

    @property
    def design_vector(self):
        """
        Flattened format is necessary for the optimizer...

        Returns
        -------

        """

        design_vector = np.empty(0)

        for obj in self.objects:
            design_vector = np.concatenate((design_vector, obj.design_vector))

        return design_vector

    # def assemble_transformation_matrices

    def decompose_design_vector(self, design_vector):
        """
        Since design vectors are irregularly sized, come up with indexing scheme.
        Not the most elegant method.

        Takes argument b.c. new design vector...

        :return:
        """

        # Get the size of the design vector for each design vector object
        design_vector_sizes = []
        for i, obj in enumerate(self.objects):
            design_vector_sizes.append(len(obj.design_vector))

        # Index values
        start, stop = 0, 0
        design_vectors = []

        for i, size in enumerate(design_vector_sizes):
            # Increment stop index

            stop += design_vector_sizes[i]

            design_vectors.append(design_vector[start:stop])

            # Increment start index
            start = stop

        return design_vectors

    def get_component_positions(self):
        """
        Get the positions of all objects in the layout.

        :return:
        """

        positions_dict = {}

        for obj in self.components:
            positions_dict[obj.__repr__()] = {'positions': obj.positions}

        return positions_dict


    def calculate_positions(self, design_vector=None, design_vector_dict=None):
        """

        :param design_vector:
        :param design_vector:
        :return:

        TODO handle both design vector and design vector dict
        TODO get positions for interconnects, structures, etc
        TODO Remove unnecessary design vector arguments
        """

        if design_vector is not None:
            design_vectors = self.decompose_design_vector(design_vector)
        else:
            design_vectors = [design_vector_dict[repr(obj)] for obj in self.objects]

        objects_dict = {}

        for component in self.components:
            object_dict = component.calculate_positions(design_vector=design_vectors[0])
            objects_dict = {**objects_dict, **object_dict}

        for interconnect in self.interconnects:
            object_dict = interconnect.calculate_positions(design_vector=design_vectors[1], objects_dict=objects_dict)
            objects_dict = {**objects_dict, **object_dict}


        return objects_dict

    def set_default_positions(self, default_positions_dict):

        objects_dict = {}

        # Map components first
        for component in self.components:
            translation, rotation, scale = list(default_positions_dict[component.__repr__()].values())
            translation = torch.tensor(translation, dtype=torch.float64).reshape(3, 1)
            rotation = torch.tensor(rotation, dtype=torch.float64).reshape(3, 1)
            scale = torch.tensor(scale, dtype=torch.float64).reshape(3, 1)
            component.set_default_positions(translation, rotation, scale)
            objects_dict = {**objects_dict, **component.object_dict}

        # Now interconnects
        for interconnect in self.interconnects:
            waypoints = list(default_positions_dict[interconnect.__repr__()].values())
            waypoints = torch.tensor(waypoints, dtype=torch.float64)
            interconnect.set_default_positions(waypoints, objects_dict)


    def set_positions(self, objects_dict):
        """set_positions Sets the positions of the objects in the layout.

        Takes a flattened design vector and sets the positions of the objects in the layout.

        :param positions_dict: A dictionary of positions for each object in the layout.
        :type positions_dict: dict
        """

        for obj in self.objects:
            obj.update_positions(objects_dict=objects_dict)


    def set_objective(self, objective: str):

        """
        Add an objective to the design study.

        TODO Move objective to the model module...?

        :param objective: The objective function to be added.
        :param options: The options for the objective function.
        """

        # SELECT THE OBJECTIVE FUNCTION HANDLE

        if objective == 'bounding box volume':
            _objective_function = bounding_box
        else:
            raise NotImplementedError

        def objective_function(x):
            return _objective_function(x, self)

        self.objective = objective_function

    def constraint_collision_components_components(self, x):
        object_pair = self.component_component_pairs
        signed_distance_vals = signed_distances(x, self, object_pair)
        max_signed_distance = kreisselmeier_steinhauser(signed_distance_vals)
        return max_signed_distance

    def constraint_collision_components_interconnects(self, x):
        object_pair = self.component_interconnect_pairs
        signed_distance_vals = signed_distances(x, self, object_pair)
        max_signed_distance = kreisselmeier_steinhauser(signed_distance_vals)
        return max_signed_distance

    def constraint_collision_interconnects_interconnects(self, x):
        object_pair = self.interconnect_interconnect_pairs
        signed_distance_vals = signed_distances(x, self, object_pair)
        max_signed_distance = kreisselmeier_steinhauser(signed_distance_vals)
        return max_signed_distance

    def calculate_objective(self, x):

        objective = self.objective(x)

        return objective

    def calculate_constraints(self, x):
        # TODO Replace with all

        # constraints = [constraint_function(x) for constraint_function in self.constraints]
        #
        # constraints = np.array(constraints)

        constraints = self.constraints[1](x)

        return constraints


    def plot(self):
        """
        Plot the model at a given state.

        TODO add option to plot all design vectors
        TODO add option to plot design vector--> move to system object

        :param x: Design vector
        """


        objects = []
        colors = []

        for obj in self.objects:
            objs, cols = obj.generate_plot_objects()
            objects.extend(objs)
            colors.extend(cols)

        fig = plot_3d(objects, colors)
