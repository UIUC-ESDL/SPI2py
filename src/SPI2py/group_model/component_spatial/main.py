import jax.numpy as np
from dataclasses import dataclass

import openmdao.api as om

from scipy.optimize import NonlinearConstraint
from itertools import combinations, product
from SPI2py.group_model.component_spatial.objects import Component, Interconnect

from SPI2py.group_model.component_spatial.objective_functions import normalized_aggregate_gap_distance
from SPI2py.group_model.component_spatial.collision_detection import discrete_collision_detection

from SPI2py.group_model.utilities import kreisselmeier_steinhauser, p_norm, induced_exponential, induced_power
from SPI2py.group_model.utilities import scale_model_based_objective

from SPI2py.group_model.component_spatial.geometric_representation import read_mdbd_file, generate_rectangular_prisms
from SPI2py.group_model.component_spatial.visualization import plot_3d


@dataclass
class SpatialState:
    """
    Defines the state of a system at a given point in time.
    """
    design_vector: np.ndarray
    objects: str


class SpatialComponent(om.ExplicitComponent):

    def initialize(self):
        pass

    def setup(self):
        pass

    def setup_partials(self):
        pass

    def compute(self, inputs, outputs):
        pass

    def compute_partials(self, inputs, partials):
        pass



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
        self.objectives = []
        self.constraints = []
        self.constraint_functions = []

        self.objects = self.get_objects()

        self.static_objects = self.get_static_objects()
        self.independent_objects = self.get_independent_objects()
        self.partially_dependent_objects = self.get_partially_dependent_objects()
        self.fully_dependent_objects = self.get_fully_dependent_objects()
        self._uncategorized_objects = self._get_uncategorized_objects()

        self.component_component_pairs = self.get_component_component_pairs()
        self.component_interconnect_pairs = self.get_component_interconnect_pairs()
        self.interconnect_interconnect_pairs = self.get_interconnect_interconnect_pairs()
        self.object_pairs = self.get_object_pairs()




    def __repr__(self):
        return f'System({self.name})'

    def __str__(self):
        return self.name

    def get_objects(self):
        return self.components + self.interconnects

    def get_static_objects(self):
        # objects = []
        # for obj in self.objects:
        #     if obj.movement_class == 'static':
        #         objects.append(obj)

        other_objects=[]
        # TODO UPDATE and ref is global
        for obj in self.objects:
            if len(obj.degrees_of_freedom) == 0:
                other_objects.append(obj)

        return other_objects

    def get_independent_objects(self):
        objects = []
        for obj in self.objects:
            if obj.movement_class == 'independent':
                objects.append(obj)
        return objects

    def get_partially_dependent_objects(self):
        objects = []
        for obj in self.objects:
            if obj.movement_class == 'partially dependent':
                objects.append(obj)
        return objects

    def get_fully_dependent_objects(self):
        objects = []
        for obj in self.objects:
            if obj.movement_class == 'fully dependent':
                objects.append(obj)
        return objects

    def _get_uncategorized_objects(self):

        """
        This list should be empty. It checks to make sure every object is categorized as one of the four types of objects.
        """

        uncategorized = []

        categorized = self.static_objects + self.independent_objects + self.partially_dependent_objects + self.fully_dependent_objects

        for obj in self.objects:
            if obj not in categorized:
                uncategorized.append(obj)

        return uncategorized

    def get_component_component_pairs(self):
        """TODO Vectorize with cartesian product"""
        return list(combinations(self.components, 2))


    def get_component_interconnect_pairs(self):
        """
        Pairing logic:
        1. Don't check for collision between a component and the interconnect that is attached to it.

        TODO Write unit tests to ensure it creates the correct pairs
        TODO Vectorize with cartesian product

        :return:
        """

        # Create a list of all component-interconnect pairs

        pairs = list(product(self.components, self.interconnects))
        check_pairs = []
        # Remove pairs that contain a component and its own interconnect
        for component, interconnect in pairs:

            check_pairs.append((component, interconnect))
            # TODO Temporarily disabled
            # if component.__repr__() != interconnect.component_1_name and component.__repr__() != interconnect.component_2_name:
            #     check_pairs.append((component, interconnect))

        return check_pairs


    def get_interconnect_interconnect_pairs(self):
        """
        Pairing logic:
        1. Don't check for collision between two segments of the same interconnect.

        TODO Write unit tests to ensure it creates the correct pairs
        TODO Vectorize with cartesian product
        TODO unmutate list

        :return:
        """

        # Create a list of all interconnect pairs
        pairs = list(combinations(self.interconnects, 2))

        # Remove segments that are from the same interconnect
        # If a pipe intersects itself, usually the pipe can just be made shorter...
        # TODO implement this feature

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

        for obj in self.independent_objects:
            design_vector = np.concatenate((design_vector, obj.design_vector))

        for obj in self.partially_dependent_objects:
            design_vector = np.concatenate((design_vector, obj.design_vector))

        return design_vector

    # TODO Implement
    def decompose_design_vector(self, design_vector: np.ndarray) -> dict:
        """
        Takes a 1D design vector and decomposes it into a dictionary of design variables.
        """

        system_dof = 0
        for obj in self.objects:
            system_dof += obj.dof

        if len(design_vector) != system_dof:
            raise ValueError('The specified design vector must be the same length as the degrees of freedom.')


        design_vector_dict = {}

        for obj in self.objects:
            design_vector_dict[obj] = design_vector[0:obj.dof]
            design_vector = design_vector[obj.dof:]


        return design_vector_dict

    def slice_design_vector(self, design_vector):
        """
        Since design vectors are irregularly sized, come up with indexing scheme.
        Not the most elegant method.

        Takes argument b.c. new design vector...

        :return:
        """

        # Get the size of the design vector for each design vector object
        design_vector_sizes = []
        for i, obj in enumerate(self.independent_objects):
            design_vector_sizes.append(obj.design_vector.size)

        for i, obj in enumerate(self.partially_dependent_objects):
            design_vector_sizes.append(obj.design_vector.size)

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
            design_vectors = self.slice_design_vector(design_vector)
        else:
            design_vectors = list(design_vector_dict.values())

        sums = np.array([np.sum(design_vector) for design_vector in design_vectors])
        objects_dict = np.sum(sums)

        # objects_dict = {}
        #
        #
        # # STATIC OBJECTS
        # for obj in self.static_objects:
        #     object_dict = obj.calculate_positions(None, objects_dict=objects_dict)
        #     objects_dict = {**objects_dict, **object_dict}
        #
        # # DYNAMIC OBJECTS - Independent then Partially Dependent
        # objects = self.independent_objects + self.partially_dependent_objects
        # for obj, design_vector_row in zip(objects, design_vectors):
        #     object_dict = obj.calculate_positions(design_vector_row, objects_dict=objects_dict)
        #     objects_dict = {**objects_dict, **object_dict}
        #
        # # DYNAMIC OBJECTS - Fully Dependent
        # # TODO remove updating edges...? just use lines
        # for obj in self.fully_dependent_objects:
        #     object_dict = obj.calculate_positions(design_vector, objects_dict=objects_dict)
        #     objects_dict = {**objects_dict, **object_dict}
        #
        # # TODO Add method for partially dependent objects
        # # DYNAMIC OBJECTS - Partially Dependent



        return objects_dict


    def set_positions(self, objects_dict):
        """set_positions Sets the positions of the objects in the layout.

        Takes a flattened design vector and sets the positions of the objects in the layout.

        :param positions_dict: A dictionary of positions for each object in the layout.
        :type positions_dict: dict
        """

        for obj in self.objects:
            obj.set_positions(objects_dict)

    def map_static_object(self, object_name, design_vector):
        """
        Maps an object to a spatial configuration.

        :param obj: The object to be mapped.
        :type obj: Component
        :param design_vector: The design vector; must be a (1, 6) array for x, y, z, rx, ry, rz.
        """

        # Get the object given its name as a string
        obj = next(obj for obj in self.objects if obj.__repr__() == object_name)

        # Get the positions of the object
        positions = obj.positions
        radii = obj.radii

        objects_dict = obj.calculate_positions(design_vector, force_update=True)

        # Set the positions of the object
        obj.set_positions(objects_dict)

        return objects_dict

    def add_objective(self,
                      objective,
                      options):

        """
        Add an objective to the design study.

        TODO Move objective to the model module...?

        :param objective: The objective function to be added.
        :param options: The options for the objective function.
        """

        # UNPACK THE OPTIONS

        design_vector_scaling_type   = options['design vector scaling type']
        design_vector_scaling_factor = options['design vector scaling factor']
        objective_scaling_type       = options['objective scaling type']
        objective_scaling_factor     = options['objective scaling factor']


        # SELECT THE OBJECTIVE FUNCTION HANDLE

        if objective == 'normalized aggregate gap distance':
            _objective_function = normalized_aggregate_gap_distance
        else:
            raise NotImplementedError


        # SCALE THE OBJECTIVE FUNCTION


        def objective_function(x):
            return scale_model_based_objective(x, _objective_function, self,
                                               design_vector_scale_type=design_vector_scaling_type,
                                               design_vector_scale_factor=design_vector_scaling_factor,
                                               objective_scale_type=objective_scaling_type,
                                               objective_scale_factor=objective_scaling_factor)

        self.objectives.append(objective_function)

    def add_constraint(self,
                       constraint,
                       options):

        """
        Add a constraint to the design study.
        """

        # UNPACK THE OPTIONS
        cd_type = options['type']
        object_class_1 = options['object class 1']
        object_class_2 = options['object class 2']
        constraint_tolerance = options['constraint tolerance']
        constraint_aggregation = options['constraint aggregation']
        constraint_aggregation_parameter = options['constraint aggregation parameter']

        # SELECT THE OBJECT PAIR

        if object_class_1 == 'component' and object_class_2 == 'component':
            object_pair = self.component_component_pairs

        elif object_class_1 == 'component' and object_class_2 == 'interconnect' or \
                object_class_1 == 'interconnect' and object_class_2 == 'component':
            object_pair = self.component_interconnect_pairs

        elif object_class_1 == 'interconnect' and object_class_2 == 'interconnect':
            object_pair = self.interconnect_interconnect_pairs

        else:
            raise NotImplementedError

        # SELECT THE CONSTRAINT FUNCTION HANDLE
        if constraint == 'signed distances':
            def _constraint_function(x): return discrete_collision_detection(x, self, object_pair, object_class_1, object_class_2)
        else:
            raise NotImplementedError

        # SELECT THE CONSTRAINT AGGREGATION FUNCTION HANDLE
        if constraint_aggregation is None:
            pass
        elif constraint_aggregation == 'kreisselmeier steinhauser':
            _constraint_aggregation_function = kreisselmeier_steinhauser
        elif constraint_aggregation == 'P-norm':
            _constraint_aggregation_function = p_norm
        elif constraint_aggregation == 'induced exponential':
            _constraint_aggregation_function = induced_exponential
        elif constraint_aggregation == 'induced power':
            _constraint_aggregation_function = induced_power
        else:
            raise NotImplementedError

        # TODO SCALE THE CONSTRAINT FUNCTION

        if constraint_aggregation is None:
            nlc = NonlinearConstraint(_constraint_function, -np.inf, constraint_tolerance)
            self.constraint_functions.append(_constraint_function)
            self.constraints.append(nlc)
        else:
            def constraint_aggregation_function(x):
                return _constraint_aggregation_function(_constraint_function(x), rho=constraint_aggregation_parameter)
            nlc = NonlinearConstraint(constraint_aggregation_function, -np.inf, constraint_tolerance)
            self.constraint_functions.append(constraint_aggregation_function)
            self.constraints.append(nlc)

    def calculate_metrics(self,
                          x,
                          requested_metrics=('objective', 'constraints')):
        """
        Calculate the objective function and constraint functions.

        """

        metrics = []
        if 'objective' in requested_metrics:
            objective = self.objectives[0](x)
            metrics.append(objective)

        if 'constraints' in requested_metrics:
            constraints = [constraint_function(x) for constraint_function in self.constraint_functions]
            metrics.append(constraints)


        return metrics


    def plot(self):
        """
        Plot the model at a given state.

        TODO add option to plot all design vectors
        TODO add option to plot design vector--> move to system object

        :param x: Design vector
        """

        # types = []
        # positions = []
        # radii = []
        # colors = []
        objects = []
        colors = []

        for obj in self.objects:
            objs, cols = obj.plot()
            objects.extend(objs)
            colors.extend(cols)
            # if isinstance(obj, Component):
            #     types.append('component')
            #     positions.append(obj.positions)
            #     radii.append(obj.radii)
            #     colors.append(obj.color)
            # elif isinstance(obj, Interconnect):
            #     types.append('interconnect')
            #     directions, heights, centers = obj.capsules
            # else:
            #     raise ValueError('Object type not recognized.')



        fig = plot_3d(objects, colors)
