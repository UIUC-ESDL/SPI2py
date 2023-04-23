# from .constraint_aggregation import kreisselmeier_steinhauser, p_norm, induced_exponential, induced_power
# from ..geometry.discrete_collision_detection import discrete_collision_detection
#
#
# def add_constraint(self,
#                    constraint,
#                    options):
#     """
#     Add a constraint to the design study.
#     """
#
#     # UNPACK THE OPTIONS
#
#     type = options['type']
#     object_class_1 = options['object class 1']
#     object_class_2 = options['object class 2']
#     constraint_tolerance = options['constraint tolerance']
#     constraint_aggregation = options['constraint aggregation']
#     constraint_aggregation_parameter = options['constraint aggregation parameter']
#
#     # SELECT THE OBJECT PAIR
#
#     if object_class_1 == 'component' and object_class_2 == 'component':
#         object_pair = self.component_component_pairs
#
#     elif object_class_1 == 'component' and object_class_2 == 'interconnect' or \
#             object_class_1 == 'interconnect' and object_class_2 == 'component':
#         object_pair = self.component_interconnect_pairs
#
#     elif object_class_1 == 'interconnect' and object_class_2 == 'interconnect':
#         object_pair = self.interconnect_interconnect_pairs
#
#     else:
#         raise NotImplementedError
#
#     # SELECT THE CONSTRAINT FUNCTION HANDLE
#     if constraint == 'signed distances':
#         def _constraint_function(x):
#             return discrete_collision_detection(x, self, object_pair)
#     else:
#         raise NotImplementedError
#
#     # SELECT THE CONSTRAINT AGGREGATION FUNCTION HANDLE
#     if constraint_aggregation is None:
#         pass
#     elif constraint_aggregation == 'kreisselmeier steinhauser':
#         _constraint_aggregation_function = kreisselmeier_steinhauser
#     elif constraint_aggregation == 'P-norm':
#         _constraint_aggregation_function = p_norm
#     elif constraint_aggregation == 'induced exponential':
#         _constraint_aggregation_function = induced_exponential
#     elif constraint_aggregation == 'induced power':
#         _constraint_aggregation_function = induced_power
#     else:
#         raise NotImplementedError
#
#     # TODO SCALE THE CONSTRAINT FUNCTION
#
#     if constraint_aggregation is None:
#         nlc = NonlinearConstraint(_constraint_function, -np.inf, constraint_tolerance)
#         self.constraint_functions.append(_constraint_function)
#         self.constraints.append(nlc)
#     else:
#         def constraint_aggregation_function(x):
#             return _constraint_aggregation_function(_constraint_function(x), rho=constraint_aggregation_parameter)
#
#         nlc = NonlinearConstraint(constraint_aggregation_function, -np.inf, constraint_tolerance)
#         self.constraint_functions.append(constraint_aggregation_function)
#         self.constraints.append(nlc)