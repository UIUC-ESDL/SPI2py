import torch
from torch.autograd.functional import jacobian
from openmdao.api import ExplicitComponent
from .kinematics_interface import KinematicsInterface

class KinematicsComponent(ExplicitComponent):

    def initialize(self):

        self.options.declare('name', types=str)
        self.options.declare('components', types=list)
        self.options.declare('interconnects', types=list)

        self.add_design_var('x')
        self.add_objective('f')
        self.add_constraint('g', upper=0)

    def setup(self):

        components = self.options['components']
        interconnects = self.options['interconnects']

        self.kinematics_interface = KinematicsInterface(components=components,
                                                        interconnects=interconnects)

        x_default = self.kinematics_interface.design_vector
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

        f = self.kinematics_interface.calculate_objective(x)
        g = self.kinematics_interface.calculate_constraints(x)

        f = f.detach().numpy()
        g = g.detach().numpy()

        outputs['f'] = f
        outputs['g'] = g

    # TODO Uncomment compute_partials
    # def compute_partials(self, inputs, partials):
    #
    #     x = inputs['x']
    #
    #     x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    #
    #     jac_f = jacobian(self.kinematic_interface.calculate_objective, x)
    #     jac_g = jacobian(self.kinematic_interface.calculate_constraints, x)
    #
    #     jac_f = jac_f.detach().numpy()
    #     jac_g = jac_g.detach().numpy()
    #
    #     partials['f', 'x'] = jac_f
    #     partials['g', 'x'] = jac_g