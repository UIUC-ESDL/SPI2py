import torch
from itertools import combinations, product
import pyvista as pv
from openmdao.core.explicitcomponent import ExplicitComponent
from torch.autograd.functional import jacobian


class KinematicsInterface(ExplicitComponent):

    def setup(self):

        self.kinematics = self.options['kinematics']
        x_default = torch.zeros(self.kinematics.design_vector_size)
        f_default = self.kinematics.calculate_objective(x_default)
        g_default = self.kinematics.calculate_constraints(x_default)

        # TODO Autoconfigure shape of design vector...

        self.add_input('x', val=x_default)
        self.add_output('f', val=f_default)
        self.add_output('g', val=g_default)

        # TODO Set up collocation_constraints


    def setup_partials(self):
        self.declare_partials('f', 'x')
        self.declare_partials('g', 'x')

    def compute(self, inputs, outputs):

        # TODO Add method to calculate # of collocation constraints

        x = inputs['x']

        x = torch.tensor(x, dtype=torch.float64)

        f = self.kinematics.calculate_objective(x)
        g = self.kinematics.calculate_constraints(x)

        f = f.detach().numpy()
        g = g.detach().numpy()

        outputs['f'] = f
        outputs['g'] = g

    def compute_partials(self, inputs, partials):

        x = inputs['x']

        x = torch.tensor(x, dtype=torch.float64, requires_grad=True)

        jac_f = jacobian(self.kinematics.calculate_objective, x)
        jac_g = jacobian(self.kinematics.calculate_constraints, x)

        jac_f = jac_f.detach().numpy()
        jac_g = jac_g.detach().numpy()

        partials['f', 'x'] = jac_f
        partials['g', 'x'] = jac_g
