import torch
from itertools import combinations, product
import pyvista as pv
from openmdao.core.explicitcomponent import ExplicitComponent
from torch.autograd.functional import jacobian


class KinematicsInterface(ExplicitComponent):

    def setup(self):

        self.kinematics = self.options['kinematics']

        translation_default = torch.zeros((self.kinematics.num_components, 3))
        rotation_default = torch.zeros((self.kinematics.num_components, 3))

        # TODO what to do if empty?
        routing_default = torch.zeros((self.kinematics.num_segments, 3))

        f_default = self.kinematics.calculate_objective(translation_default, rotation_default, routing_default)
        g_default = self.kinematics.calculate_constraints(translation_default, rotation_default, routing_default)

        self.add_input('translation', val=translation_default)
        self.add_input('rotation', val=rotation_default)
        self.add_input('routing', val=routing_default)
        self.add_output('f', val=f_default)
        self.add_output('g', val=g_default)

        # TODO Set up collocation_constraints


    def setup_partials(self):
        self.declare_partials('f', ['translation', 'rotation', 'routing'])
        self.declare_partials('g', ['translation', 'rotation', 'routing'])

    def compute(self, inputs, outputs):

        translation = inputs['translation']
        rotation = inputs['rotation']
        routing = inputs['routing']

        translation = torch.tensor(translation, dtype=torch.float64)
        rotation = torch.tensor(rotation, dtype=torch.float64)
        routing = torch.tensor(routing, dtype=torch.float64)

        f = self.kinematics.calculate_objective(translation, rotation, routing)
        g = self.kinematics.calculate_constraints(translation, rotation, routing)

        f = f.detach().numpy()
        g = g.detach().numpy()

        outputs['f'] = f
        outputs['g'] = g

    def compute_partials(self, inputs, partials):

        translation = inputs['translation']
        rotation = inputs['rotation']
        routing = inputs['routing']

        translation = torch.tensor(translation, dtype=torch.float64, requires_grad=True)
        rotation = torch.tensor(rotation, dtype=torch.float64, requires_grad=True)
        routing = torch.tensor(routing, dtype=torch.float64, requires_grad=True)

        jac_f = jacobian(self.kinematics.calculate_objective, [translation, rotation, routing])
        jac_g = jacobian(self.kinematics.calculate_constraints, [translation, rotation, routing])

        jac_f_translation = jac_f[0]
        jac_f_rotation = jac_f[1]
        jac_f_routing = jac_f[2]

        jac_g_translation = jac_g[0]
        jac_g_rotation = jac_g[1]
        jac_g_routing = jac_g[2]

        jac_f_translation = jac_f_translation.detach().numpy()
        jac_f_rotation = jac_f_rotation.detach().numpy()
        jac_f_routing = jac_f_routing.detach().numpy()

        jac_g_translation = jac_g_translation.detach().numpy()
        jac_g_rotation = jac_g_rotation.detach().numpy()
        jac_g_routing = jac_g_routing.detach().numpy()

        partials['f', 'translation'] = jac_f_translation
        partials['f', 'rotation'] = jac_f_rotation
        partials['f', 'routing'] = jac_f_routing

        partials['g', 'translation'] = jac_g_translation
        partials['g', 'rotation'] = jac_g_rotation
        partials['g', 'routing'] = jac_g_routing
