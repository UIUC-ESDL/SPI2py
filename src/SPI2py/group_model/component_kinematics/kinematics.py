import torch
from itertools import combinations, product
import pyvista as pv
from openmdao.core.explicitcomponent import ExplicitComponent
from torch.autograd.functional import jacobian


class KinematicsInterface(ExplicitComponent):

    def setup(self):

        self.kinematics = self.options['kinematics']

        translations_default = torch.zeros((self.kinematics.translations_shape), dtype=torch.float64)
        rotations_default = torch.zeros((self.kinematics.rotations_shape), dtype=torch.float64)
        routings_default = torch.zeros((self.kinematics.routings_shape), dtype=torch.float64)

        f_default = self.kinematics.calculate_objective(translations_default, rotations_default, routings_default)
        g_default = self.kinematics.calculate_constraints(translations_default, rotations_default, routings_default)

        self.add_input('translations', val=translations_default)
        self.add_input('rotations', val=rotations_default)
        self.add_input('routings', val=routings_default)
        self.add_output('f', val=f_default)
        self.add_output('g', val=g_default)


    def setup_partials(self):
        self.declare_partials('f', ['translations', 'rotations', 'routings'])
        self.declare_partials('g', ['translations', 'rotations', 'routings'])

    def compute(self, inputs, outputs):

        translations = inputs['translations']
        rotations = inputs['rotations']
        routings = inputs['routings']

        translations = torch.tensor(translations, dtype=torch.float64)
        rotations = torch.tensor(rotations, dtype=torch.float64)
        routings = torch.tensor(routings, dtype=torch.float64)

        f = self.kinematics.calculate_objective(translations, rotations, routings)
        g = self.kinematics.calculate_constraints(translations, rotations, routings)

        f = f.detach().numpy()
        g = g.detach().numpy()

        outputs['f'] = f
        outputs['g'] = g

    def compute_partials(self, inputs, partials):

        translations = inputs['translations']
        rotations = inputs['rotations']
        routings = inputs['routings']

        translations = torch.tensor(translations, dtype=torch.float64, requires_grad=True)
        rotations = torch.tensor(rotations, dtype=torch.float64, requires_grad=True)
        routings = torch.tensor(routings, dtype=torch.float64, requires_grad=True)

        jac_f = jacobian(self.kinematics.calculate_objective, [translations, rotations, routings])
        jac_g = jacobian(self.kinematics.calculate_constraints, [translations, rotations, routings])

        jac_f_translations = jac_f[0]
        jac_f_rotations = jac_f[1]
        jac_f_routings = jac_f[2]

        jac_g_translations = jac_g[0]
        jac_g_rotations = jac_g[1]
        jac_g_routings = jac_g[2]

        jac_f_translations = jac_f_translations.detach().numpy()
        jac_f_rotations = jac_f_rotations.detach().numpy()
        jac_f_routings = jac_f_routings.detach().numpy()

        jac_g_translations = jac_g_translations.detach().numpy()
        jac_g_rotations = jac_g_rotations.detach().numpy()
        jac_g_routings = jac_g_routings.detach().numpy()

        partials['f', 'translations'] = jac_f_translations
        partials['f', 'rotations'] = jac_f_rotations
        partials['f', 'routings'] = jac_f_routings

        partials['g', 'translations'] = jac_g_translations
        partials['g', 'rotations'] = jac_g_rotations
        partials['g', 'routings'] = jac_g_routings
