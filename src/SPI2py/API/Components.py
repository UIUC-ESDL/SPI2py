import numpy as np
import torch
from torch.autograd.functional import jacobian
from openmdao.api import ExplicitComponent, Group

from ..models.geometry.finite_sphere_method import read_xyzr_file
from ..models.kinematics.rigid_body_transformations import assemble_transformation_matrix, \
    apply_transformation_matrix

class Components(Group):
    def initialize(self):
        self.options.declare('input_dict', types=dict)

    def setup(self):

        # Create the components
        self.aggregate_sphere_positions = []
        self.aggregate_sphere_radii = []
        self.aggregate_port_positins = []

        components_dict = self.options['input_dict']['components']
        for i, key in enumerate(components_dict.keys()):
            name = components_dict[key]['name']
            spheres_filepath = components_dict[key]['spheres_filepath']
            n_spheres = components_dict[key]['n_spheres']
            port_positions = components_dict[key]['port_positions']
            color = components_dict[key]['color']

            sphere_positions, sphere_radii = read_xyzr_file(spheres_filepath, num_spheres=n_spheres)

            component = Component(name=name,
                                  color=color,
                                  sphere_positions=sphere_positions,
                                  sphere_radii=sphere_radii,
                                  port_positions=port_positions)

            self.add_subsystem(f'comp_{i}', component)

            self.aggregate_sphere_positions.append(sphere_positions)
            self.aggregate_sphere_radii.append(sphere_radii)
            self.aggregate_port_positins.append(port_positions)



    def configure(self):

        # TODO Don't override default positions from Component
        components_dict = self.options['input_dict']['components']

        # Determine input/output shapes

        # Get the components
        components = list(components_dict.keys())
        n_components = len(components)
        n_spheres = [components_dict[component]['n_spheres'] for component in components]
        total_n_spheres = sum(n_spheres)

        # Add the Component Objects to the Group
        indices_translations, indices_rotations = self._get_src_indices(n_components)
        indices_spheres, indices_radii = self._get_sphere_indices(n_spheres)

        # Get the default sphere positions and radii from each component setup
        sphere_positions = np.vstack(self.aggregate_sphere_positions)
        sphere_radii = np.hstack(self.aggregate_sphere_radii)
        # default_port_positions = np.vstack(self.aggregate_port_positins)
        self.set_input_defaults('sphere_positions', sphere_positions)
        self.set_input_defaults('sphere_radii', sphere_radii)
        # self.set_input_defaults('port_positions', default_port_positions)


        # Set the default design variable inputs
        default_translation = np.zeros((n_components, 3))
        default_rotation = np.zeros((n_components, 3))

        self.set_input_defaults('translation', default_translation)
        self.set_input_defaults('rotation', default_rotation)


        for component, translations_i, rotations_i, spheres_i, radii_i in zip(components, indices_translations, indices_rotations, indices_spheres, indices_radii):

            self.promotes(f'comp_{component}',
                          inputs=['translation'],
                          src_indices=translations_i,
                          flat_src_indices=True)

            self.promotes(f'comp_{component}',
                          inputs=['rotation'],
                          src_indices=rotations_i,
                          flat_src_indices=True)

            self.promotes(f'comp_{component}',
                          inputs=['sphere_positions'],
                          src_indices=spheres_i,
                          flat_src_indices=True)

            self.promotes(f'comp_{component}',
                            inputs=['sphere_radii'],
                            src_indices=radii_i,
                            flat_src_indices=True)

            # self.promotes(f'comp_{component}',
            #               inputs=['port_positions'],
            #               src_indices=self.indices_ports[i],
            #               flat_src_indices=True)



    # @staticmethod
    # def _create_component(components_dict, key):
    #
    #     comp_name = components_dict[key]['name']
    #     comp_spheres_filepath = components_dict[key]['spheres_filepath']
    #     comp_n_spheres = components_dict[key]['n_spheres']
    #     comp_port_positions = components_dict[key]['port_positions']
    #     comp_color = components_dict[key]['color']
    #
    #     comp_sphere_positions, comp_sphere_radii = read_xyzr_file(comp_spheres_filepath, num_spheres=comp_n_spheres)
    #     # FIXME Make radii (-1, 1) not (-1,)
    #     component = Component(name=comp_name,
    #                           color=comp_color,
    #                           sphere_positions=comp_sphere_positions,
    #                           sphere_radii=comp_sphere_radii,
    #                           port_positions=comp_port_positions)
    #
    #     return component

    @staticmethod
    def _get_src_indices(n_components):

        translations_dof = np.arange(3 * n_components).reshape(-1, 3)
        rotations_dof = np.arange(3 * n_components).reshape(-1, 3)

        # Split up the indices
        indices_translations = np.split(translations_dof, n_components)
        indices_rotations = np.split(rotations_dof, n_components)

        # Convert arrays to lists
        indices_translations = [indices.flatten().tolist() for indices in indices_translations]
        indices_rotations = [indices.flatten().tolist() for indices in indices_rotations]

        return indices_translations, indices_rotations

    @staticmethod
    def _get_sphere_indices(n_spheres):
        # FIXME (-1,3) not (-1, 1)?

        indices_spheres = []
        i = 0
        for n_sphere in n_spheres:
            n_variables = 3 * n_sphere  # 3 for x, y, z
            indices_component_spheres = [j for j in range(i, i + n_variables)]
            indices_spheres.append(indices_component_spheres)
            i += n_sphere

        indices_radii = []
        i = 0
        for n_sphere in n_spheres:
            n_variables = n_sphere  # 1 for radius
            indices_component_radii = [j for j in range(i, i + n_variables)]
            indices_radii.append(indices_component_radii)
            i += n_sphere

        return indices_spheres, indices_radii


    #
    #     # Iterate through components to add design variables
    #     for subsys_name, subsys in self._subsystems_allprocs.items():
    #         if hasattr(subsys, 'design_var_info'):
    #             self.add_design_var(subsys_name + '.' + subsys.design_var_info['name'],
    #                                 ref=subsys.design_var_info['ref'],
    #                                 ref0=subsys.design_var_info['ref0'])







class Component(ExplicitComponent):

    def initialize(self):
        self.options.declare('name', types=str)
        self.options.declare('color', types=str)
        self.options.declare('sphere_positions', types=list)
        self.options.declare('sphere_radii', types=list)
        self.options.declare('port_positions', types=list)

        # self.design_var_info = {'name':'translation', 'ref':1, 'ref0':0}
        # self.design_var_info = {'name':'rotation', 'ref':2*torch.pi, 'ref0':0}

    def setup(self):

        self.name = self.options['name']
        self.color = self.options['color']

        sphere_positions = self.options['sphere_positions']
        sphere_radii = self.options['sphere_radii']
        port_positions = self.options['port_positions']

        # Convert the lists to numpy arrays
        sphere_positions = np.array(sphere_positions)
        sphere_radii = np.array(sphere_radii)
        port_positions = np.array(port_positions)

        default_translation = [0.0, 0.0, 0.0]
        default_rotation = [0.0, 0.0, 0.0]

        # Define the input shapes
        self.add_input('sphere_positions', val=sphere_positions)
        self.add_input('sphere_radii', val=sphere_radii)
        self.add_input('port_positions', val=port_positions)
        self.add_input('translation', val=default_translation)
        self.add_input('rotation', val=default_rotation)

        # Define the outputs
        self.add_output('transformed_sphere_positions', val=sphere_positions)
        self.add_output('transformed_sphere_radii', val=sphere_radii)
        # self.add_output('transformed_port_positions', val=port_positions)

    def setup_partials(self):
        self.declare_partials('transformed_sphere_positions', ['translation', 'rotation'])
        # self.declare_partials('port_positions', ['translation', 'rotation'])
        # self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        # Get the input variables
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']
        # port_positions = inputs['port_positions']
        translation = inputs['translation']
        rotation = inputs['rotation']

        # Convert the input variables to torch tensors
        sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64)
        sphere_radii = torch.tensor(sphere_radii, dtype=torch.float64)
        # port_positions = torch.tensor(port_positions, dtype=torch.float64)
        translation = torch.tensor(translation, dtype=torch.float64).reshape(1, 3)
        rotation = torch.tensor(rotation, dtype=torch.float64).reshape(1, 3)

        # Calculate the transformed sphere positions and port positions
        sphere_positions_transformed = self.compute_transformation(sphere_positions, translation, rotation)
        # port_positions_transformed = self.compute_transformation(port_positions, translation, rotation)

        # Convert to numpy
        sphere_positions_transformed = sphere_positions_transformed.detach().numpy()
        # port_positions_transformed = port_positions_transformed.detach().numpy()

        # Set the outputs
        outputs['transformed_sphere_positions'] = sphere_positions_transformed
        outputs['transformed_sphere_radii'] = sphere_radii
        # outputs['transformed_port_positions'] = port_positions_transformed

    def compute_partials(self, inputs, partials):
        # TODO Jacfwd instead of rev?

        # Get the input variables
        sphere_positions = inputs['sphere_positions']
        # sphere_radii = inputs['sphere_radii']
        # port_positions = inputs['port_positions']
        translation = inputs['translation']
        rotation = inputs['rotation']

        # Convert the input variables to torch tensors
        sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64, requires_grad=False)
        # sphere_radii = torch.tensor(sphere_radii, dtype=torch.float64, requires_grad=True)
        # port_positions = torch.tensor(port_positions, dtype=torch.float64, requires_grad=True)
        translation = torch.tensor(translation, dtype=torch.float64, requires_grad=True).reshape(1, 3)
        rotation = torch.tensor(rotation, dtype=torch.float64, requires_grad=True).reshape(1, 3)

        # Calculate the Jacobian matrices
        jac_sphere_positions = jacobian(self.compute_transformation, (sphere_positions, translation, rotation))
        # jac_port_positions = jacobian(self.compute_transformation, (port_positions, translation, rotation))

        # Slice the Jacobian matrices
        # TODO Verify no zeroth index for sphere_positions
        # grad_sphere_positions_sphere_positions = jac_sphere_positions[0]
        grad_sphere_positions_translation = jac_sphere_positions[1]
        grad_sphere_positions_rotation = jac_sphere_positions[2]

        # grad_port_positions_port_positions = jac_port_positions[0]
        # grad_port_positions_translation = jac_port_positions[1]
        # grad_port_positions_rotation = jac_port_positions[2]

        # Convert to numpy
        # grad_sphere_positions_sphere_positions = grad_sphere_positions_sphere_positions.detach().numpy()
        grad_sphere_positions_translation = grad_sphere_positions_translation.detach().numpy()
        grad_sphere_positions_rotation = grad_sphere_positions_rotation.detach().numpy()

        # grad_port_positions_port_positions = grad_port_positions_port_positions.detach().numpy()
        # grad_port_positions_translation = grad_port_positions_translation.detach().numpy()
        # grad_port_positions_rotation = grad_port_positions_rotation.detach().numpy()

        # Set the outputs
        partials['transformed_sphere_positions', 'translation'] = grad_sphere_positions_translation
        partials['transformed_sphere_positions', 'rotation'] = grad_sphere_positions_rotation

        # partials['port_positions', 'translation'] = grad_port_positions_translation
        # partials['port_positions', 'rotation'] = grad_port_positions_rotation

    @staticmethod
    def compute_transformation(positions, translation, rotation):

        # Assemble the transformation matrix
        t = assemble_transformation_matrix(translation, rotation)

        # Apply the transformation matrix to the sphere positions and port positions
        # Use the translation vector as the origin
        positions_transformed = apply_transformation_matrix(translation.T,
                                                            positions.T,
                                                            t).T

        return positions_transformed

    # def get_design_variables(self):
    #     # TODO Static (?)
    #
    #     # Configure the translation design variables
    #     translation = {'translation': {'ref': 1, 'ref0': 0}}
    #
    #     # Configure the rotation design variables
    #     rotation = {'rotation': {'ref': 2 * torch.pi, 'ref0': 0}}
    #
    #     # Combine the design variables
    #     design_vars = {**translation, **rotation}
    #
    #     return design_vars
