import torch
from torch.autograd.functional import jacobian
from openmdao.api import ExplicitComponent, Group

from ..models.geometry.finite_sphere_method import read_xyzr_file
from ..models.kinematics.rigid_body_transformations import assemble_transformation_matrix, \
    apply_transformation_matrix

class Components(Group):
    pass
    # def setup(self):
    #
    #     # Iterate through components to add design variables
    #     for subsys_name, subsys in self._subsystems_allprocs.items():
    #         if hasattr(subsys, 'design_var_info'):
    #             self.add_design_var(subsys_name + '.' + subsys.design_var_info['name'],
    #                                 ref=subsys.design_var_info['ref'],
    #                                 ref0=subsys.design_var_info['ref0'])







class Component(ExplicitComponent):

    def initialize(self):
        self.options.declare('spheres_filepath', types=str)
        self.options.declare('ports', types=dict)
        self.options.declare('color', types=str)

        # self.design_var_info = {'name':'translation', 'ref':1, 'ref0':0}
        # self.design_var_info = {'name':'rotation', 'ref':2*torch.pi, 'ref0':0}

    def setup(self):

        # Get the options
        initial_sphere_positions, initial_sphere_radii = read_xyzr_file(self.options['spheres_filepath'])
        # port_names = self.options['ports'].keys()
        # initial_port_positions = torch.tensor([self.options['ports'][port_name] for port_name in port_names])
        self.color = self.options['color']

        # Define the input shapes
        self.add_input('sphere_positions', val=initial_sphere_positions)
        self.add_input('sphere_radii', val=initial_sphere_radii)
        # self.add_input('port_positions', val=initial_port_positions)
        self.add_input('translation', val=torch.zeros(1, 3))
        self.add_input('rotation', val=torch.zeros(1, 3))

        # Define the outputs
        self.add_output('transformed_sphere_positions', val=initial_sphere_positions)
        self.add_output('transformed_sphere_radii', val=initial_sphere_radii)
        # self.add_output('transformed_port_positions', val=initial_port_positions)

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
        translation = torch.tensor(translation, dtype=torch.float64)
        rotation = torch.tensor(rotation, dtype=torch.float64)

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
        translation = torch.tensor(translation, dtype=torch.float64, requires_grad=True)
        rotation = torch.tensor(rotation, dtype=torch.float64, requires_grad=True)

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
