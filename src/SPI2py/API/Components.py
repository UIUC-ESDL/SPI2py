import torch
from torch.autograd.functional import jacobian
from openmdao.api import ExplicitComponent

from ..models.geometry.finite_sphere_method import read_xyzr_file
from ..models.kinematics.rigid_body_transformations import assemble_transformation_matrix, \
    apply_transformation_matrix


class Component(ExplicitComponent):

    def initialize(self):
        self.options.declare('spheres_filepath', types=str)
        self.options.declare('ports', types=dict)
        self.options.declare('color', types=str)

    def setup(self):

        # Get the options
        self.sphere_positions, self.sphere_radii = read_xyzr_file(self.options['spheres_filepath'])
        self.port_names = self.options['ports'].keys()
        self.port_positions = torch.tensor([self.options['ports'][port_name] for port_name in self.port_names])
        self.color = self.options['color']

        #
        self.add_input('translation', val=torch.zeros((1, 3)), shape=(1, 3))
        self.add_input('rotation', val=torch.zeros((1, 3)), shape=(1, 3))

        # TODO Specify individual ports... ?
        self.add_output('sphere_positions', val=self.sphere_positions)
        self.add_output('sphere_radii', val=self.sphere_radii)
        self.add_output('port_positions', val=self.port_positions)

    def setup_partials(self):
        self.declare_partials('sphere_positions', ['translation', 'rotation'])
        self.declare_partials('port_positions', ['translation', 'rotation'])

    def compute(self, inputs, outputs):

        # Get the input variables
        translation = inputs['translation']
        rotation = inputs['rotation']

        # Convert the input variables to torch tensors
        translation = torch.tensor(translation, dtype=torch.float64)
        rotation = torch.tensor(rotation, dtype=torch.float64)

        # # Convert the state variables to torch tensors
        # sphere_positions = self.sphere_positions, dtype=torch.float64)
        # port_positions = torch.tensor(self.port_positions, dtype=torch.float64)

        # Calculate the transformed sphere positions and port positions
        sphere_positions_transformed = self.compute_transformation(self.sphere_positions, translation, rotation)
        port_positions_transformed = self.compute_transformation(self.port_positions, translation, rotation)

        # Convert to numpy
        sphere_positions_transformed = sphere_positions_transformed.detach().numpy()
        port_positions_transformed = port_positions_transformed.detach().numpy()

        # Set the outputs
        outputs['sphere_positions'] = sphere_positions_transformed
        outputs['sphere_radii'] = self.sphere_radii
        outputs['port_positions'] = port_positions_transformed

    def compute_partials(self, inputs, partials):
        # TODO Jacfwd instead of rev?

        # Get the input variables
        translation = inputs['translation']
        rotation = inputs['rotation']

        # Convert the input variables to torch tensors
        translation = torch.tensor(translation, dtype=torch.float64, requires_grad=True)
        rotation = torch.tensor(rotation, dtype=torch.float64, requires_grad=True)

        # Convert the state variables to torch tensors
        # sphere_positions = torch.tensor(self.sphere_positions, dtype=torch.float64, requires_grad=False)
        # port_positions = torch.tensor(self.port_positions, dtype=torch.float64, requires_grad=False)

        # Calculate the Jacobian matrices
        jac_sphere_positions = jacobian(self.compute_transformation, (self.sphere_positions, translation, rotation))
        jac_port_positions = jacobian(self.compute_transformation, (self.port_positions, translation, rotation))

        # Convert to numpy
        jac_sphere_positions = jac_sphere_positions.detach().numpy()
        jac_port_positions = jac_port_positions.detach().numpy()

        # Slice the Jacobian matrices

        # Set the outputs
        partials['sphere_positions', 'translation'] = jac_sphere_positions[0]
        partials['sphere_positions', 'rotation'] = jac_sphere_positions[1]

        partials['port_positions', 'translation'] = jac_port_positions[0]
        partials['port_positions', 'rotation'] = jac_port_positions[1]

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

    def get_design_variables(self):
        # TODO Static (?)

        # Configure the translation design variables
        translation = {'translation': {'ref': 1, 'ref0': 0}}

        # Configure the rotation design variables
        rotation = {'rotation': {'ref': 2 * torch.pi, 'ref0': 0}}

        # Combine the design variables
        design_vars = {**translation, **rotation}

        return design_vars
