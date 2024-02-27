import numpy as np
import torch
from torch.func import jacrev, jacfwd
from openmdao.api import ExplicitComponent, Group

from ..models.geometry.finite_sphere_method import read_xyzr_file
from ..models.kinematics.linear_spline_transformations import translate_linear_spline
from ..models.kinematics.rigid_body_transformations import assemble_transformation_matrix, \
    apply_transformation_matrix

class System(Group):

    def initialize(self):
        self.options.declare('input_dict', types=dict)

    def setup(self):
        input_dict = self.options['input_dict']
        self.add_subsystem('components', Components(input_dict=input_dict))
        self.add_subsystem('interconnects', Interconnects(input_dict=input_dict))

class Components(Group):
    """
    A Group used to create Component Objects from an input file and add them to the system.
    """
    def initialize(self):
        self.options.declare('input_dict', types=dict)

    def setup(self):

        # Create the components
        components_dict = self.options['input_dict']['components']
        for i, key in enumerate(components_dict.keys()):
            description = components_dict[key]['description']
            spheres_filepath = components_dict[key]['spheres_filepath']
            num_spheres = components_dict[key]['num_spheres']
            port_positions = components_dict[key]['port_positions']
            color = components_dict[key]['color']

            sphere_positions, sphere_radii = read_xyzr_file(spheres_filepath, num_spheres=num_spheres)

            component = Component(description=description,
                                  color=color,
                                  sphere_positions=sphere_positions,
                                  sphere_radii=sphere_radii,
                                  port_positions=port_positions)

            self.add_subsystem(f'comp_{i}', component)


class Component(ExplicitComponent):

    def initialize(self):
        self.options.declare('description', types=str)
        self.options.declare('color', types=str)
        self.options.declare('sphere_positions', types=list)
        self.options.declare('sphere_radii', types=list)
        self.options.declare('port_positions', types=list)

    def setup(self):

        self.description = self.options['description']
        self.color = self.options['color']

        sphere_positions = self.options['sphere_positions']
        sphere_radii = self.options['sphere_radii']
        port_positions = self.options['port_positions']

        # Convert the lists to numpy arrays
        sphere_positions = np.array(sphere_positions).reshape(-1, 3)
        sphere_radii = np.array(sphere_radii).reshape(-1, 1)
        port_positions = np.array(port_positions).reshape(-1, 3)

        default_translation = np.array([[0.0, 0.0, 0.0]])
        default_rotation = np.array([[0.0, 0.0, 0.0]])

        # Define the input shapes
        self.add_input('sphere_positions', val=sphere_positions)
        self.add_input('sphere_radii', val=sphere_radii)
        self.add_input('port_positions', val=port_positions)
        self.add_input('translation', val=default_translation)
        self.add_input('rotation', val=default_rotation)

        # Define the outputs
        self.add_output('transformed_sphere_positions', val=sphere_positions)
        self.add_output('transformed_sphere_radii', val=sphere_radii)
        self.add_output('transformed_port_positions', val=port_positions)

    def setup_partials(self):
        self.declare_partials('transformed_sphere_positions', ['translation', 'rotation'])
        self.declare_partials('transformed_port_positions', ['translation', 'rotation'])

    def compute(self, inputs, outputs):

        # Get the input variables
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']
        port_positions = inputs['port_positions']
        translation = inputs['translation']
        rotation = inputs['rotation']

        # Convert the input variables to torch tensors
        sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64)
        sphere_radii = torch.tensor(sphere_radii, dtype=torch.float64)
        port_positions = torch.tensor(port_positions, dtype=torch.float64)
        translation = torch.tensor(translation, dtype=torch.float64)
        rotation = torch.tensor(rotation, dtype=torch.float64)

        # Calculate the transformed sphere positions and port positions
        sphere_positions_transformed = self.compute_transformation(sphere_positions, translation, rotation)
        port_positions_transformed = self.compute_transformation(port_positions, translation, rotation)

        # Convert to numpy
        sphere_positions_transformed = sphere_positions_transformed.detach().numpy()
        port_positions_transformed = port_positions_transformed.detach().numpy()

        # Set the outputs
        outputs['transformed_sphere_positions'] = sphere_positions_transformed
        outputs['transformed_sphere_radii'] = sphere_radii
        outputs['transformed_port_positions'] = port_positions_transformed

    def compute_partials(self, inputs, partials):

        # Get the input variables
        sphere_positions = inputs['sphere_positions']
        port_positions = inputs['port_positions']
        translation = inputs['translation']
        rotation = inputs['rotation']

        # Convert the input variables to torch tensors
        sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64, requires_grad=False)
        port_positions = torch.tensor(port_positions, dtype=torch.float64, requires_grad=False)
        translation = torch.tensor(translation, dtype=torch.float64, requires_grad=True)
        rotation = torch.tensor(rotation, dtype=torch.float64, requires_grad=True)

        # Define the Jacobian matrices using PyTorch Autograd
        jac_sphere_positions = jacfwd(self.compute_transformation, argnums=(1, 2))
        jac_port_positions = jacfwd(self.compute_transformation, argnums=(1, 2))

        # Evaluate the Jacobian matrices
        jac_sphere_positions_val = jac_sphere_positions(sphere_positions, translation, rotation)
        jac_port_positions_val = jac_port_positions(port_positions, translation, rotation)

        # Slice the Jacobian matrices
        grad_sphere_positions_translation = jac_sphere_positions_val[0]
        grad_sphere_positions_rotation = jac_sphere_positions_val[1]
        grad_port_positions_translation = jac_port_positions_val[0]
        grad_port_positions_rotation = jac_port_positions_val[1]

        # Convert to numpy
        grad_sphere_positions_translation = grad_sphere_positions_translation.detach().numpy()
        grad_sphere_positions_rotation = grad_sphere_positions_rotation.detach().numpy()
        grad_port_positions_translation = grad_port_positions_translation.detach().numpy()
        grad_port_positions_rotation = grad_port_positions_rotation.detach().numpy()

        # Set the outputs
        partials['transformed_sphere_positions', 'translation'] = grad_sphere_positions_translation
        partials['transformed_sphere_positions', 'rotation'] = grad_sphere_positions_rotation
        partials['transformed_port_positions', 'translation'] = grad_port_positions_translation
        partials['transformed_port_positions', 'rotation'] = grad_port_positions_rotation

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


class Interconnects(Group):
    """
    A Group used to create Interconnect Objects from an input file and add them to the system.
    """
    def initialize(self):
        self.options.declare('input_dict', types=dict)

    def setup(self):

        # Create the interconnects
        interconnects_dict = self.options['input_dict']['interconnects']
        for i, key in enumerate(interconnects_dict.keys()):
            component_1 = interconnects_dict[key]['component_1']
            port_1 = interconnects_dict[key]['port_1']
            component_2 = interconnects_dict[key]['component_2']
            port_2 = interconnects_dict[key]['port_2']
            n_segments = interconnects_dict[key]['n_segments']
            n_spheres_per_segment = interconnects_dict[key]['n_spheres_per_segment']
            radius = interconnects_dict[key]['radius']
            color = interconnects_dict[key]['color']

            # TODO connect the ports...
            interconnect = Interconnect(n_segments=n_segments,
                                        n_spheres_per_segment=n_spheres_per_segment,
                                        radius=radius,
                                        color=color)

            self.add_subsystem(f'int_{i}', interconnect)


class Interconnect(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_segments', types=int)
        self.options.declare('n_spheres_per_segment', types=int)
        self.options.declare('radius', types=float)
        self.options.declare('color', types=str)

    def setup(self):

        # Unpack the options
        n_segments = self.options['n_segments']
        n_spheres_per_segment = self.options['n_spheres_per_segment']
        radius = self.options['radius']

        # Define the input shapes
        shape_control_points = (n_segments - 1, 3)
        shape_positions = (n_spheres_per_segment * n_segments, 3)
        shape_radii = (n_spheres_per_segment * n_segments, 1)

        # Define the input values
        positions = np.zeros(shape_positions)
        radii = radius * np.ones(shape_radii)

        # Define the inputs
        # self.add_input('start_point', shape_by_conn=True)
        self.add_input('start_point', shape=(1, 3))
        self.add_input('control_points', shape=shape_control_points)
        # self.add_input('end_point', shape_by_conn=True)
        self.add_input('end_point', shape=(1, 3))
        self.add_input('positions', val=positions)
        self.add_input('radii', val=radii)

        # Define the outputs
        self.add_output('transformed_positions', shape=shape_positions)
        self.add_output('transformed_radii', shape=shape_radii)

    # def setup_partials(self):
    #     pass

    def compute(self, inputs, outputs):

        # Unpack the inputs
        start_point = inputs['start_point']
        control_points = inputs['control_points']
        end_point = inputs['end_point']
        positions = inputs['positions']
        radii = inputs['radii']

        # Unpack the options
        n_spheres_per_segment = self.options['n_spheres_per_segment']

        # Convert the inputs to torch tensors
        start_point = torch.tensor(start_point, dtype=torch.float64)
        control_points = torch.tensor(control_points, dtype=torch.float64)
        end_point = torch.tensor(end_point, dtype=torch.float64)
        positions = torch.tensor(positions, dtype=torch.float64)
        radii = torch.tensor(radii, dtype=torch.float64)

        # Calculate the positions
        translated_positions = translate_linear_spline(positions, start_point, control_points, end_point, n_spheres_per_segment)

        # Convert the outputs to numpy arrays
        translated_positions = translated_positions.detach().numpy()

        # Set the outputs
        outputs['transformed_positions'] = translated_positions
        outputs['transformed_radii'] = radii

    # def compute_partials(self, inputs, partials):
    #     pass
