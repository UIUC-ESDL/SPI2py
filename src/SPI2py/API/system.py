import numpy as np
import torch
from torch.func import jacrev, jacfwd
import openmdao.api as om
from openmdao.api import ExplicitComponent, Group

from ..models.geometry.point_clouds import generate_point_cloud
from ..models.geometry.finite_sphere_method import read_xyzr_file
from ..models.kinematics.linear_spline_transformations import translate_linear_spline
from ..models.kinematics.rigid_body_transformations import assemble_transformation_matrix, \
    apply_transformation_matrix

class System(Group):

    def initialize(self):
        self.options.declare('input_dict', types=dict)
        self.options.declare('upper', types=(int, float), desc='Upper bound for the translation design variables')
        self.options.declare('lower', types=(int, float), desc='Lower bound for the translation design variables')

    def setup(self):
        input_dict = self.options['input_dict']
        upper = self.options['upper']
        lower = self.options['lower']

        components_dict = input_dict['components']
        interconnects_dict = input_dict['interconnects']

        # Check if the components dictionary is empty
        if len(components_dict) > 0:

            components = om.Group()

            # Create the components
            components_dict = input_dict['components']
            for i, key in enumerate(components_dict.keys()):
                description = components_dict[key]['description']
                filepath = components_dict[key]['filepath']
                n_spheres = components_dict[key]['n_spheres']
                ports = components_dict[key]['ports']
                color = components_dict[key]['color']

                sphere_positions, sphere_radii = read_xyzr_file(filepath, num_spheres=n_spheres)
                # points = read_xyz_file(point_cloud_filepath, n_points)
                # points, n_points_per_1x1x1_cube = generate_point_cloud(point_cloud_filepath, n_points=n_points, plot=False)

                component = Component(description=description,
                                      color=color,
                                      sphere_positions=sphere_positions,
                                      sphere_radii=sphere_radii,
                                      ports=ports,
                                      upper=upper,
                                      lower=lower)

                components.add_subsystem(f'comp_{i}', component)

            self.add_subsystem('components', components)


        # Create the interconnects



        # Check if the interconnects dictionary is empty
        if len(interconnects_dict) > 0:

            interconnects = om.Group()


            for i, key in enumerate(interconnects_dict.keys()):
                component_1 = interconnects_dict[key]['component_1']
                port_1 = interconnects_dict[key]['port_1']
                component_2 = interconnects_dict[key]['component_2']
                port_2 = interconnects_dict[key]['port_2']
                n_segments = interconnects_dict[key]['n_segments']
                n_spheres_per_segment = interconnects_dict[key]['n_spheres_per_segment']
                radius = interconnects_dict[key]['radius']
                color = interconnects_dict[key]['color']

                interconnect = Interconnect(n_segments=n_segments,
                                            n_spheres_per_segment=n_spheres_per_segment,
                                            radius=radius,
                                            color=color)

                interconnects.add_subsystem(f'int_{i}', interconnect)

                # Connect the interconnects to the components
                self.connect(f'components.comp_{component_1}.transformed_ports',
                             f'interconnects.int_{i}.start_point',
                             src_indices=om.slicer[port_1, :])
                self.connect(f'components.comp_{component_2}.transformed_ports',
                             f'interconnects.int_{i}.end_point',
                             src_indices=om.slicer[port_2, :])

            self.add_subsystem('interconnects', interconnects)


class Component(ExplicitComponent):

    def initialize(self):
        self.options.declare('description', types=str)
        self.options.declare('color', types=str)
        self.options.declare('sphere_positions', types=list)
        self.options.declare('sphere_radii', types=list)
        self.options.declare('ports', types=list)
        self.options.declare('upper', types=(int, float), desc='Upper bound for the translation design variables')
        self.options.declare('lower', types=(int, float), desc='Lower bound for the translation design variables')

    def setup(self):

        # Unpack the options
        sphere_positions = self.options['sphere_positions']
        sphere_radii = self.options['sphere_radii']
        ports = self.options['ports']
        upper = self.options['upper']
        lower = self.options['lower']

        # Convert the lists to numpy arrays
        sphere_positions = np.array(sphere_positions).reshape(-1, 3)
        sphere_radii = np.array(sphere_radii).reshape(-1, 1)
        ports = np.array(ports).reshape(-1, 3)

        default_translation = np.array([[0.0, 0.0, 0.0]])
        default_rotation = np.array([[0.0, 0.0, 0.0]])

        # Define the input shapes
        self.add_input('sphere_positions', val=sphere_positions)
        self.add_input('sphere_radii', val=sphere_radii)
        self.add_input('ports', val=ports)
        self.add_input('translation', val=default_translation)
        self.add_input('rotation', val=default_rotation)

        # Define the outputs
        self.add_output('transformed_sphere_positions', val=sphere_positions)
        self.add_output('transformed_sphere_radii', val=sphere_radii)
        self.add_output('transformed_ports', val=ports)

        # Define the design variables
        # TODO Remove upper/lower
        # self.add_design_var('translation', ref=10, lower=lower, upper=upper)
        # self.add_design_var('rotation', ref=2*3.14159)

    def setup_partials(self):
        self.declare_partials('transformed_sphere_positions', ['translation', 'rotation'])
        self.declare_partials('transformed_ports', ['translation', 'rotation'])

    def compute(self, inputs, outputs):

        # Get the input variables
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']
        ports = inputs['ports']
        translation = inputs['translation']
        rotation = inputs['rotation']

        # Convert the input variables to torch tensors
        sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64)
        sphere_radii = torch.tensor(sphere_radii, dtype=torch.float64)
        ports = torch.tensor(ports, dtype=torch.float64)
        translation = torch.tensor(translation, dtype=torch.float64)
        rotation = torch.tensor(rotation, dtype=torch.float64)

        # Calculate the transformed sphere positions and port positions
        sphere_positions_transformed = self.compute_transformation(sphere_positions, translation, rotation)
        ports_transformed = self.compute_transformation(ports, translation, rotation)

        # Calculate the reference density
        # TODO Implement


        # Convert to numpy
        sphere_positions_transformed = sphere_positions_transformed.detach().numpy()
        ports_transformed = ports_transformed.detach().numpy()

        # Set the outputs
        outputs['transformed_sphere_positions'] = sphere_positions_transformed
        outputs['transformed_sphere_radii'] = sphere_radii
        outputs['transformed_ports'] = ports_transformed

    def compute_partials(self, inputs, partials):

        # Get the input variables
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']
        ports = inputs['ports']
        translation = inputs['translation']
        rotation = inputs['rotation']

        # Convert the input variables to torch tensors
        sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64, requires_grad=True)
        sphere_radii = torch.tensor(sphere_radii, dtype=torch.float64)
        ports = torch.tensor(ports, dtype=torch.float64, requires_grad=True)
        translation = torch.tensor(translation, dtype=torch.float64)
        rotation = torch.tensor(rotation, dtype=torch.float64)

        # Define the Jacobian matrices using PyTorch Autograd
        jac_sphere_positions = jacfwd(self.compute_transformation, argnums=(1, 2))
        jac_ports = jacfwd(self.compute_transformation, argnums=(1, 2))

        # Evaluate the Jacobian matrices
        jac_sphere_positions_val = jac_sphere_positions(sphere_positions, translation, rotation)
        jac_ports_val = jac_ports(ports, translation, rotation)

        # Slice the Jacobian matrices
        grad_sphere_positions_translation = jac_sphere_positions_val[0]
        grad_sphere_positions_rotation = jac_sphere_positions_val[1]
        grad_ports_translation = jac_ports_val[0]
        grad_ports_rotation = jac_ports_val[1]

        # Convert to numpy
        grad_sphere_positions_translation = grad_sphere_positions_translation.detach().numpy()
        grad_sphere_positions_rotation = grad_sphere_positions_rotation.detach().numpy()
        grad_ports_translation = grad_ports_translation.detach().numpy()
        grad_ports_rotation = grad_ports_rotation.detach().numpy()

        # Set the outputs
        partials['transformed_sphere_positions', 'translation'] = grad_sphere_positions_translation
        partials['transformed_sphere_positions', 'rotation'] = grad_sphere_positions_rotation
        partials['transformed_ports', 'translation'] = grad_ports_translation
        partials['transformed_ports', 'rotation'] = grad_ports_rotation

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
        self.add_input('start_point', shape=(1, 3))
        self.add_input('control_points', shape=shape_control_points)
        self.add_input('end_point', shape=(1, 3))
        self.add_input('positions', val=positions)
        self.add_input('radii', val=radii)

        # Define the outputs
        self.add_output('transformed_positions', shape=shape_positions)
        self.add_output('transformed_radii', shape=shape_radii)

        # Define the design variables
        self.add_design_var('control_points')

    def setup_partials(self):
        self.declare_partials('transformed_positions', ['start_point', 'control_points', 'end_point'])
        self.declare_partials('transformed_radii', ['start_point', 'control_points', 'end_point'])

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

        print(' ')

    def compute_partials(self, inputs, partials):

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

        # Calculate the partial derivatives
        jac_translated_positions = jacfwd(translate_linear_spline, argnums=(1, 2, 3))
        jac_translated_positions_val = jac_translated_positions(positions, start_point, control_points, end_point, n_spheres_per_segment)

        # Slice the Jacobian
        jac_translated_positions_start_point = jac_translated_positions_val[0].detach().numpy()
        jac_translated_positions_control_points = jac_translated_positions_val[1].detach().numpy()
        jac_translated_positions_end_point = jac_translated_positions_val[2].detach().numpy()

        # Set the outputs
        partials['transformed_positions', 'start_point'] = jac_translated_positions_start_point
        partials['transformed_positions', 'control_points'] = jac_translated_positions_control_points
        partials['transformed_positions', 'end_point'] = jac_translated_positions_end_point
