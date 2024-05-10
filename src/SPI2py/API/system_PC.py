import jax.numpy as jnp
from jax import jacfwd

import openmdao.api as om
from openmdao.api import ExplicitComponent, Group

from ..models.geometry.point_clouds import generate_point_cloud, read_xyz_file
from ..models.utilities.inputs import read_xyzr_file
from ..models.kinematics.spline_transformations import translate_linear_spline
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
                point_cloud_filepath = components_dict[key]['filepath']
                n_points = components_dict[key]['n_points']
                ports = components_dict[key]['ports']
                color = components_dict[key]['color']

                # sphere_positions, sphere_radii = read_xyzr_file(filepath, num_spheres=n_spheres)
                points = read_xyz_file(point_cloud_filepath, n_points)
                # points, n_points_per_1x1x1_cube = generate_point_cloud(point_cloud_filepath, n_points=n_points, plot=False)

                component = Component(description=description,
                                      color=color,
                                      points=points,
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
        self.options.declare('points', types=list)
        self.options.declare('ports', types=list)
        self.options.declare('upper', types=(int, float), desc='Upper bound for the translation design variables')
        self.options.declare('lower', types=(int, float), desc='Lower bound for the translation design variables')

    def setup(self):

        # Unpack the options
        points = self.options['points']
        ports = self.options['ports']
        upper = self.options['upper']
        lower = self.options['lower']

        # Convert the lists to numpy arrays
        points = jnp.array(points).reshape(-1, 3)
        ports = jnp.array(ports).reshape(-1, 3)
        ports = jnp.array(ports).reshape(-1, 3)

        default_translation = jnp.array([[0.0, 0.0, 0.0]])
        default_rotation = jnp.array([[0.0, 0.0, 0.0]])

        # Define the input shapes
        self.add_input('points', val=points)
        self.add_input('ports', val=ports)
        self.add_input('translation', val=default_translation)
        self.add_input('rotation', val=default_rotation)

        # Define the outputs
        self.add_output('transformed_points', val=points)
        self.add_output('transformed_ports', val=ports)

        # Define the design variables
        # TODO Remove upper/lower
        # self.add_design_var('translation', ref=10, lower=lower, upper=upper)
        # self.add_design_var('rotation', ref=2*3.14159)

    def setup_partials(self):
        self.declare_partials('transformed_points', ['translation', 'rotation'])
        self.declare_partials('transformed_ports', ['translation', 'rotation'])

    def compute(self, inputs, outputs):

        # Get the input variables
        points = inputs['points']
        ports = inputs['ports']
        translation = inputs['translation']
        rotation = inputs['rotation']

        # Convert the input variables to Jax arrays
        points = jnp.array(points)
        ports = jnp.array(ports)
        translation = jnp.array(translation)
        rotation = jnp.array(rotation)

        # Calculate the transformed sphere positions and port positions
        points_transformed = self.compute_transformation(points, translation, rotation)
        ports_transformed = self.compute_transformation(ports, translation, rotation)

        # Set the outputs
        outputs['transformed_points'] = points_transformed
        outputs['transformed_ports'] = ports_transformed

    def compute_partials(self, inputs, partials):

        # Get the input variables
        points = inputs['points']
        ports = inputs['ports']
        translation = inputs['translation']
        rotation = inputs['rotation']

        # Convert the input variables to Jax arrays
        points = jnp.array(points)
        ports = jnp.array(ports)
        translation = jnp.array(translation)
        rotation = jnp.array(rotation)

        # Define the Jacobian matrices using PyTorch Autograd
        jac_points = jacfwd(self.compute_transformation, argnums=(1, 2))
        jac_ports = jacfwd(self.compute_transformation, argnums=(1, 2))

        # Evaluate the Jacobian matrices
        jac_points_val = jac_points(points, translation, rotation)
        jac_ports_val = jac_ports(ports, translation, rotation)

        # Slice the Jacobian matrices
        grad_points_translation = jac_points_val[0]
        grad_points_rotation = jac_points_val[1]
        grad_ports_translation = jac_ports_val[0]
        grad_ports_rotation = jac_ports_val[1]

        # Set the outputs
        partials['transformed_points', 'translation'] = grad_points_translation
        partials['transformed_points', 'rotation'] = grad_points_rotation
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
        positions = jnp.zeros(shape_positions)
        radii = radius * jnp.ones(shape_radii)

        # Define the inputs
        self.add_input('start_point', shape=(1, 3))
        self.add_input('control_points', shape=shape_control_points)
        self.add_input('end_point', shape=(1, 3))
        self.add_input('positions', val=positions)
        self.add_input('radii', val=radii)

        # Define the outputs
        self.add_output('transformed_sphere_positions', shape=shape_positions)
        self.add_output('transformed_sphere_radii', shape=shape_radii)
        self.add_output('volume', val=0.0)
        self.add_output('AABB', shape=(1, 6), desc='Axis-aligned bounding box')

        # Define the design variables
        # self.add_design_var('control_points')

    def setup_partials(self):
        self.declare_partials('transformed_sphere_positions', ['start_point', 'control_points', 'end_point'])
        self.declare_partials('transformed_sphere_radii', ['start_point', 'control_points', 'end_point'])

    def compute(self, inputs, outputs):

        # Unpack the inputs
        start_point = inputs['start_point']
        control_points = inputs['control_points']
        end_point = inputs['end_point']
        positions = inputs['positions']
        radii = inputs['radii']

        # Unpack the options
        n_spheres_per_segment = self.options['n_spheres_per_segment']

        # Convert the inputs to Jax arrays
        start_point = jnp.array(start_point)
        control_points = jnp.array(control_points)
        end_point = jnp.array(end_point)
        positions = jnp.array(positions)
        radii = jnp.array(radii)

        # Calculate the positions
        translated_positions = translate_linear_spline(positions, start_point, control_points, end_point, n_spheres_per_segment)

        # Calculate the axis-aligned bounding box
        min_coords = translated_positions - radii
        max_coords = translated_positions + radii
        x_min, y_min, z_min = jnp.min(min_coords, axis=0).reshape(3, 1)
        x_max, y_max, z_max = jnp.max(max_coords, axis=0).reshape(3, 1)
        aabb = jnp.concatenate((x_min, x_max, y_min, y_max, z_min, z_max))

        # Set the outputs
        outputs['transformed_sphere_positions'] = translated_positions
        outputs['transformed_sphere_radii'] = radii
        outputs['AABB'] = aabb

    def compute_partials(self, inputs, partials):

        # Unpack the inputs
        start_point = inputs['start_point']
        control_points = inputs['control_points']
        end_point = inputs['end_point']
        positions = inputs['positions']
        radii = inputs['radii']

        # Unpack the options
        n_spheres_per_segment = self.options['n_spheres_per_segment']

        # Convert the inputs to Jax arrays
        start_point = jnp.array(start_point)
        control_points = jnp.array(control_points)
        end_point = jnp.array(end_point)
        positions = jnp.array(positions)
        radii = jnp.array(radii)

        # Calculate the partial derivatives
        jac_translated_positions = jacfwd(translate_linear_spline, argnums=(1, 2, 3))
        jac_translated_positions_val = jac_translated_positions(positions, start_point, control_points, end_point, n_spheres_per_segment)

        # Slice the Jacobian
        jac_translated_positions_start_point = jac_translated_positions_val[0]
        jac_translated_positions_control_points = jac_translated_positions_val[1]
        jac_translated_positions_end_point = jac_translated_positions_val[2]

        # Set the outputs
        partials['transformed_sphere_positions', 'start_point'] = jac_translated_positions_start_point
        partials['transformed_sphere_positions', 'control_points'] = jac_translated_positions_control_points
        partials['transformed_sphere_positions', 'end_point'] = jac_translated_positions_end_point
