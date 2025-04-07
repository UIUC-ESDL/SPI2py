# Standard imports
import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jacrev
from openmdao.api import ExplicitComponent, Group

# Custom imports
from ..models.mechanics.homogenous_transformation import transform_points
from ..models.utilities.input_and_output import read_xyzr_file, read_csv_file


class System(Group):
    pass



class Components(Group):
    pass


class Interconnects(Group):
    pass


class PointCloudComponent(ExplicitComponent):
    raise NotImplementedError("PointCloudComponent is not implemented yet.")


class MDBDComponent(ExplicitComponent):

    def initialize(self):
        self.options.declare('filepath', types=str)
        self.options.declare('port_positions', types=list)
        self.options.declare('minimum_radius', types=int)

    def setup(self):

        # Unpack the options
        filepath       = self.options['filepath']
        port_positions = self.options['port_positions']
        min_radius     = self.options['minimum_radius']

        # Read the sphere positions and radii from the CSV file
        sphere_positions, sphere_radii = read_csv_file(filepath, min_radius)

        # Convert the lists to numpy arrays
        sphere_positions = np.array(sphere_positions).reshape(-1, 3)
        sphere_radii     = np.array(sphere_radii).reshape(-1, 1)
        ports            = np.array(port_positions).reshape(-1, 3)

        # Determine the number of spheres and ports
        self.num_spheres = sphere_positions.shape[0]
        self.num_ports = ports.shape[0]

        # Define the input shapes
        self.add_input('sphere_positions', val=sphere_positions)
        self.add_input('sphere_radii', val=sphere_radii)
        self.add_input('ports', val=ports)
        self.add_input('translation', val=np.array([0.0, 0.0, 0.0]))
        self.add_input('rotation', val=np.array([0.0, 0.0, 0.0]))

        # Outputs:
        self.add_output('updated_sphere_positions', val=sphere_positions)
        self.add_output('updated_sphere_radii', val=sphere_radii)
        self.add_output('updated_ports', val=ports)

    def setup_partials(self):

        # Declare the partials for the outputs wrt the design variables
        self.declare_partials('updated_sphere_positions', ['translation', 'rotation'])
        self.declare_partials('updated_ports', ['translation', 'rotation'])

        # Declare the partials for the outputs wrt the static inputs
        # Note: The default check_partials step size of 1e-6 results in numerical errors on
        # some off-diagonal terms, which raises an error about non-zero rows and columns. Use 1e-4.
        I_s = jnp.eye(self.num_spheres * 3)
        rows_s, cols_s = jnp.where(I_s)
        self.declare_partials('updated_sphere_positions', 'sphere_positions', rows=rows_s, cols=cols_s, val=1.0, method='exact')

        I_p = jnp.eye(self.num_ports * 3)
        rows_p, cols_p = jnp.where(I_p)
        self.declare_partials('updated_ports', 'ports', rows=rows_p, cols=cols_p, val=1.0,
                              method='exact')

        I_r = jnp.eye(self.num_spheres)
        rows_r, cols_r = jnp.where(I_r)
        self.declare_partials('updated_sphere_radii', 'sphere_radii', rows=rows_r, cols=cols_r, val=1.0, method='exact')

    def compute(self, inputs, outputs):

        # Get the input variables
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']
        port_positions = inputs['ports']
        translation = inputs['translation']
        rotation = inputs['rotation']

        # Calculate the transformed sphere positions and port positions
        sphere_positions_transformed, ports_transformed = self._compute_primal(sphere_positions, port_positions, translation, rotation)

        # Set the outputs
        outputs['updated_sphere_positions'] = sphere_positions_transformed
        outputs['updated_sphere_radii'] = sphere_radii
        outputs['updated_ports'] = ports_transformed

    def compute_partials(self, inputs, partials):

        # Get the input variables
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']
        ports = inputs['ports']
        translation = inputs['translation']
        rotation = inputs['rotation']

        # Convert the input variables to Jax arrays
        sphere_positions = jnp.array(sphere_positions)
        sphere_radii = jnp.array(sphere_radii)
        ports = jnp.array(ports)
        translation = jnp.array(translation)
        rotation = jnp.array(rotation)

        # Define the Jacobian matrices using PyTorch Autograd
        jac_fun = jacfwd(self._compute_primal, argnums=(2, 3))

        # Evaluate the Jacobian matrices
        jac_sphere_positions_val, jac_ports_val = jac_fun(sphere_positions, ports, translation, rotation)

        # Slice the Jacobian matrices
        grad_sphere_positions_translation = jac_sphere_positions_val[0]
        grad_sphere_positions_rotation = jac_sphere_positions_val[1]
        grad_ports_translation = jac_ports_val[0]
        grad_ports_rotation = jac_ports_val[1]

        # Set the outputs
        partials['updated_sphere_positions', 'translation'] = grad_sphere_positions_translation
        partials['updated_sphere_positions', 'rotation'] = grad_sphere_positions_rotation
        partials['updated_ports', 'translation'] = grad_ports_translation
        partials['updated_ports', 'rotation'] = grad_ports_rotation

    @staticmethod
    def _compute_primal(sphere_positions, port_positions, translation, rotation):

        # Get the ref points
        spheres_ref_point = sphere_positions[0]
        ports_ref_point = port_positions[0]

        spheres_positions_transformed = transform_points(sphere_positions,
                                                         spheres_ref_point,
                                               translation.flatten(),
                                               rotation.flatten())

        ports_transformed = transform_points(port_positions,
                                             ports_ref_point,
                                               translation.flatten(),
                                               rotation.flatten())

        return spheres_positions_transformed, ports_transformed


class LinearSplineComponent(ExplicitComponent):

    def initialize(self):
        self.options.declare('start_points', types=list)
        self.options.declare('end_points', types=list)
        self.options.declare('radii', types=list)
        self.options.declare('port_positions', types=list)

    def setup(self):

        # Unpack the options
        start_points = self.options['start_points']
        end_points = self.options['end_points']
        radii = self.options['radii']
        ports = self.options['port_positions']

        # Convert the lists to JAX numpy arrays
        start_points = jnp.array(start_points).reshape(-1, 3)
        end_points   = jnp.array(end_points).reshape(-1, 3)
        radii        = jnp.array(radii).reshape(-1, 1)
        ports        = jnp.array(ports).reshape(-1, 3)

        # Define the input shapes
        self.add_input('start_points', val=start_points)
        self.add_input('end_points', val=end_points)
        self.add_input('radii', val=radii)
        self.add_input('ports', val=ports)
        self.add_input('translation', val=np.array([0.0, 0.0, 0.0]))
        self.add_input('rotation', val=np.array([0.0, 0.0, 0.0]))

        # Outputs:
        self.add_output('updated_start_points', val=start_points)
        self.add_output('updated_end_points', val=end_points)
        self.add_output('updated_radii', val=radii)
        self.add_output('updated_ports', val=ports)

    def setup_partials(self):

        # Declare the partials for the outputs wrt the design variables
        self.declare_partials('updated_start_points', ['translation', 'rotation'])
        self.declare_partials('updated_end_points', ['translation', 'rotation'])
        self.declare_partials('updated_radii', ['translation', 'rotation'])
        self.declare_partials('updated_ports', ['translation', 'rotation'])

    def compute(self, inputs, outputs):

        # Get the input variables
        start_points = jnp.array(inputs['start_points'])
        end_points = jnp.array(inputs['end_points'])
        radii = jnp.array(inputs['radii'])
        ports = jnp.array(inputs['ports'])

        translation = jnp.array(inputs['translation'])
        rotation = jnp.array(inputs['rotation'])

        # Calculate the transformed sphere positions and port positions
        updated_start_points, updated_end_points, updated_radii, updated_ports = self._compute_primal(start_points, end_points, radii, ports, translation, rotation)

        # Set the outputs
        outputs['updated_start_points'] = updated_start_points
        outputs['updated_end_points'] = updated_end_points
        outputs['updated_radii'] = updated_radii
        outputs['updated_ports'] = updated_ports

    def compute_partials(self, inputs, partials):
        # Get the input variables.
        start_points = jnp.array(inputs['start_points'])
        end_points = jnp.array(inputs['end_points'])
        radii = jnp.array(inputs['radii'])
        ports = jnp.array(inputs['ports'])
        translation = jnp.array(inputs['translation'])
        rotation = jnp.array(inputs['rotation'])

        # Compute the Jacobian with respect to translation (argnum 4)
        jac_translation = jacfwd(self._compute_primal, argnums=4)(
            start_points, end_points, radii, ports, translation, rotation
        )
        # Compute the Jacobian with respect to rotation (argnum 5)
        jac_rotation = jacfwd(self._compute_primal, argnums=5)(
            start_points, end_points, radii, ports, translation, rotation
        )

        # Each jacobian is a tuple of four arrays corresponding to the outputs:
        # (updated_start_points, updated_end_points, radii, updated_ports).
        # Set the partials for each output.
        partials['updated_start_points', 'translation'] = np.asarray(jac_translation[0])
        partials['updated_start_points', 'rotation'] = np.asarray(jac_rotation[0])
        partials['updated_end_points', 'translation'] = np.asarray(jac_translation[1])
        partials['updated_end_points', 'rotation'] = np.asarray(jac_rotation[1])
        partials['updated_radii', 'translation'] = np.asarray(jac_translation[2])
        partials['updated_radii', 'rotation'] = np.asarray(jac_rotation[2])
        partials['updated_ports', 'translation'] = np.asarray(jac_translation[3])
        partials['updated_ports', 'rotation'] = np.asarray(jac_rotation[3])

    @staticmethod
    def _compute_primal(start_points, end_points, radii, ports, translation, rotation):

        # Identify the reference point
        reference_point = start_points[0]

        updated_start_points = transform_points(start_points,
                                                reference_point,
                                                translation.flatten(),
                                                rotation.flatten())

        updated_end_points = transform_points(end_points,
                                                reference_point,
                                                translation.flatten(),
                                                rotation.flatten())

        updated_ports = transform_points(ports,
                                                reference_point,
                                                translation.flatten(),
                                                rotation.flatten())

        return updated_start_points, updated_end_points, radii, updated_ports


class Interconnect(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_segments', types=int)
        self.options.declare('radius', types=float)
        self.options.declare('color', types=str)

    def setup(self):

        # Unpack the options
        n_segments = self.options['n_segments']
        radius = self.options['radius']

        # Define the input shapes
        shape_control_points = (n_segments - 1, 3)
        shape_positions = (n_segments + 1, 3)

        # Define the inputs
        # T_in, flow rate, ... and out
        self.add_input('start_point', shape=(1, 3))
        self.add_input('control_points', shape=shape_control_points)
        self.add_input('end_point', shape=(1, 3))
        self.add_input('radius', val=radius)

        # Define the outputs
        self.add_output('updated_cyl_positions', shape=shape_positions)
        # self.add_output('updated_cyl_radius', val=radius)
        self.add_output('updated_cyl_radius', shape=(n_segments + 1, 1))

    def setup_partials(self):
        self.declare_partials('updated_cyl_positions', ['start_point', 'control_points', 'end_point'])
        self.declare_partials('updated_cyl_radius', ['radius'])


    def compute(self, inputs, outputs):

        # Unpack the inputs
        start_point = jnp.array(inputs['start_point'])
        control_points = jnp.array(inputs['control_points'])
        end_point = jnp.array(inputs['end_point'])
        radius = jnp.array(inputs['radius'])

        # Calculate the positions
        points, radius = self._compute_primal(start_point, control_points, end_point, radius)

        # Set the outputs
        outputs['updated_cyl_positions'] = points
        outputs['updated_cyl_radius'] = radius

    @staticmethod
    def _compute_primal(start_point, control_points, end_point, radius):

        points = jnp.vstack([start_point, control_points, end_point])

        # One radius per point
        radius = jnp.full((points.shape[0], 1), radius)

        return points, radius


    def compute_partials(self, inputs, partials):

        # Unpack the inputs
        start_point = jnp.array(inputs['start_point'])
        control_points = jnp.array(inputs['control_points'])
        end_point = jnp.array(inputs['end_point'])
        radius = jnp.array(inputs['radius'])

        # Calculate the partial derivatives
        jac_translated_positions = jacfwd(self._compute_primal, argnums=(0, 1, 2))
        jac_radius = jacfwd(self._compute_primal, argnums=(3))

        jac_translated_positions_val, _ = jac_translated_positions(start_point, control_points, end_point, radius)
        _, jac_radius_val = jac_radius(start_point, control_points, end_point, radius)

        # Slice the Jacobian
        jac_translated_positions_start_point = jac_translated_positions_val[0]
        jac_translated_positions_control_points = jac_translated_positions_val[1]
        jac_translated_positions_end_point = jac_translated_positions_val[2]
        jac_translated_positions_radius = jac_radius_val

        # Set the outputs
        partials['updated_cyl_positions', 'start_point'] = jac_translated_positions_start_point
        partials['updated_cyl_positions', 'control_points'] = jac_translated_positions_control_points
        partials['updated_cyl_positions', 'end_point'] = jac_translated_positions_end_point
        partials['updated_cyl_radius', 'radius'] = jac_translated_positions_radius



