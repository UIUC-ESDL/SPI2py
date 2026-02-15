# Standard imports
import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jacrev
from jax import jvp, vjp
from openmdao.api import ExplicitComponent, Group

# SPI2py imports
from ..models.mechanics.homogenous_transformation import transform_points
from ..models.utilities.input_and_output import read_xyzr_file, read_csv_file
from ..models.utilities.visualization import plot_spheres, plot_capsules, plot_AABB_spheres, plot_translation_sensitivities


class System(Group):
    """
    A group to represent a physical system, which can contain components and interconnects.
    """

    def draw(self, plotter, subplot, prob, debug=True):
        for element in self.system_iter(recurse=False, include_self=False):
            element.draw(plotter, subplot, prob, debug=True)



class Components(Group):
    """
    A group to logically organize the components of a system.
    """

    def draw(self, plotter, subplot, prob, debug=True):
        for component in self.system_iter(recurse=False, include_self=False):
            component.draw(plotter, subplot, prob, debug=True)


class Interconnects(Group):
    """
    A group to logically organize the interconnects of a system.
    """

    def draw(self, plotter, subplot, prob, debug=True):
        for interconnect in self.system_iter(recurse=False, include_self=False):
            interconnect.draw(plotter, subplot, prob, debug=True)



class MDBDComponent(ExplicitComponent):

    def initialize(self):
        self.options.declare('filepath', types=str)
        self.options.declare('color', types=str)
        self.options.declare('port_positions', types=list)
        self.options.declare('minimum_radius', types=(int, float))

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

        # TODO output MDBD Volume, AABB bounds, etc.

    # def setup_partials(self):
    #
    #     # Declare the partials for the outputs wrt the design variables
    #     self.declare_partials('updated_sphere_positions', ['translation', 'rotation'])
    #     self.declare_partials('updated_ports', ['translation', 'rotation'])
    #
    #     self.declare_partials('updated_sphere_radii', 'sphere_radii', val=1.0)
    #
    #
    #     # Declare the partials for the outputs wrt the static inputs
    #     # Note: The default check_partials step size of 1e-6 results in numerical errors on
    #     # some off-diagonal terms, which raises an error about non-zero rows and columns. Use 1e-4.
    #     I_s = jnp.eye(self.num_spheres * 3)
    #     rows_s, cols_s = jnp.where(I_s)
    #     self.declare_partials('updated_sphere_positions', 'sphere_positions', rows=rows_s, cols=cols_s, val=1.0, method='exact')
    #
    #     I_p = jnp.eye(self.num_ports * 3)
    #     rows_p, cols_p = jnp.where(I_p)
    #     self.declare_partials('updated_ports', 'ports', rows=rows_p, cols=cols_p, val=1.0,
    #                           method='exact')
    #
    #     I_r = jnp.eye(self.num_spheres)
    #     rows_r, cols_r = jnp.where(I_r)
    #     self.declare_partials('updated_sphere_radii', 'sphere_radii', rows=rows_r, cols=cols_r, val=1.0, method='exact')

    def setup_partials(self):
        self.declare_partials('updated_sphere_positions',
                              ['sphere_positions', 'ports', 'translation', 'rotation'],
                              method='exact')
        self.declare_partials('updated_ports',
                              ['sphere_positions', 'ports', 'translation', 'rotation'],
                              method='exact')
        self.declare_partials('updated_sphere_radii', 'sphere_radii', method='exact')

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

    # def compute_partials(self, inputs, partials):
    #
    #     # Get the input variables
    #     sphere_positions = inputs['sphere_positions']
    #     sphere_radii = inputs['sphere_radii']
    #     ports = inputs['ports']
    #     translation = inputs['translation']
    #     rotation = inputs['rotation']
    #
    #     # Convert the input variables to Jax arrays
    #     sphere_positions = jnp.array(sphere_positions)
    #     ports = jnp.array(ports)
    #     translation = jnp.array(translation)
    #     rotation = jnp.array(rotation)
    #
    #     # Define the Jacobian matrices using PyTorch Autograd
    #     jac_fun = jacfwd(self._compute_primal, argnums=(2, 3))
    #
    #     # Evaluate the Jacobian matrices
    #     jac_sphere_positions_val, jac_ports_val = jac_fun(sphere_positions, ports, translation, rotation)
    #
    #     # Slice the Jacobian matrices
    #     grad_sphere_positions_translation = jac_sphere_positions_val[0]
    #     grad_sphere_positions_rotation = jac_sphere_positions_val[1]
    #     grad_ports_translation = jac_ports_val[0]
    #     grad_ports_rotation = jac_ports_val[1]
    #
    #     # Set the outputs
    #     partials['updated_sphere_positions', 'translation'] = grad_sphere_positions_translation
    #     partials['updated_sphere_positions', 'rotation'] = grad_sphere_positions_rotation
    #     partials['updated_ports', 'translation'] = grad_ports_translation
    #     partials['updated_ports', 'rotation'] = grad_ports_rotation

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):
        # Primals
        sphere_positions = jnp.array(inputs['sphere_positions'])
        ports = jnp.array(inputs['ports'])
        translation = jnp.array(inputs['translation'])
        rotation = jnp.array(inputs['rotation'])
        sphere_radii = jnp.array(inputs['sphere_radii'])

        primals = (sphere_positions, ports, translation, rotation)

        if mode == 'fwd':
            # Tangent seeds (only for vars that appear in d_inputs)
            t_sphere_positions = jnp.array(
                d_inputs['sphere_positions']) if 'sphere_positions' in d_inputs else jnp.zeros_like(sphere_positions)
            t_ports = jnp.array(d_inputs['ports']) if 'ports' in d_inputs else jnp.zeros_like(ports)
            t_translation = jnp.array(d_inputs['translation']) if 'translation' in d_inputs else jnp.zeros_like(
                translation)
            t_rotation = jnp.array(d_inputs['rotation']) if 'rotation' in d_inputs else jnp.zeros_like(rotation)

            _, tangent_out = jvp(self._compute_primal, primals,
                                 (t_sphere_positions, t_ports, t_translation, t_rotation))

            # _compute_primal returns: (updated_sphere_positions, updated_ports)
            if 'updated_sphere_positions' in d_outputs:
                d_outputs['updated_sphere_positions'] += np.asarray(tangent_out[0])
            if 'updated_ports' in d_outputs:
                d_outputs['updated_ports'] += np.asarray(tangent_out[1])

            # Pass-through radii: updated_sphere_radii = sphere_radii
            if 'updated_sphere_radii' in d_outputs and 'sphere_radii' in d_inputs:
                d_outputs['updated_sphere_radii'] += np.asarray(d_inputs['sphere_radii'])

        else:  # mode == 'rev'
            _, pullback = vjp(self._compute_primal, *primals)

            # Cotangents for outputs (only what’s present/seeded)
            ct_spheres = jnp.array(
                d_outputs['updated_sphere_positions']) if 'updated_sphere_positions' in d_outputs else jnp.zeros_like(
                sphere_positions)
            ct_ports = jnp.array(d_outputs['updated_ports']) if 'updated_ports' in d_outputs else jnp.zeros_like(ports)

            grads = pullback((ct_spheres, ct_ports))
            # grads correspond to primals in order:
            # 0 sphere_positions, 1 ports, 2 translation, 3 rotation

            if 'sphere_positions' in d_inputs:
                d_inputs['sphere_positions'] += np.asarray(grads[0])
            if 'ports' in d_inputs:
                d_inputs['ports'] += np.asarray(grads[1])
            if 'translation' in d_inputs:
                d_inputs['translation'] += np.asarray(grads[2])
            if 'rotation' in d_inputs:
                d_inputs['rotation'] += np.asarray(grads[3])

            # Pass-through radii adjoint: updated_sphere_radii = sphere_radii
            if 'sphere_radii' in d_inputs and 'updated_sphere_radii' in d_outputs:
                d_inputs['sphere_radii'] += np.asarray(d_outputs['updated_sphere_radii'])


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

    def draw(self, plotter, subplot, prob, opacity=0.5, debug=True):
        centers = prob.get_val(self.pathname + '.' + 'updated_sphere_positions')
        radii   = prob.get_val(self.pathname + '.' + 'updated_sphere_radii')
        color   = self.options['color']
        plot_spheres(plotter, subplot, centers, radii, color, opacity=opacity)

        # TODO Implement utility plots
        if debug:
            plot_AABB_spheres(plotter, subplot, centers, radii, color='gray', opacity=0)
            # plot_translation_sensitivities(plotter, subplot, origin, tot_before_comp_1, color=color,
            #                            factor=2.0)
            # plot_rotation_sensitivities(plotter, subplot, origin, tot_before_comp_2, color=color,
            #                             factor=2.0)
            # plot_stl_file


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

    # def setup_partials(self):
    #
    #     # Declare the partials for the outputs wrt the design variables
    #     self.declare_partials('updated_start_points', ['translation', 'rotation'])
    #     self.declare_partials('updated_end_points', ['translation', 'rotation'])
    #     self.declare_partials('updated_radii', ['translation', 'rotation'])
    #     self.declare_partials('updated_ports', ['translation', 'rotation'])

    def setup_partials(self):
        self.declare_partials('updated_start_points',
                              ['start_points', 'translation', 'rotation'],
                              method='exact')
        self.declare_partials('updated_end_points',
                              ['end_points', 'translation', 'rotation'],
                              method='exact')
        self.declare_partials('updated_ports',
                              ['ports', 'translation', 'rotation'],
                              method='exact')

        # Pass-through radii: updated_radii = radii
        n = int(np.prod(self._inputs['radii'].shape))  # or store radii size in setup
        rows = np.arange(n, dtype=int)
        cols = np.arange(n, dtype=int)
        self.declare_partials('updated_radii', 'radii', rows=rows, cols=cols, val=1.0)

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

    # def compute_partials(self, inputs, partials):
    #     # Get the input variables.
    #     start_points = jnp.array(inputs['start_points'])
    #     end_points = jnp.array(inputs['end_points'])
    #     radii = jnp.array(inputs['radii'])
    #     ports = jnp.array(inputs['ports'])
    #     translation = jnp.array(inputs['translation'])
    #     rotation = jnp.array(inputs['rotation'])
    #
    #     # Compute the Jacobian with respect to translation (argnum 4)
    #     jac_translation = jacfwd(self._compute_primal, argnums=4)(
    #         start_points, end_points, radii, ports, translation, rotation
    #     )
    #     # Compute the Jacobian with respect to rotation (argnum 5)
    #     jac_rotation = jacfwd(self._compute_primal, argnums=5)(
    #         start_points, end_points, radii, ports, translation, rotation
    #     )
    #
    #     # Each jacobian is a tuple of four arrays corresponding to the outputs:
    #     # (updated_start_points, updated_end_points, radii, updated_ports).
    #     # Set the partials for each output.
    #     partials['updated_start_points', 'translation'] = np.asarray(jac_translation[0])
    #     partials['updated_start_points', 'rotation'] = np.asarray(jac_rotation[0])
    #     partials['updated_end_points', 'translation'] = np.asarray(jac_translation[1])
    #     partials['updated_end_points', 'rotation'] = np.asarray(jac_rotation[1])
    #     partials['updated_radii', 'translation'] = np.asarray(jac_translation[2])
    #     partials['updated_radii', 'rotation'] = np.asarray(jac_rotation[2])
    #     partials['updated_ports', 'translation'] = np.asarray(jac_translation[3])
    #     partials['updated_ports', 'rotation'] = np.asarray(jac_rotation[3])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):
        # Primals
        start_points = jnp.array(inputs['start_points'])
        end_points = jnp.array(inputs['end_points'])
        radii = jnp.array(inputs['radii'])
        ports = jnp.array(inputs['ports'])
        translation = jnp.array(inputs['translation'])
        rotation = jnp.array(inputs['rotation'])

        primals = (start_points, end_points, radii, ports, translation, rotation)

        if mode == 'fwd':
            t_start = jnp.array(d_inputs['start_points']) if 'start_points' in d_inputs else jnp.zeros_like(
                start_points)
            t_end = jnp.array(d_inputs['end_points']) if 'end_points' in d_inputs else jnp.zeros_like(end_points)
            t_radii = jnp.array(d_inputs['radii']) if 'radii' in d_inputs else jnp.zeros_like(radii)
            t_ports = jnp.array(d_inputs['ports']) if 'ports' in d_inputs else jnp.zeros_like(ports)
            t_trans = jnp.array(d_inputs['translation']) if 'translation' in d_inputs else jnp.zeros_like(translation)
            t_rot = jnp.array(d_inputs['rotation']) if 'rotation' in d_inputs else jnp.zeros_like(rotation)

            _, tangent_out = jvp(self._compute_primal, primals, (t_start, t_end, t_radii, t_ports, t_trans, t_rot))

            # _compute_primal returns: (updated_start_points, updated_end_points, radii, updated_ports)
            if 'updated_start_points' in d_outputs:
                d_outputs['updated_start_points'] += np.asarray(tangent_out[0])
            if 'updated_end_points' in d_outputs:
                d_outputs['updated_end_points'] += np.asarray(tangent_out[1])
            if 'updated_radii' in d_outputs:
                d_outputs['updated_radii'] += np.asarray(tangent_out[2])
            if 'updated_ports' in d_outputs:
                d_outputs['updated_ports'] += np.asarray(tangent_out[3])

        else:  # mode == 'rev'
            _, pullback = vjp(self._compute_primal, *primals)

            ct_start = jnp.array(
                d_outputs['updated_start_points']) if 'updated_start_points' in d_outputs else jnp.zeros_like(
                start_points)
            ct_end = jnp.array(
                d_outputs['updated_end_points']) if 'updated_end_points' in d_outputs else jnp.zeros_like(end_points)
            ct_radii = jnp.array(d_outputs['updated_radii']) if 'updated_radii' in d_outputs else jnp.zeros_like(radii)
            ct_ports = jnp.array(d_outputs['updated_ports']) if 'updated_ports' in d_outputs else jnp.zeros_like(ports)

            grads = pullback((ct_start, ct_end, ct_radii, ct_ports))
            # grads correspond to primals:
            # 0 start, 1 end, 2 radii, 3 ports, 4 translation, 5 rotation

            if 'start_points' in d_inputs:
                d_inputs['start_points'] += np.asarray(grads[0])
            if 'end_points' in d_inputs:
                d_inputs['end_points'] += np.asarray(grads[1])
            if 'radii' in d_inputs:
                d_inputs['radii'] += np.asarray(grads[2])
            if 'ports' in d_inputs:
                d_inputs['ports'] += np.asarray(grads[3])
            if 'translation' in d_inputs:
                d_inputs['translation'] += np.asarray(grads[4])
            if 'rotation' in d_inputs:
                d_inputs['rotation'] += np.asarray(grads[5])

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
        self.options.declare('color', types=str, default='black')

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
        self.add_output('updated_cyl_radius', shape=(n_segments + 1, 1))

    # def setup_partials(self):
    #     self.declare_partials('updated_cyl_positions', ['start_point', 'control_points', 'end_point'])
    #     self.declare_partials('updated_cyl_radius', ['radius'])

    def setup_partials(self):
        self.declare_partials('updated_cyl_positions',
                              ['start_point', 'control_points', 'end_point'],
                              method='exact')
        self.declare_partials('updated_cyl_radius', 'radius', method='exact')

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


    # def compute_partials(self, inputs, partials):
    #
    #     # Unpack the inputs
    #     start_point = jnp.array(inputs['start_point'])
    #     control_points = jnp.array(inputs['control_points'])
    #     end_point = jnp.array(inputs['end_point'])
    #     radius = jnp.array(inputs['radius'])
    #
    #     # Calculate the partial derivatives
    #     jac_translated_positions = jacfwd(self._compute_primal, argnums=(0, 1, 2))
    #     jac_radius = jacfwd(self._compute_primal, argnums=(3))
    #
    #     jac_translated_positions_val, _ = jac_translated_positions(start_point, control_points, end_point, radius)
    #     _, jac_radius_val = jac_radius(start_point, control_points, end_point, radius)
    #
    #     # Slice the Jacobian
    #     jac_translated_positions_start_point = jac_translated_positions_val[0]
    #     jac_translated_positions_control_points = jac_translated_positions_val[1]
    #     jac_translated_positions_end_point = jac_translated_positions_val[2]
    #     jac_translated_positions_radius = jac_radius_val
    #
    #     # Set the outputs
    #     partials['updated_cyl_positions', 'start_point'] = jac_translated_positions_start_point
    #     partials['updated_cyl_positions', 'control_points'] = jac_translated_positions_control_points
    #     partials['updated_cyl_positions', 'end_point'] = jac_translated_positions_end_point
    #     partials['updated_cyl_radius', 'radius'] = jac_translated_positions_radius

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):
        """
        Matrix-free Jacobian-vector products for Interconnect.

        _compute_primal(start_point, control_points, end_point, radius) returns:
            points: (npts, 3) where points = vstack([start_point, control_points, end_point])
            cyl_radius: (npts, 1) where cyl_radius = radius * ones((npts,1))

        This implementation:
          - uses JAX jvp/vjp for updated_cyl_positions (though it's linear/structured),
          - handles updated_cyl_radius analytically (broadcast / sum) to avoid shape quirks.
        """
        npts = self.options['n_segments'] + 1

        # Primals
        start_point = jnp.array(inputs['start_point'])
        control_points = jnp.array(inputs['control_points'])
        end_point = jnp.array(inputs['end_point'])
        radius = jnp.array(inputs['radius'])  # scalar-like

        primals = (start_point, control_points, end_point, radius)

        if mode == 'fwd':
            # Tangent seeds
            t_start = jnp.array(d_inputs['start_point']) if 'start_point' in d_inputs else jnp.zeros_like(start_point)
            t_ctrl = jnp.array(d_inputs['control_points']) if 'control_points' in d_inputs else jnp.zeros_like(
                control_points)
            t_end = jnp.array(d_inputs['end_point']) if 'end_point' in d_inputs else jnp.zeros_like(end_point)

            # radius tangent might come in as shape (1,) or scalar; normalize to scalar
            if 'radius' in d_inputs:
                t_rad = jnp.asarray(d_inputs['radius']).reshape(())
            else:
                t_rad = jnp.asarray(0.0)

            # JVP through primal for positions (and radius too, but we’ll handle radius analytically)
            _, tangent_out = jvp(self._compute_primal, primals, (t_start, t_ctrl, t_end, t_rad))

            # Unpack tangent outputs: (d_points, d_cyl_radius)
            d_points = tangent_out[0]

            if 'updated_cyl_positions' in d_outputs:
                d_outputs['updated_cyl_positions'] += np.asarray(d_points)

            # Handle radius broadcast analytically (more robust than relying on tangent_out[1])
            if 'updated_cyl_radius' in d_outputs and 'radius' in d_inputs:
                d_outputs['updated_cyl_radius'] += np.asarray(t_rad) * np.ones((npts, 1))

        else:  # mode == 'rev'
            # VJP pullback for positions
            _, pullback = vjp(self._compute_primal, *primals)

            # Cotangents for outputs
            ct_pos = jnp.array(
                d_outputs['updated_cyl_positions']) if 'updated_cyl_positions' in d_outputs else jnp.zeros((npts, 3))
            ct_rad = jnp.zeros((npts, 1))  # we’ll handle radius analytically below

            grads = pullback((ct_pos, ct_rad))
            # grads correspond to primals: (start_point, control_points, end_point, radius)

            if 'start_point' in d_inputs:
                d_inputs['start_point'] += np.asarray(grads[0])
            if 'control_points' in d_inputs:
                d_inputs['control_points'] += np.asarray(grads[1])
            if 'end_point' in d_inputs:
                d_inputs['end_point'] += np.asarray(grads[2])

            # Analytic reverse for broadcast radius: cyl_radius = radius * ones((npts,1))
            if 'radius' in d_inputs and 'updated_cyl_radius' in d_outputs:
                # dradius += sum_i cotangent_i * 1
                d_inputs['radius'] += np.asarray(d_outputs['updated_cyl_radius']).sum()

    def draw(self, plotter, subplot, prob, opacity=0.5, debug=True):
        centers = prob.get_val(self.pathname + '.' + 'updated_cyl_positions')
        radii   = prob.get_val(self.pathname + '.' + 'updated_cyl_radius')
        color   = self.options['color']
        plot_capsules(plotter, subplot, centers, radii, color, opacity=opacity)

        # TODO Implement utility plots
        if debug:
            for segment in range(centers.shape[0] - 1):
                plot_AABB_spheres(plotter, subplot, centers[segment:segment+2], radii[segment:segment+2], color='gray', opacity=0)
        # plot_AABB(plotter, subplot, bounds, color='gray', opacity=0.15)
        # plot_translation_sensitivities(plotter, subplot, origin, tot_before_comp_1, color=color,
        #                                factor=2.0)




