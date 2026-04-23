import jax.numpy as jnp
import numpy as np
from jax import jacrev
from openmdao.api import ExplicitComponent
from SPI2py.models.geometry.bounds import smooth_bounding_box_bounds, bounding_box_volume
from SPI2py.models.physics.lumped.pressure_drop import WATER, pressure_drop_primal


class BoundingBoxVolume(ExplicitComponent):

    def setup(self):
        self.add_input('centers', shape_by_conn=True)
        self.add_input('radii', shape_by_conn=True)

        self.add_output('volume', shape=(1,))
        self.add_output('bounds', shape=(6,))

    def setup_partials(self):
        self.declare_partials('bounds', 'centers', method='exact')
        self.declare_partials('volume', 'centers', method='exact')
        self.declare_partials('bounds', 'radii', method='exact')
        self.declare_partials('volume', 'radii', method='exact')

    def compute(self, inputs, outputs):

        # Get the input variables
        positions = jnp.array(inputs['centers'])
        radii = jnp.array(inputs['radii'])

        # Calculate the bounding box bounds and volume
        volume, bounds = self._compute_primal(positions, radii)

        # Set the outputs
        outputs['bounds'] = bounds
        outputs['volume'] = volume

    def compute_partials(self, inputs, partials):

        # Get the input variables
        positions = jnp.array(inputs['centers'])
        radii = jnp.array(inputs['radii'])

        # Calculate the jacobian of the bounding box volume
        jac_volume, jac_bounds = jacrev(self._compute_primal, argnums=(0, 1))(positions, radii)

        # Set the outputs
        partials['volume', 'centers'] = jac_volume[0]
        partials['volume', 'radii'] = jac_volume[1]
        partials['bounds', 'centers'] = jac_bounds[0]
        partials['bounds', 'radii'] = jac_bounds[1]


    @staticmethod
    def _compute_primal(positions, radii):
        bounds = smooth_bounding_box_bounds(positions, radii)
        volume = bounding_box_volume(bounds)
        return volume, bounds


class PressureDrop(ExplicitComponent):

    def initialize(self):
        self.options.declare('fluid', default=WATER, desc='Fluid properties for the pipe flow')
        self.options.declare('bend_radius_ratio', default=3.0, types=(int, float),
                             desc='Bend radius divided by pipe diameter')

    def setup(self):
        self.add_input('coordinates', shape_by_conn=True)
        self.add_input('pipe_radius', shape_by_conn=True)
        self.add_input('flow_rate', val=1.0)

        self.add_output('pressure_drop', shape=(1,))

    def setup_partials(self):
        self.declare_partials('pressure_drop', 'coordinates', method='exact')
        self.declare_partials('pressure_drop', 'pipe_radius', method='exact')
        self.declare_partials('pressure_drop', 'flow_rate', method='exact')

    def compute(self, inputs, outputs):
        coordinates = jnp.array(inputs['coordinates'])
        pipe_radius = jnp.array(inputs['pipe_radius'])
        flow_rate = jnp.array(inputs['flow_rate'])

        outputs['pressure_drop'] = np.asarray(
            self._compute_primal(coordinates, pipe_radius, flow_rate)
        )

    def compute_partials(self, inputs, partials):
        coordinates = jnp.array(inputs['coordinates'])
        pipe_radius = jnp.array(inputs['pipe_radius'])
        flow_rate = jnp.array(inputs['flow_rate'])

        jac_coordinates, jac_radius, jac_flow = jacrev(
            self._compute_primal, argnums=(0, 1, 2)
        )(coordinates, pipe_radius, flow_rate)

        partials['pressure_drop', 'coordinates'] = np.asarray(jac_coordinates).reshape(1, -1)
        partials['pressure_drop', 'pipe_radius'] = np.asarray(jac_radius).reshape(1, -1)
        partials['pressure_drop', 'flow_rate'] = np.asarray(jac_flow).reshape(1, -1)

    def _compute_primal(self, coordinates, pipe_radius, flow_rate):
        fluid = self.options['fluid']
        return pressure_drop_primal(
            coordinates,
            pipe_radius,
            flow_rate,
            fluid.density,
            fluid.dynamic_viscosity,
            self.options['bend_radius_ratio'],
        )
