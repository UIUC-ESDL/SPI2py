import jax.numpy as jnp
from jax import jacrev
from openmdao.api import ExplicitComponent
from SPI2py.models.geometry.bounds import smooth_bounding_box_bounds, bounding_box_volume


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
