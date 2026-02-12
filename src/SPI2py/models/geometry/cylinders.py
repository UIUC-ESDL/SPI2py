import jax.numpy as jnp


def create_cylinders(points, radius):
    x1 = points[:-1]  # Start positions (-1, 3)
    x2 = points[1:]
    if len(radius)>1:  # FIXME
        radius = radius[0][0]# Stop positions (-1, 3)
    r  = jnp.full((x1.shape[0], 1), radius)
    return x1, x2, r


