import jax.numpy as jnp

def kreisselmeier_steinhauser_max(constraints, rho=100):
    """
    Computes the Kreisselmeier-Steinhauser (KS) aggregation for the maximum constraint value,
    accounting for operator overflow, using JAX.

    Parameters:
    - constraints: A JAX array containing constraint values.
    - rho: A positive scalar that controls the sharpness of the approximation.

    Returns:
    - The smooth maximum of the constraint values.
    """

    # Avoid overflow by subtracting the maximum value from all constraints
    max_constraint = jnp.max(constraints)
    shifted_constraints = constraints - max_constraint

    # Compute the KS aggregation on the shifted constraints
    ks_aggregated_max = max_constraint + jnp.log(jnp.sum(jnp.exp(rho * shifted_constraints))) / rho

    return ks_aggregated_max


def kreisselmeier_steinhauser_min(constraints, rho=100):
    """
    Computes the Kreisselmeier-Steinhauser (KS) aggregation for the minimum constraint value,
    accounting for operator overflow, using JAX.

    Parameters:
    - constraints: A JAX array containing constraint values.
    - rho: A positive scalar that controls the sharpness of the approximation.

    Returns:
    - The smooth minimum of the constraint values.
    """

    # Negate the constraints to target the minimum
    neg_constraints = -constraints

    # Avoid overflow by subtracting the maximum (now minimum since negated) value
    max_neg_constraint = jnp.max(neg_constraints)
    shifted_neg_constraints = neg_constraints - max_neg_constraint

    # Compute the KS aggregation on the shifted, negated constraints
    ks_aggregated_neg = max_neg_constraint + jnp.log(jnp.sum(jnp.exp(rho * shifted_neg_constraints))) / rho

    # Negate the result to obtain the smooth minimum
    ks_aggregated_min = -ks_aggregated_neg

    return ks_aggregated_min