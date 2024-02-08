import torch


def kreisselmeier_steinhauser(g, rho=100, type='max'):
    """
    The Kreisselmeier-Steinhauser (KS) method for constraint aggregation.

    This alternate implementation of the KS
    method is based on MDO book Chapter 5.7 to avoid overflow from the exponential function disproportionally
    weighing higher positive values.

    While this algorithm uses the recommended default of rho=100, it may need to be evaluated for each problem.
    As rho increases, so does the curvature of the constraint aggregation function, which can cause ill-conditioning.

    This function has been modified to handle both max and min constraints. The following relationship is used:
    min(x,y) = -max(-x,-y)

    :param g: A row vector of constraint values.
    :param rho: Aggregation parameter.
    :param type: The type of aggregation. Must be either 'max' or 'min'.
    :return:
    """

    if type == 'max':
        g_bar_ks = torch.max(g) + 1 / rho * torch.log(torch.sum(torch.exp(rho * (g - torch.max(g)))))

    elif type == 'min':
        g = -g
        g_bar_ks = torch.max(g) + 1 / rho * torch.log(torch.sum(torch.exp(rho * (g - torch.max(g)))))
        # g_bar_ks = -g_bar_ks
    else:
        raise ValueError('Invalid type. Must be either "max" or "min".')

    # g_bar_ks = torch.max(g) + 1 / rho * torch.log(torch.sum(torch.exp(rho * (g - torch.max(g)))))

    return g_bar_ks


def kreisselmeier_steinhauser_max(constraints, rho=100):
    """
    Computes the Kreisselmeier-Steinhauser (KS) aggregation for the maximum constraint value,
    accounting for operator overflow.

    Parameters:
    - constraints: A PyTorch tensor containing constraint values.
    - rho: A positive scalar that controls the sharpness of the approximation.

    Returns:
    - The smooth maximum of the constraint values.
    """

    # Avoid overflow by subtracting the maximum value from all constraints
    max_constraint = torch.max(constraints)
    shifted_constraints = constraints - max_constraint

    # Compute the KS aggregation on the shifted constraints
    ks_aggregated_max = max_constraint + torch.log(torch.sum(torch.exp(rho * shifted_constraints))) / rho

    return ks_aggregated_max

def kreisselmeier_steinhauser_min(constraints, rho=100):
    """
    Computes the Kreisselmeier-Steinhauser (KS) aggregation for the minimum constraint value,
    accounting for operator overflow.

    Parameters:
    - constraints: A PyTorch tensor containing constraint values.
    - rho: A positive scalar that controls the sharpness of the approximation.

    Returns:
    - The smooth minimum of the constraint values.
    """

    # Negate the constraints to target the minimum
    neg_constraints = -constraints

    # Avoid overflow by subtracting the maximum (now minimum since negated) value
    max_neg_constraint = torch.max(neg_constraints)
    shifted_neg_constraints = neg_constraints - max_neg_constraint

    # Compute the KS aggregation on the shifted, negated constraints
    ks_aggregated_neg = max_neg_constraint + torch.log(torch.sum(torch.exp(rho * shifted_neg_constraints))) / rho

    # Negate the result to obtain the smooth minimum
    ks_aggregated_min = -ks_aggregated_neg

    return ks_aggregated_min


def induced_power_function(g, rho=3):
    """
    The induced power function method for constraint aggregation.

    TODO Validate the induced power function method for constraint aggregation.
    TODO Add input constrains (i.e.,  g_j >= 0)

    :param g: A row vector of constraint values.
    :param rho: Aggregation parameter.
    :return:
    """

    g_bar_ipf = torch.sum(g**(rho+1)) / torch.sum(g**rho)

    return g_bar_ipf