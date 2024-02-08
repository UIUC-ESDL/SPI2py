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
        g_bar_ks = -g_bar_ks
    else:
        raise ValueError('Invalid type. Must be either "max" or "min".')

    # g_bar_ks = torch.max(g) + 1 / rho * torch.log(torch.sum(torch.exp(rho * (g - torch.max(g)))))

    return g_bar_ks




