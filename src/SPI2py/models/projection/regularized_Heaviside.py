"""
From Norato's paper...
"""

def phi_b(x, z_b):
    pass

def H_tilde(x):
    return 0.5 + 0.75*x - 0.25*x**3

def rho_b(phi_b, r):

    if phi_b/r < -1:
        return 0

    elif -1 <= phi_b/r <= 1:
        return H_tilde(phi_b/r)

    elif phi_b/r > 1:
        return 1

    else:
        raise ValueError('Something went wrong')


# Vectorized versions of the code...