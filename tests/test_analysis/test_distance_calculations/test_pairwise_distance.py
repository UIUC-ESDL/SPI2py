import numpy as np
import autograd.numpy as anp
from autograd import elementwise_grad
from scipy.spatial.distance import cdist
from SPI2py.analysis.distance import pairwise_distance

def test_pairwise_distance():
    """
    Calculates the pairwise distance between two sets of points.

    We know that Scipy cdist does this correctly so check with it.
    """
    a = np.random.rand(20,3)
    b = np.random.rand(17,3)

    c = pairwise_distance(a, b)

    cd = cdist(a, b).reshape(-1)

    assert all(np.isclose(c,cd))

def test_pairwise_distance_autograd():

   a = anp.random.rand(20, 3)
   b = anp.random.rand(17, 3)

   c = pairwise_distance(a, b)

   cd = cdist(a, b).reshape(-1)

   assert all(anp.isclose(c,cd))

def test_pairwise_distance_grad_autograd():

    a = anp.random.rand(20, 3)
    b = anp.random.rand(17, 3)

    g_c = elementwise_grad(pairwise_distance)

    g_c_val = g_c(a, b)



