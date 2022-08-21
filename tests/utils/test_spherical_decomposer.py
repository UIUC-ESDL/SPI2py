from utils.spherical_decomposer import generate_rectangular_prism


def test_no_origin():
    pos, rad = generate_rectangular_prism([1, 1, 1], 0.5)


def test_origin()
    pos, rad = generate_rectangular_prism([1, 1, 1], 0.5, [1, 1, 1])
