from SPI2Py.utils.shape_generator import generate_rectangular_prism


def test_origin():
    pos, rad = generate_rectangular_prism([0, 0, 0], [1, 1, 1])


def test_away_from_origin():
    pos, rad = generate_rectangular_prism([1, 1, 1], [1, 1, 1])
