import sys
sys.path
sys.path.append('C:/Users/cpgui/PycharmProjects/SPI2Py/')
# print(sys.path)

from src.SPI2Py.utils import shape_generator #import generate_rectangular_prism



def test_origin():
    pos, rad = shape_generator.generate_rectangular_prism([0, 0, 0], [1, 1, 1])

test_origin()

print("done")

# def test_away_from_origin():
#     pos, rad = shape_generator.generate_rectangular_prism([1, 1, 1], [1, 1, 1])
