import yaml
from utils.shape_generator import generate_rectangular_prism, generate_rectangular_prisms
from utils.visualization import plot, plot_sphere
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def spherical_decomposition(self, dimensions, diameters, origins):
    """
    Generates a set of spheres packaged into a given volume

    For now, we are only using rectangular prisms

    :param dimensions:
    :param diameters:
    :param origins:
    :return:
    """

    positions, radii = generate_rectangular_prisms(dimensions, diameters, origins)

    return positions, radii

def generate_layout(inputs):

    pass



# dimensions = [1,1,1]
# diameter = 0.5
# position = [0,0,0]
#
# pos, rad = generate_rectangular_prism(dimensions,diameter, position)
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# x = pos[:,0]
# y = pos[:,1]
# z = pos[:,2]
#
# ax.scatter(x,y,z, marker='.', s=100)
#
# plot_dict = {'object1':{'positions':pos,
#                         'radius': rad,
#                         'color': 'r'}
#              }
#
# plot(plot_dict)
#
#
# dimensions = [[1,1,1]]
# diameters = [0.5]
# positions = [[0,0,0]]
#
# pos, rad = generate_rectangular_prisms(dimensions,diameters, positions)
