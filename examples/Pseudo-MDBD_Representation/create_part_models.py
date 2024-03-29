"""

"""
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from SPI2py.models.geometry.maximal_disjoint_ball_decomposition import mdbd
from SPI2py.models.geometry.finite_sphere_method import read_xyzr_file

# filenames = os.listdir(path='Thingi10k_models/')

# num_spheres = 1000
# min_radius = 0.01
# increments = 30

mdbd_unit_cube = pv.Cube(bounds=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5))
mdbd_unit_cube.save('mdbd_unit_cube.stl')

mdbd('', 'mdbd_unit_cube.stl','mdbd_unit_cube.xyzr',
     num_spheres=3000,
     min_radius=0.0001,
     meshgrid_increment=50,
     plot=False)

positions, radii = read_xyzr_file('mdbd_unit_cube.xyzr', num_spheres=1000)





# positions = np.array(positions)
radii = np.array(radii)

percent_accuracy = 0.933

expected_max_volume = 1.0

# Calculate cumulative volume
cumulative_volume = []
total_volume = 0
target_volume = expected_max_volume * percent_accuracy
num_spheres_required = 0
for radius in radii:
    volume = 4.0/3.0*np.pi*radius**3
    total_volume += volume
    cumulative_volume.append(total_volume)
    num_spheres_required += 1  # Increment counter
    # if total_volume >= target_volume:  # Check if cumulative volume has reached target volume
    #     break  # If so, break the loop

# Create scatter plot
plt.scatter(range(len(radii)), cumulative_volume)
plt.axhline(y=expected_max_volume, color='r', linestyle='--')  # Line for expected max volume
plt.xlabel('Number of Spheres')
plt.ylabel('Cumulative Volume')
# Set y-axis from 0 to 1.3
plt.ylim(0, 1.3)
plt.title('Cumulative Volume vs Number of Spheres')
plt.show()



# for filename in filenames:
#     mdbd('', 'cad_models/'+filename, 'mdbds/'+filename+'.xyzr', num_spheres=num_spheres, min_radius=min_radius, meshgrid_increment=increments)
# mdbd('', 'Compressor_reduced.stl', 'Compressor_reduced.xyzr', color='#1873BA', num_spheres=num_spheres, meshgrid_increment=increments)