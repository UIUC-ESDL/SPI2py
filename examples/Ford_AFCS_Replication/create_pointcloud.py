
from SPI2py.models.geometry.maximal_disjoint_ball_decomposition import generate_point_cloud

partnames = ['fuel_cell_stack', 'heater_core', 'particle_filter', 'pump','radiator_and_ion_exchanger', 'WEG_heater_and_pump', ]
# generate_point_cloud('components/', 'CAD_Files/pump.stl', 'point_clouds/pump.xyz', meshgrid_increment=25, plot=True)

for partname in partnames:
    generate_point_cloud('components/', f'CAD_Files/{partname}.stl', f'point_clouds/{partname}.xyz', meshgrid_increment=15, plot=True)
