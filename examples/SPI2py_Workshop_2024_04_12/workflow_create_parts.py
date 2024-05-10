from SPI2py.models.geometry.point_clouds import generate_point_cloud

directory = 'models/'
filename_bot_eye = 'Bot_Eye.stl'
filename_cog_driven_gear = 'CogDrivenGear.stl'
filename_cross_head_pin = 'CrossHead_Pin.stl'

# import pyvista as pv
# cog_driven_gear = pv.read(directory+filename_cog_driven_gear)
# cog_driven_gear.plot()

generate_point_cloud(directory, filename_bot_eye, 'Bot_Eye.xyz', meshgrid_increment=25, plot=True)
generate_point_cloud(directory, filename_cog_driven_gear, 'CogDrivenGear.xyz', meshgrid_increment=25, plot=True)
generate_point_cloud(directory, filename_cross_head_pin, 'CrossHead_Pin.xyz', meshgrid_increment=25, plot=True)
