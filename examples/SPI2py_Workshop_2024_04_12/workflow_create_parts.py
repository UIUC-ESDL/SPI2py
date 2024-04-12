from SPI2py.models.geometry.spherical_decomposition import pseudo_mdbd

directory = 'models/'
filename_bot_eye = 'Bot_Eye.stl'
filename_cog_driven_gear = 'CogDrivenGear.stl'
filename_cross_head_pin = 'CrossHead_Pin.stl'

pseudo_mdbd(directory, filename_bot_eye, 'Bot_Eye.xyzr', num_spheres=100, min_radius=0.01, meshgrid_increment=20, scale=0.1, plot=True)
pseudo_mdbd(directory, filename_cog_driven_gear, 'CogDrivenGear.xyzr', num_spheres=100, min_radius=0.01, meshgrid_increment=20, scale=0.1,  plot=True)
pseudo_mdbd(directory, filename_cross_head_pin, 'CrossHead_Pin.xyzr', num_spheres=100, min_radius=0.01, meshgrid_increment=20, scale=0.1,  plot=True)