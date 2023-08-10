import jax.numpy as np



def bounding_box(positions_dict):

    # TODO include radii
    # TODO verify calc

    positions_array = np.vstack([np.vstack(positions_dict[key]['positions']) for key in positions_dict.keys()])

    min_x = np.min(positions_array[:, 0])
    max_x = np.max(positions_array[:, 0])
    min_y = np.min(positions_array[:, 1])
    max_y = np.max(positions_array[:, 1])
    min_z = np.min(positions_array[:, 2])
    max_z = np.max(positions_array[:, 2])

    volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)

    return volume
