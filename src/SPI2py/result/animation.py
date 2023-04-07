import os
import tempfile

from imageio import v2 as imageio


def generate_gif(layout, design_vector_log, frames_per_figure, gif_directory, gif_filename):
    """

    :param layout:
    :param design_vector_log:
    :param frames_per_figure:
    :param GIFfilepath:
    :return:
    """
    def plot_all(design_vectors):
        """

        :param design_vectors:
        :return:
        """
        i = 1
        for xk in design_vectors:
            filepath = tempDir+'/'+str(i)
            pos_dict = layout.calculate_positions(xk)
            layout.set_positions(pos_dict)

            layout.plot(savefig=True, directory=filepath)

            i += 1

    # Create a temporary directory to save to plotted figures
    temp = tempfile.TemporaryDirectory()
    tempDir = temp.name

    plot_all(design_vector_log)

    files = os.listdir(tempDir)

    # Sort files based on numerical order (i.e., 1,2,... 11,12 not 1,11,12,2,...)
    order = [int(file[0:-4]) for file in files]
    files = [file for _, file in sorted(zip(order, files))]

    filenames = [tempDir + '/' + filename for filename in files]

    # Generate the GIF
    images = []
    for filename in filenames:
        for _ in range(frames_per_figure):
            images.append(imageio.imread(filename))

    # TODO Add config file input for GIF name
    output_filepath = gif_directory + gif_filename

    imageio.mimsave(output_filepath, images)
