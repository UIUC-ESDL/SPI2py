import torch
from openmdao.api import ExplicitComponent

from SPI2py.group_model.component_geometry.spherical_decomposition_methods.finite_sphere_method import read_xyzr_file
from SPI2py.group_model.component_kinematics.distance_calculations import centroid
from SPI2py.group_model.component_kinematics.spatial_transformations import assemble_transformation_matrix, \
    apply_transformation_matrix


class Component(ExplicitComponent):

    def initialize(self):
        self.options.declare('name', types=str)
        self.options.declare('filepath', types=str)
        self.options.declare('translation_dof', types=list)
        self.options.declare('rotation_dof', types=list)
        self.options.declare('ports', types=dict)
        self.options.declare('color', types=str)

    def setup(self):

        self.add_input('translation', val=torch.zeros((1, 3)), shape=(1, 3))
        self.add_input('rotation', val=torch.zeros((1, 3)), shape=(1, 3))
        self.add_output('sphere_positions', val=positions)
        self.add_output('sphere_radii', val=radii)
        self.add_output('port_positions', val=port_positions)

    # def setup_partials(self):
    #     d_sphere_positions/d_translation
    #     d_sphere_positions/d_rotation
    #     d_radii/d_translation (0)
    #     d_radii/d_rotation (0)
    #     d_port_positions/d_translation
    #     d_port_positions/d_rotation

    def compute(self, inputs, outputs):

        # Get the input variables
        translation = inputs['translation']
        rotation = inputs['rotation']

        # Convert the input variables to torch tensors
        translation = torch.tensor(translation, dtype=torch.float64, requires_grad=True)
        rotation = torch.tensor(rotation, dtype=torch.float64, requires_grad=True)

        # Calculate the transformed positions and radii
        sphere_positions_transformed, sphere_radii_transformed = self.compute(translation, rotation)
        port_positions_transformed = self.compute(translation, rotation)

        # Convert to numpy
        sphere_positions_transformed = sphere_positions_transformed.detach().numpy()
        sphere_radii_transformed = sphere_radii_transformed.detach().numpy()
        port_positions_transformed = port_positions_transformed.detach().numpy()

        # Set the outputs
        outputs['sphere_positions'] = sphere_positions_transformed
        outputs['sphere_radii'] = sphere_radii_transformed
        outputs['port_positions'] = port_positions_transformed

    # def compute_partials(self, inputs, partials):
    #     d_sphere_positions/d_translation
    #     d_sphere_positions/d_rotation
    #     d_radii/d_translation (0)
    #     d_radii/d_rotation (0)
    #     d_port_positions/d_translation
    #     d_port_positions/d_rotation

    def


class Component:
    def __init__(self, name, filepath, ports, color):
        self.name = name
        self.filepath = filepath
        self.ports = ports
        self.color = color

        self.component = Component(self.name, self.filepath, self.ports, self.color)

        self.positions, self.radii = read_xyzr_file(self.filepath)

        # Initialize the ports
        self.port_indices = {}
        if self.ports is not None:
            for port in self.ports:
                self.port_indices[port['name']] = len(self.positions - 1)
                self.positions = torch.vstack((self.positions, torch.tensor(port['origin'], dtype=torch.float64)))
                self.radii = torch.cat((self.radii, torch.tensor([port['radius']], dtype=torch.float64)))

        self.design_vector_indices = {dof_i: i for i, dof_i in enumerate(self.dof)}
        self.design_vector_size = len(self.design_vector_indices)

        # TODO Indices?
        self.translation_default = torch.zeros((1, 3), dtype=torch.float64)
        self.rotation_default = torch.zeros((1, 3), dtype=torch.float64)

    def read_input_file(self):

        with open(self.input_file, mode="rb") as fp:
            inputs = tomli.load(fp)

        return inputs

    def centroid(self):
        return centroid(self.positions, self.radii)

    def set_default_positions(self, translation, rotation):
        """
        Calculates the positions of the object's spheres.
        """

        t = assemble_transformation_matrix(translation, rotation)

        new_positions = apply_transformation_matrix(self.centroid.reshape(-1, 1),
                                                    self.positions.T,
                                                    t).T

        self.positions = new_positions

    def compute(self, translation, rotation):
        """
        Calculates the positions of the object's spheres.
        """

        t = assemble_transformation_matrix(translation, rotation)

        new_positions = apply_transformation_matrix(self.centroid.reshape(-1, 1),
                                                    self.positions.T,
                                                    t).T

        object_dict = {self.__repr__(): {'positions': new_positions,
                                         'radii': self.radii,
                                         'port_indices': self.port_indices}}

        return object_dict

    def set_positions(self, objects_dict: dict):
        """
        Update positions of object spheres given a design vector

        :param objects_dict:
        :param design_vector:
        :return:
        """

        self.positions = objects_dict[self.__repr__()]['positions']
        self.radii = objects_dict[self.__repr__()]['radii']

    def get_design_vars(self):
        # TODO Configure
        translation = {'translation': {'ref': 1, 'ref0': 0}}
        rotation = {'rotation': {'ref': 2*torch.pi, 'ref0': 0}}
        design_vars = {**translation, **rotation}
        return design_vars
