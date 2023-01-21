class InputValidation:
    def _validate_positions(self, positions):

        if positions is None:
            raise ValueError('Positions have not been set.')

        if not isinstance(positions, list) and not isinstance(positions, np.ndarray):
            raise ValueError('Positions must be a list or numpy array.')

        if isinstance(positions, list):
            warnings.warning('Positions should be a numpy array.')
            positions = np.array(positions)

        if len(positions.shape) == 1:
            warnings.warning('Positions are not 2D.')
            positions = positions.reshape(-1, 3)

        if positions.shape[1] != 3:
            raise ValueError('Positions must be 3D. Where did you get the extra dimension from?')

        return positions
    def _validate_radii(self, radii):

            if radii is None:
                raise ValueError('Radii have not been set.')

            if len(radii.shape[0]) != len(self.positions.shape[0]):
                raise ValueError('Radii must be the same length as positions.')

            return radii

    def _validate_colors(self, color):
        if color is None:
            raise ValueError('Color has not been set.')

        if len(color) == 1:
            raise ValueError('Color must be a 3 element list.')

        if len(color) > 1:

        return color