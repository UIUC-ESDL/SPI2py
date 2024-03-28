import unittest
import torch


def overlap_volume_sphere_sphere(r_1, r_2, d):

    # Calculate the volume of two spheres
    volume_1 = (4 / 3) * torch.pi * r_1 ** 3
    volume_2 = (4 / 3) * torch.pi * r_2 ** 3
    smaller_volume = torch.minimum(volume_1, volume_2)

    # If the distance between sphere centers is zero, use the volume of the smaller sphere
    if d == 0:
        overlap_volume = smaller_volume

    # Analytic solution for the volume of the intersection of two spheres does not apply when d > r_1 + r_2
    elif d >= r_1 + r_2:
        overlap_volume = torch.tensor([0.0])

    # Otherwise, calculate the volume of the intersection
    else:
        numerator = torch.pi * (r_1 + r_2 - d) ** 2 * (
                    d ** 2 + 2 * d * r_2 - 3 * r_2 ** 2 + 2 * d * r_1 + 6 * r_2 * r_1 - 3 * r_1 ** 2)
        denominator = 12 * d
        overlap_volume = numerator / denominator

    # Perform Sanity checks
    # assert overlap_volume >= torch.tensor(0.0)
    # assert torch.round(overlap_volume, decimals=1) <= torch.round(smaller_volume, decimals=1)
    # assert overlap_volume != torch.nan

    return overlap_volume

def overlap_volume_spheres_spheres(R_1, R_2, D):

    overlap_volume = torch.tensor([0.0])

    for r_1, r_2, d in zip(R_1, R_2, D):
        overlap_volume += overlap_volume_sphere_sphere(r_1, r_2, d)

    return overlap_volume

# def overlap_volume_sphere_set_of_spheres(r_1, R_2, D):


class TestOverlapVolume(unittest.TestCase):
    def test_perfect_overlap(self):
        r_1 = torch.tensor([5.0])
        r_2 = torch.tensor([5.0])
        d = torch.tensor([0.0])
        expected = (4 / 3) * torch.pi * r_1 ** 3
        result = overlap_volume_sphere_sphere(r_1, r_2, d)
        self.assertTrue(torch.allclose(result, expected), "Perfect overlap test failed")

    def test_no_overlap(self):
        r_1 = torch.tensor([5.0])
        r_2 = torch.tensor([5.0])
        d = torch.tensor([10.0])
        expected = torch.tensor([0.0])
        result = overlap_volume_sphere_sphere(r_1, r_2, d)
        self.assertTrue(torch.allclose(result, expected), "No overlap test failed")

    def test_no_overlap_2(self):
        r_1 = torch.tensor([5.0])
        r_2 = torch.tensor([5.0])
        d = torch.tensor([12.0])
        expected = torch.tensor([0.0])
        result = overlap_volume_sphere_sphere(r_1, r_2, d)
        self.assertTrue(torch.allclose(result, expected), "No overlap test failed")

    def test_almost_no_overlap(self):
        r_1 = torch.tensor([5.0])
        r_2 = torch.tensor([5.0])
        d = torch.tensor([0.001])  # A small distance should not result in "almost infinity"
        expected = torch.tensor([523.6])  # Per http://ambrnet.com/TrigoCalc/Sphere/TwoSpheres/Intersection.htm
        result = overlap_volume_sphere_sphere(r_1, r_2, d)
        expected_rounded = torch.round(expected, decimals=1)
        result_rounded = torch.round(result, decimals=1)
        self.assertTrue(torch.allclose(result_rounded, expected_rounded), "No overlap test failed")

    def test_partial_overlap(self):
        r_1 = torch.tensor([5.0])
        r_2 = torch.tensor([5.0])
        d = torch.tensor([5.0])
        expected = torch.tensor(163.62)  # Per http://ambrnet.com/TrigoCalc/Sphere/TwoSpheres/Intersection.htm
        result = overlap_volume_sphere_sphere(r_1, r_2, d)
        result = torch.round(result, decimals=2)
        self.assertTrue(torch.allclose(result, expected), "Partial overlap test failed")


if __name__ == '__main__':

    # Set the default data type
    torch.set_default_dtype(torch.float64)

    # TODO Add test for d > r_1 + r_2

    unittest.main()
