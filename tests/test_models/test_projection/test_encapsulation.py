import torch
import unittest
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