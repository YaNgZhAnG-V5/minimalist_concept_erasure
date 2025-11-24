import unittest

import torch

from diffsolver.utils import hard_concrete_distribution


class TestHardConcreteDistribution(unittest.TestCase):
    def test_reproducibility(self):
        shapes = [
            (3,),  # 1D tensor
            (2, 3),  # 2D tensor
            (4, 2, 3),  # 3D tensor
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                for _ in range(10):  # Run the test 10 times
                    p = torch.rand(shape, device="cpu", dtype=torch.float16)

                    torch.manual_seed(42)
                    s1 = hard_concrete_distribution(p)
                    s2 = hard_concrete_distribution(p)

                    torch.manual_seed(42)
                    s3 = hard_concrete_distribution(p)
                    s4 = hard_concrete_distribution(p)

                    self.assertTrue(
                        torch.allclose(s1, s3), f"Outputs are not reproducible with the same seed for shape {shape}"
                    )

                    self.assertTrue(
                        torch.allclose(s2, s4), f"Outputs are not reproducible with the same seed for shape {shape}"
                    )


if __name__ == "__main__":
    unittest.main()
