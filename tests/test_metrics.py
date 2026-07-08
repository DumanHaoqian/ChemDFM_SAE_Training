import unittest

import torch

from metrics import FVUAccumulator


class FVUAccumulatorTest(unittest.TestCase):
    def test_uses_global_feature_mean_across_batches(self):
        x1 = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        r1 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        x2 = torch.tensor([[4.0, 0.0], [6.0, 0.0]])
        r2 = torch.tensor([[4.0, 0.0], [5.0, 0.0]])

        acc = FVUAccumulator()
        acc.update(x1, r1)
        acc.update(x2, r2)

        x = torch.cat([x1, x2], dim=0)
        r = torch.cat([r1, r2], dim=0)
        expected = (r - x).pow(2).sum() / (x - x.mean(dim=0, keepdim=True)).pow(2).sum()
        self.assertAlmostEqual(acc.fvu(), float(expected.item()), places=7)


if __name__ == "__main__":
    unittest.main()
