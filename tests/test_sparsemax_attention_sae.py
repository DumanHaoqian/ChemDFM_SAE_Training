import tempfile
import unittest
from pathlib import Path

import torch

from sparsemax_attention_sae import SparsemaxAttentionSAE, SparsemaxAttentionSAEConfig


class SparsemaxAttentionSAETest(unittest.TestCase):
    def test_cpu_save_load_round_trip_legacy_compatible(self):
        cfg = SparsemaxAttentionSAEConfig(d_in=4, d_sae=8, device="cpu", dtype="float32", key_dim=4)
        model = SparsemaxAttentionSAE(cfg)
        x = torch.randn(3, 4)
        before, acts_before = model(x)

        with tempfile.TemporaryDirectory() as td:
            weights_path, cfg_path = model.save_inference_model(td)
            self.assertTrue(Path(weights_path).exists())
            self.assertTrue(Path(cfg_path).exists())
            loaded = SparsemaxAttentionSAE.load_from_disk(td, device="cpu")

        after, acts_after = loaded(x)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))
        self.assertTrue(torch.allclose(acts_before, acts_after, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
