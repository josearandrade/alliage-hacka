"""Verify cached final-block training matches full-image inference."""

import unittest

import torch
from transformers import Dinov2Config, Dinov2Model

from adapted_model import AdaptedHead
from pipeline import IMAGE_SIZE


class AdaptedModelTests(unittest.TestCase):
    def test_cached_and_online_tokens_have_identical_logits(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        backbone = Dinov2Model(Dinov2Config(hidden_size=24, num_hidden_layers=2, num_attention_heads=3)).eval()
        network = AdaptedHead(backbone).eval()
        pixels = torch.randn(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
        with torch.no_grad():
            output = backbone(pixel_values=pixels, output_hidden_states=True)
            cached = network(output.hidden_states[-2])
            online = network.from_final_tokens(output.last_hidden_state)
        torch.testing.assert_close(cached, online)
        loss = network(output.hidden_states[-2].detach()).square().sum()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None))
        self.assertTrue(any(p.grad is not None for p in network.block.parameters()))


if __name__ == "__main__":
    unittest.main()
