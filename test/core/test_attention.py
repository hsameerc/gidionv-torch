import unittest

import torch

from src.lib.core.attention import MultiHeadAttention


class TestAttentionManual(unittest.TestCase):
    def setUp(self):
        """Set up a stateful PyTorch AttentionManual module for each test."""
        self.batch_size = 2
        self.seq_len_q = 4
        self.seq_len_kv = 4
        self.d_model = 16
        self.num_heads = 4
        self.test_dtype = torch.float32

        torch.manual_seed(42)

        self.mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads, dropout_rate=0.0,
                                      dtype=self.test_dtype)

        # Dummy torch tensors
        self.dummy_q_input = torch.randn(self.batch_size, self.seq_len_q, self.d_model, dtype=self.test_dtype)
        self.dummy_k_input = torch.randn(self.batch_size, self.seq_len_kv, self.d_model, dtype=self.test_dtype)
        self.dummy_v_input = torch.randn(self.batch_size, self.seq_len_kv, self.d_model, dtype=self.test_dtype)

    def test_output_shape(self):
        """Tests if the forward pass produces the correct output shapes."""
        self.mha.eval()
        output, attn_weights, _ = self.mha(self.dummy_q_input, self.dummy_k_input, self.dummy_v_input)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len_q, self.d_model))
        self.assertEqual(attn_weights.shape, (self.batch_size, self.num_heads, self.seq_len_q, self.seq_len_kv))
        print("✅ Output shape test passed!")

    def test_causal_mask(self):
        """Tests if the causal mask correctly zeros out future attention weights."""
        self.mha.eval()
        _, attn_weights, _ = self.mha(self.dummy_q_input, self.dummy_k_input, self.dummy_v_input, is_causal=True)
        for i in range(self.seq_len_q):
            for j in range(i + 1, self.seq_len_kv):
                # Check that all weights for future positions are zero
                self.assertTrue(torch.all(attn_weights[:, :, i, j] == 0.0))
        print("✅ Causal mask test passed!")

    def test_external_padding_mask(self):
        """Tests if the padding mask correctly zeros out attention to padded keys."""
        self.mha.eval()
        mask = torch.ones(self.batch_size, self.seq_len_kv, dtype=torch.float32)
        mask[:, -1] = 0
        _, attn_weights, _ = self.mha(self.dummy_q_input, self.dummy_k_input, self.dummy_v_input, attn_mask=mask)
        # Check that all attention weights to the last key position are zero
        self.assertTrue(torch.all(attn_weights[:, :, :, -1] == 0.0))
        print("✅ External padding mask test passed!")

    def test_dropout_in_eval_vs_train_mode(self):
        """Tests if dropout is active in train mode and inactive in eval mode."""
        torch.manual_seed(123)
        mha_dropout = MultiHeadAttention(self.d_model, self.num_heads, dropout_rate=0.9, dtype=self.test_dtype)

        # Train mode should be stochastic
        mha_dropout.train()
        out_train1, _, _ = mha_dropout(self.dummy_q_input, self.dummy_k_input, self.dummy_v_input)
        out_train2, _, _ = mha_dropout(self.dummy_q_input, self.dummy_k_input, self.dummy_v_input)
        self.assertFalse(torch.allclose(out_train1, out_train2))

        # Eval mode should be deterministic
        mha_dropout.eval()
        out_eval1, _, _ = mha_dropout(self.dummy_q_input, self.dummy_k_input, self.dummy_v_input)
        out_eval2, _, _ = mha_dropout(self.dummy_q_input, self.dummy_k_input, self.dummy_v_input)
        self.assertTrue(torch.allclose(out_eval1, out_eval2))
        print("✅ Dropout mode test passed!")

    def test_backward_pass_smoke_test(self):
        """
        A 'smoke test' to ensure gradients are computed for all parameters
        and have the correct shapes.
        """
        self.mha.train()

        # Forward pass
        output, _, _ = self.mha(self.dummy_q_input, self.dummy_k_input, self.dummy_v_input)

        # Create a dummy loss and call backward
        loss = output.sum()
        loss.backward()

        print("\n Testing MHA Gradients")
        for name, param in self.mha.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient for '{name}' should not be None.")
            self.assertEqual(param.grad.shape, param.shape, f"Shape mismatch for gradient of '{name}'.")
            self.assertFalse(torch.all(param.grad == 0), f"Gradient for '{name}' is all zeros.")
            print(f"✅ Gradient for '{name}' is valid.")

    def test_numerical_gradient_with_grad_check(self):
        """
        Uses PyTorch's built-in grad check utility to verify the correctness of the
        manual attention implementation.

        Note: grad check requires double precision (float64) and works on functions
        that take tensors as input and return a single tensor.
        """
        print("\nTesting Numerical Gradients with grad check")
        q = self.dummy_q_input.double().requires_grad_()
        k = self.dummy_k_input.double().requires_grad_()
        v = self.dummy_v_input.double().requires_grad_()
        mha_double = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads, dropout_rate=0.0,
                                        dtype=torch.float64)
        def mha_func(query, key, value):
            output, _, _ = mha_double(query, key, value)
            return output
        is_correct = torch.autograd.gradcheck(mha_func, (q, k, v), eps=1e-6, atol=1e-4)
        self.assertTrue(is_correct, "Numerical gradient check failed!")
        print("✅ Numerical gradient check passed!")


if __name__ == '__main__':
    unittest.main()
