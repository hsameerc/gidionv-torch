import unittest

import torch
import torch.nn as nn

from src.lib.core.lif_rnn import LIFRnn


class TestLIFRnn(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for all tests."""
        self.input_size = 16
        self.output_size = 10
        self.hidden_config = [32, 64]  # Two hidden layers
        self.batch_size = 4
        self.seq_len = 20
        # Check for available hardware
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"\n--- Running test on device: {self.device} ---")

    def test_01_initialization(self):
        """Test if the model initializes without errors."""
        try:
            model = LIFRnn(
                input_size=self.input_size,
                output_size=self.output_size,
                hidden_layers_config=self.hidden_config
            )
            # Check number of layers
            self.assertEqual(len(model.lif_layers), 2)
            # Check layer dimensions
            self.assertEqual(model.lif_layers[0].linear_in.in_features, self.input_size)
            self.assertEqual(model.lif_layers[0].linear_in.out_features, self.hidden_config[0])
            self.assertEqual(model.lif_layers[1].linear_in.in_features, self.hidden_config[0])
            self.assertEqual(model.lif_layers[1].linear_in.out_features, self.hidden_config[1])
            self.assertEqual(model.fc_out.out_features, self.output_size)
            print("test_01_initialization: PASS")
        except Exception as e:
            self.fail(f"Model initialization failed with an exception: {e}")

    def test_02_forward_pass_shape(self):
        """Test if the forward pass returns a tensor of the correct shape."""
        model = LIFRnn(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layers_config=self.hidden_config
        ).to(self.device)

        input_tensor = torch.randn(self.batch_size, self.seq_len, self.input_size).to(self.device)
        output = model(input_tensor)

        expected_shape = (self.batch_size, self.seq_len, self.output_size)
        self.assertEqual(output.shape, expected_shape)
        print("test_02_forward_pass_shape: PASS")

    def test_03_trainability(self):
        """
        Test if the model is trainable by checking if a core parameter
        is updated after one optimization step.
        """
        model = LIFRnn(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layers_config=self.hidden_config
        ).to(self.device)

        input_tensor = torch.randn(self.batch_size, self.seq_len, self.input_size).to(self.device)
        targets = torch.randint(0, self.output_size, (self.batch_size, self.seq_len)).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # We monitor the bias of the first linear layer. This parameter is guaranteed
        # to be used in every forward pass.
        param_to_monitor = model.lif_layers[0].linear_in.bias
        initial_param_val = param_to_monitor.clone().detach()

        # Perform a single training step
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = loss_fn(output.view(-1, self.output_size), targets.view(-1))

        # Check that the loss is a valid number before backprop
        self.assertFalse(torch.isnan(loss), "Loss was NaN, indicating a numerical instability.")

        loss.backward()

        # Optional: Check if the gradient for the monitored parameter exists
        self.assertIsNotNone(param_to_monitor.grad, "Gradient for the monitored parameter is None.")

        optimizer.step()

        final_param_val = param_to_monitor.clone().detach()

        # The core assertion: the parameter must have changed
        self.assertFalse(
            torch.equal(initial_param_val, final_param_val),
            "Parameter did not change after one optimization step."
        )
        print(f"test_03_trainability: PASS (Loss: {loss.item():.4f})")

    def test_04_spike_output_mode(self):
        """Test if the `return_spike` flag works correctly."""
        model = LIFRnn(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layers_config=self.hidden_config
        ).to(self.device)

        input_tensor = torch.randn(self.batch_size, self.seq_len, self.input_size).to(self.device)

        # We need to call the internal layers to check their spike output
        # Get the output of the first hidden layer in spike mode
        first_layer = model.lif_layers[0]
        state = first_layer.init_state(self.batch_size, self.device)
        spike_output, _ = first_layer(input_tensor[:, 0, :], state, return_spike=True)

        # Check if the output is binary (contains only 0s and 1s)
        is_binary = torch.all((spike_output == 0) | (spike_output == 1))
        self.assertTrue(is_binary)
        print("test_04_spike_output_mode: PASS")


if __name__ == '__main__':
    unittest.main()
