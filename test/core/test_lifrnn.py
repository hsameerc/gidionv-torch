import unittest

import torch
import torch.nn as nn

from src.lib.core.lif_rnn import DualStateRNN


class TestLIFRnn(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for all tests."""
        self.input_size = 16
        self.output_size = 10
        self.hidden_config = [32, 64]
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
            model = DualStateRNN(
                input_size=self.input_size,
                output_size=self.output_size,
                hidden_layers_config=self.hidden_config
            )
            self.assertEqual(len(model.rnn_cells), 2)
            self.assertEqual(model.rnn_cells[0].linear_in_v.in_features, self.input_size)
            self.assertEqual(model.rnn_cells[0].linear_in_v.out_features, self.hidden_config[0])
            self.assertEqual(model.rnn_cells[1].linear_in_v.in_features, self.hidden_config[0])
            self.assertEqual(model.rnn_cells[1].linear_in_v.out_features, self.hidden_config[1])
            self.assertEqual(model.fc_out.out_features, self.output_size)
            print("test_01_initialization: PASS")
        except Exception as e:
            self.fail(f"Model initialization failed with an exception: {e}")

    def test_02_forward_pass_shape(self):
        """Test if the forward pass returns a tensor of the correct shape."""
        model = DualStateRNN(
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
        model = DualStateRNN(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layers_config=self.hidden_config
        ).to(self.device)

        input_tensor = torch.randn(self.batch_size, self.seq_len, self.input_size).to(self.device)
        targets = torch.randint(0, self.output_size, (self.batch_size, self.seq_len)).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        param_to_monitor = model.rnn_cells[0].linear_in_v.bias
        initial_param_val = param_to_monitor.clone().detach()

        optimizer.zero_grad()
        output = model(input_tensor)
        loss = loss_fn(output.view(-1, self.output_size), targets.view(-1))

        self.assertFalse(torch.isnan(loss), "Loss was NaN, indicating a numerical instability.")

        loss.backward()

        self.assertIsNotNone(param_to_monitor.grad, "Gradient for the monitored parameter is None.")

        optimizer.step()

        final_param_val = param_to_monitor.clone().detach()
 
        self.assertFalse(
            torch.equal(initial_param_val, final_param_val),
            "Parameter did not change after one optimization step."
        )
        print(f"test_03_trainability: PASS (Loss: {loss.item():.4f})")

    def test_04_spike_output_mode(self):
        """Test if the `return_spike` flag works correctly."""
        model = DualStateRNN(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layers_config=self.hidden_config
        ).to(self.device)
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.input_size).to(self.device)
        first_layer = model.rnn_cells[0]
        state = first_layer.init_state(self.batch_size, self.device)
        spike_output, _ = first_layer(input_tensor[:, 0, :], state, return_spike=True)
        is_binary = torch.all((spike_output == 0) | (spike_output == 1))
        self.assertTrue(is_binary)
        print("test_04_spike_output_mode: PASS")


if __name__ == '__main__':
    unittest.main()
