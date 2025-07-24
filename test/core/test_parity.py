# File: test_parity.py
# Description: A formal unit test to verify that the DualStateRNN can successfully
#              learn the Temporal Parity Task.

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import random
import numpy as np
from typing import List, Tuple


# ==============================================================================
# SECTION 1: All Necessary Model and Component Code
# (Putting everything in one file for a self-contained, portable test)
# ==============================================================================

class SFU(nn.Module):
    def __init__(self, num_features: int, theta_init: float = 0.0, gamma_init: float = 1.0):
        super(SFU, self).__init__()
        self.theta = nn.Parameter(torch.full((1, num_features), theta_init))
        self.gamma = nn.Parameter(torch.full((1, num_features), gamma_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gamma * (x - self.theta))


class SurrogateSpike(Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i);
        return (i > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        return grad_output * (1 / (1 + 10 * torch.abs(i))).pow(2)


spike_fn = SurrogateSpike.apply


class DualStateLIFLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_in_v = nn.Linear(input_size, hidden_size)
        self.leak_tau_v = nn.Parameter(torch.randn(hidden_size))
        self.flip_threshold = nn.Parameter(torch.full((hidden_size,), 0.5))
        self.fc_out = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.output_activation = nn.Tanh()

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        V_prev, D_prev = (s.detach() for s in state)
        leak_alpha = torch.exp(-F.softplus(self.leak_tau_v))
        input_current = self.linear_in_v(x_t)
        V_t = leak_alpha * V_prev + input_current
        spike = spike_fn(V_t - self.flip_threshold)
        D_t = D_prev * (1 - spike) + (1 - D_prev) * spike
        combined_state = torch.cat([V_t, D_t], dim=1)
        output = self.output_activation(self.fc_out(combined_state))
        return output, (V_t, D_t)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = self.flip_threshold.dtype
        V0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        D0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return (V0, D0)


class DualStateRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super().__init__()
        # For a fair test, we can stack the layers.
        self.rnn_cells = nn.ModuleList()
        layer_input_size = input_size
        for _ in range(num_layers):
            self.rnn_cells.append(DualStateLIFLayer(layer_input_size, hidden_size))
            layer_input_size = hidden_size

        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        device = x.device

        states = [cell.init_state(batch_size, device) for cell in self.rnn_cells]

        for t in range(sequence_length):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.rnn_cells):
                x_t, new_state = cell(x_t, states[i])
                states[i] = new_state

        return self.fc_out(x_t)


def set_seed(seed: int):
    """Sets a random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ==============================================================================
# SECTION 2: The Parity Task Unit Test
# ==============================================================================

class TestParityTask(unittest.TestCase):

    def test_dual_state_rnn_learns_parity(self):
        """
        Tests if the DualStateRNN can achieve high accuracy on the Temporal Parity Task.
        """
        # --- 1. Test Configuration ---
        print("\n--- Running Parity Task Verification Test ---")
        set_seed(42)

        INPUT_SIZE = 1
        HIDDEN_SIZE = 32
        NUM_LAYERS = 2
        NUM_CLASSES = 2
        SEQ_LEN = 50
        BATCH_SIZE = 256
        EPOCHS = 50
        LR = 1e-3
        ACCURACY_THRESHOLD = 0.95  # The model must achieve at least 95% accuracy

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # --- 2. Data Generation ---
        num_samples = BATCH_SIZE * 20
        # Input: Sequence of 0s and 1s for clarity
        X_data = (torch.rand(num_samples, SEQ_LEN, INPUT_SIZE) > 0.5).float()
        # Target: 1 if the number of 1s is odd, 0 if even.
        y_data = (torch.sum(X_data, dim=(1, 2)) % 2 == 1).long()

        dataset = torch.utils.data.TensorDataset(X_data, y_data)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # --- 3. Model Initialization ---
        model = DualStateRNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=NUM_CLASSES,
            num_layers=NUM_LAYERS
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        # --- 4. Training Loop ---
        print("Training model...")
        for epoch in range(EPOCHS):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")

        # --- 5. Final Evaluation and Assertion ---
        print("Evaluating final model on validation set...")
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())

        final_accuracy = np.mean(np.array(all_preds) == np.array(all_true))

        print(f"Final Validation Accuracy: {final_accuracy:.2%}")

        # This is the formal test assertion
        self.assertGreaterEqual(
            final_accuracy,
            ACCURACY_THRESHOLD,
            f"Model failed to learn the Parity Task. "
            f"Final accuracy was {final_accuracy:.2%}, "
            f"which is below the required threshold of {ACCURACY_THRESHOLD:.2%}."
        )
        print("\n✅ SUCCESS: The DualStateRNN successfully learned the Temporal Parity Task.")


if __name__ == '__main__':
    unittest.main()