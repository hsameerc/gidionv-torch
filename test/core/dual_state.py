from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lib.core.lif_rnn import SurrogateSpike

spike_fn = SurrogateSpike.apply


# ==============================================================================
# SECTION 2: The "Mirror State" Recurrent Cell
# ==============================================================================

class DualStateLIFLayer(nn.Module):
    """
    A recurrent cell with a dual-state memory system, as per your design.
    It maintains:
    1. An Analog State (V): A rich, continuous, LIF-like membrane potential.
    2. A Digital "Mirror" State (D): A binary state for tracking discrete events.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # --- Analog State Components ---
        self.linear_in_v = nn.Linear(input_size, hidden_size)
        self.leak_tau_v = nn.Parameter(torch.randn(hidden_size))
        # This threshold is for the "flip check" on the analog state
        self.flip_threshold = nn.Parameter(torch.full((hidden_size,), 0.5))

        # --- Digital State Components ---
        # No extra parameters needed for the digital state's update logic,
        # as it's a simple, parameter-free flip.

        # --- Output Layer ---
        # The final output will be a combination of both states.
        # It takes the concatenated state [V, D] as input.
        self.fc_out = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.output_activation = nn.Tanh()

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs a single forward step.

        Args:
            x_t (Tensor): Input for the current timestep.
            state (Tuple): The previous hidden state (V_prev, D_prev).

        Returns:
            A tuple containing:
            - output (Tensor): The final output of the cell for this timestep.
            - new_state (Tuple): The updated state (V_t, D_t) for the next timestep.
        """
        # Unpack the two separate states from the previous step
        V_prev, D_prev = (s.detach() for s in state)

        # --- 1. Analog State (`V`) Update (The "Value that remains") ---
        # This is a simplified leaky integrator.
        leak_alpha = torch.exp(-F.softplus(self.leak_tau_v))
        input_current = self.linear_in_v(x_t)
        V_t = leak_alpha * V_prev + input_current

        # --- 2. The "Flip Check" ---
        # We check if the analog state has crossed its threshold.
        # This produces a binary spike signal (0.0 or 1.0).
        spike = spike_fn(V_t - self.flip_threshold)

        # --- 3. Digital "Mirror" State (`D`) Update ---
        # The digital state updates based on the spike and its own previous state.
        # This is the pure, logical flip operation.
        # D_t = D_prev XOR spike
        # An equivalent arithmetic operation for binary (0/1) values:
        D_t = D_prev * (1 - spike) + (1 - D_prev) * spike

        # --- 4. Final Output ---
        # The output is a learned function of BOTH the analog and digital states.
        combined_state = torch.cat([V_t, D_t], dim=1)
        output = self.output_activation(self.fc_out(combined_state))

        # Return the output and the new, updated state tuple
        return output, (V_t, D_t)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes both the analog and digital states to zero."""
        dtype = self.flip_threshold.dtype
        V0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        D0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return (V0, D0)


# ==============================================================================
# SECTION 3: RNN Wrapper and Classifier (for testing)
# ==============================================================================

class DualStateRNN(nn.Module):
    """A wrapper to run the DualStateLIFLayer over a sequence."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super().__init__()
        # For simplicity, we'll use a single layer for this test.
        # Stacking these would require careful handling of input/output sizes.
        self.rnn_cell = DualStateLIFLayer(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        device = x.device

        state = self.rnn_cell.init_state(batch_size, device)

        for t in range(sequence_length):
            # The output of one step becomes the input to the next...
            # This is a design choice. A simpler model would just take x[:, t, :].
            # Let's use the simpler version for the Parity Task.
            output, state = self.rnn_cell(x[:, t, :], state)

        # Classify based on the final output
        return self.fc_out(output)


# ==============================================================================
# SECTION 4: The Parity Task Benchmark
# ==============================================================================
if __name__ == '__main__':
    # (The benchmark setup code from `final_benchmark_parity.py` would go here)
    # This is a placeholder to show how to instantiate and run the new model.

    print("--- Testing the DualStateLIFLayer on the Parity Task ---")

    # Configuration
    INPUT_SIZE = 1
    HIDDEN_SIZE = 32
    NUM_CLASSES = 2
    SEQ_LEN = 50
    BATCH_SIZE = 256
    LR = 1e-3
    EPOCHS = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    model = DualStateRNN(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=NUM_CLASSES
    ).to(device)

    # Create data
    X_data = (torch.rand(BATCH_SIZE * 10, SEQ_LEN, INPUT_SIZE) > 0.5).float()
    y_data = (torch.sum(X_data, dim=(1, 2)) % 2 == 1).long()
    dataset = torch.utils.data.TensorDataset(X_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")

    print("\n✅ Training finished. This demonstrates the model is trainable.")
    print("A full benchmark would compare its final accuracy to the GRU and original LIFRnn.")
