# File: benchmark_surrogates.py
# Description: A definitive benchmark to test the impact of different surrogate
#              gradient functions on the LIFRnn's ability to solve the
#              Temporal Parity Task.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import time
import numpy as np
import random
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from typing import List, Callable


# ==============================================================================
# SECTION 1: SURROGATE GRADIENT DEFINITIONS
# ==============================================================================

# --- Surrogate 1: The Original Fast Sigmoid Derivative ---
class FastSigmoidSpike(Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i);
        return (i > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        grad_input = grad_output * (1 / (1 + 10 * torch.abs(i))).pow(2)
        return grad_input


fast_sigmoid_spike_fn = FastSigmoidSpike.apply


# --- Surrogate 2: The Clipped Straight-Through Estimator (STE) ---
class STESpike(Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i);
        return (i > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        # Gradient is passed through, but only for inputs close to the threshold
        grad_mask = (torch.abs(i) < 1.0).float()
        return grad_output * grad_mask


ste_spike_fn = STESpike.apply


# --- Surrogate 3: The Standard Sigmoid Derivative ---
class SigmoidSpike(Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i);
        return (i > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        sigmoid_val = torch.sigmoid(i)
        surrogate_grad = sigmoid_val * (1 - sigmoid_val)
        return grad_output * surrogate_grad


sigmoid_spike_fn = SigmoidSpike.apply


# ==============================================================================
# SECTION 2: THE LIF LAYER (MODIFIED TO ACCEPT A SPIKE FUNCTION)
# ==============================================================================

class SFU(nn.Module):  # (Unchanged)
    def __init__(self, num_features: int, **kwargs):
        super().__init__()
        self.theta = nn.Parameter(torch.full((1, num_features), kwargs.get('theta_init', 0.0)))
        self.gamma = nn.Parameter(torch.full((1, num_features), kwargs.get('gamma_init', 1.0)))

    def forward(self, x): return x * torch.sigmoid(self.gamma * (x - self.theta))


class LIFLayer(nn.Module):
    """
    This version is modified to accept a `spike_fn` as an argument,
    allowing us to easily swap out surrogate gradient functions.
    """

    def __init__(self, input_size: int, hidden_size: int, spike_fn: Callable, refractory_steps: int = 3):
        super().__init__()
        self.hidden_size, self.refractory_steps = hidden_size, float(refractory_steps)
        self.spike_fn = spike_fn  # <-- KEY CHANGE

        self.linear_in = nn.Linear(input_size, hidden_size)
        self.sfu = SFU(hidden_size)
        self.reset_factor_param = nn.Parameter(torch.randn(hidden_size))
        self.firing_threshold = nn.Parameter(torch.full((hidden_size,), 0.5))
        self.adaptation_tau_param = nn.Parameter(torch.randn(hidden_size))
        self.leak_tau_param = nn.Parameter(torch.randn(hidden_size))
        self.V_rest = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, state, return_spike=False):
        V_prev, R_prev, B_prev = (s.detach() for s in state)
        is_refractory = (R_prev > 0)
        V_t_integrated = torch.exp(-F.softplus(self.leak_tau_param)) * V_prev + self.linear_in(x_t)
        effective_threshold = self.firing_threshold + B_prev
        S_t = (1 - is_refractory.float()) * self.spike_fn(V_t_integrated - effective_threshold)
        reset_factor = torch.sigmoid(self.reset_factor_param)
        reset_value = reset_factor * self.V_rest + (1 - reset_factor) * V_t_integrated
        V_next = V_t_integrated * (1 - S_t) + reset_value * S_t
        B_next = torch.exp(-F.softplus(self.adaptation_tau_param)) * B_prev + S_t
        R_next = torch.relu(R_prev - 1) + self.refractory_steps * S_t
        y_t_analog = self.sfu(V_t_integrated)
        output = (1 - is_refractory.float()) * (S_t if return_spike else y_t_analog)
        return output, (V_next, R_next, B_next)

    def init_state(self, batch_size, device):
        dtype = self.V_rest.dtype
        return (torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
                torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
                torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype))


class LIFRnn(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers_config: List[int], spike_fn: Callable,
                 **kwargs):
        super().__init__()
        self.lif_layers = nn.ModuleList()
        layer_input_size = input_size
        for hidden_size in hidden_layers_config:
            self.lif_layers.append(LIFLayer(layer_input_size, hidden_size, spike_fn=spike_fn, **kwargs))
            layer_input_size = hidden_size
        self.fc_out = nn.Linear(hidden_layers_config[-1], output_size)

    def forward(self, x: torch.Tensor):
        batch_size, sequence_length, _ = x.shape
        device = x.device
        states = [layer.init_state(batch_size, device) for layer in self.lif_layers]
        x_t = None
        for t in range(sequence_length):
            x_t = x[:, t, :]
            for i, layer in enumerate(self.lif_layers):
                x_t, new_state = layer(x_t, states[i])
                states[i] = new_state
        return self.fc_out(x_t)


# ==============================================================================
# SECTION 3: The Benchmark Framework
# ==============================================================================

class RNNClassifier(nn.Module):  # (Unchanged)
    def __init__(self, rnn_backbone, hidden_size, num_classes):
        super().__init__();
        self.rnn = rnn_backbone;
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x): return self.classifier(self.rnn(x))


class GRUExtractor(nn.Module):  # (Unchanged)
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__();
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x): _, h = self.gru(x); return h[-1]


def set_seed(seed: int):  # (Unchanged)
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False;
    torch.backends.cudnn.deterministic = True


def run_single_trial(model_name, model, device, train_loader, val_loader, epochs, lr):
    print(f"  --- Starting Trial for {model_name} (LR={lr}) ---")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    val_accuracies = []
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())
        epoch_accuracy = accuracy_score(all_true, all_preds)
        val_accuracies.append(epoch_accuracy)
        if (epoch + 1) % (epochs // 4) == 0:
            print(f"    Epoch {epoch + 1}/{epochs}, Val Accuracy: {epoch_accuracy:.2%}")
    return val_accuracies


def run_surrogate_benchmark():
    # --- 1. Configuration ---
    print("--- Setting up Surrogate Gradient Benchmark ---")
    set_seed(42)
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS = 1, 32, 2
    NUM_CLASSES, SEQ_LEN, BATCH_SIZE = 2, 50, 256
    EPOCHS, LR = 20, 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # --- 2. Data ---
    print("Generating Temporal Parity Task data...")
    dataset = torch.utils.data.TensorDataset(
        torch.randint(0, 2, (BATCH_SIZE * 40, SEQ_LEN, INPUT_SIZE)).float() * 2 - 1,
        (torch.randint(0, 2, (BATCH_SIZE * 40, SEQ_LEN, INPUT_SIZE)).float().sum(dim=(1, 2)) % 2 == 1).long()
    )
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)),
                                                                         len(dataset) - int(0.8 * len(dataset))])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- 3. Models to Benchmark ---
    # GRU Baseline
    gru_backbone = GRUExtractor(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    gru_model = RNNClassifier(gru_backbone, HIDDEN_SIZE, NUM_CLASSES)

    # LIFRnn models with different spike functions
    lif_fast_sigmoid_backbone = LIFRnn(INPUT_SIZE, HIDDEN_SIZE, [HIDDEN_SIZE] * NUM_LAYERS,
                                       spike_fn=fast_sigmoid_spike_fn)
    lif_fast_sigmoid_model = RNNClassifier(lif_fast_sigmoid_backbone, HIDDEN_SIZE, NUM_CLASSES)

    lif_ste_backbone = LIFRnn(INPUT_SIZE, HIDDEN_SIZE, [HIDDEN_SIZE] * NUM_LAYERS, spike_fn=ste_spike_fn)
    lif_ste_model = RNNClassifier(lif_ste_backbone, HIDDEN_SIZE, NUM_CLASSES)

    models_to_test = {
        "GRU (Baseline)": gru_model,
        "LIFRnn (FastSigmoid)": lif_fast_sigmoid_model,
        "LIFRnn (STE)": lif_ste_model
    }

    # --- 4. Run Trials ---
    all_learning_curves = {}
    final_accuracies = {}
    print("\n--- Starting Benchmark Trials ---")
    for name, model in models_to_test.items():
        set_seed(42)
        learning_curve = run_single_trial(name, model, device, train_loader, val_loader, EPOCHS, LR)
        all_learning_curves[name] = learning_curve
        final_accuracies[name] = learning_curve[-1]

    # --- 5. Print Report & Plot ---
    print("\n\n" + "=" * 30 + " SURROGATE GRADIENT BENCHMARK SUMMARY " + "=" * 30)
    for name, acc in final_accuracies.items():
        print(f"  {name:<25} -> Final Validation Accuracy: {acc:.2%}")
    print("=" * 78)

    plt.figure(figsize=(12, 7))
    for name, curve in all_learning_curves.items():
        plt.plot(range(1, EPOCHS + 1), curve, marker='o', linestyle='-', label=f"{name} (Final Acc: {curve[-1]:.2%})")

    plt.title("Performance of Different Surrogate Gradients on Parity Task", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.ylim(bottom=0.45, top=1.02)
    plt.show()


if __name__ == '__main__':
    run_surrogate_benchmark()