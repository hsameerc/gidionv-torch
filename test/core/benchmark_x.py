import random
import time
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.autograd import Function


class SFU(nn.Module):
    """
       A Smooth Firing Unit (SFU) activation function.

       This function serves as a continuous and differentiable approximation of a neuron's
       firing rate. It is inspired by the Leaky Integrate-and-Fire model, applying
       a smooth, thresholded gating mechanism to its input.

       The activation is defined by the formula:
       f(z; θ, γ) = z * sigmoid(γ * (z - θ))

       The parameters theta (θ, the firing threshold) and gamma (γ, the sensitivity or
       steepness) are learnable, allowing the network to adapt the activation shape
       for each neuron.

       Args:
           num_features (int): The number of features in the input tensor, corresponding
                               to the number of neurons in the layer.
           theta_init (float, optional): The initial value for the firing threshold θ.
                                         Defaults to 0.0.
           gamma_init (float, optional): The initial value for the sensitivity γ.
                                         Defaults to 1.0.
       """

    def __init__(self, num_features: int, theta_init: float = 0.0, gamma_init: float = 1.0):
        super(SFU, self).__init__()
        self.shape = (1, num_features)
        self.theta = nn.Parameter(torch.full(self.shape, theta_init))
        self.gamma = nn.Parameter(torch.full(self.shape, gamma_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the SFU activation function."""
        return x * torch.sigmoid(self.gamma * (x - self.theta))


class SurrogateSpike(Function):
    """
        A surrogate gradient function for the discrete spiking mechanism.

        In the forward pass, this function behaves like a Heaviside step function,
        producing a binary spike (0 or 1). This is biologically plausible but has a
        gradient that is zero almost everywhere, making it incompatible with
        gradient-based learning.

        In the backward pass, this function substitutes the true gradient with a
        "surrogate" gradient, which is a smooth, bell-shaped curve. This provides
        a useful learning signal to the optimizer, allowing the network to learn
        how to produce desirable spike patterns.
    """

    @staticmethod
    def forward(ctx, i):
        """
            Performs the forward pass (a hard threshold).

            Args:
                   ctx: A context object for saving information for the backward pass.
                   i: The membrane potential minus the firing threshold (V - θ).

               Returns:
                   A binary tensor of spikes.
        """
        ctx.save_for_backward(i)
        return (i > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
            Performs the backward pass (the surrogate gradient).

            Args:
                   ctx: The context object with saved tensors.
                   grad_output: The gradient from the subsequent layer.

            Returns:
                   The surrogate gradient multiplied by the incoming gradient (chain rule).
        """
        i, = ctx.saved_tensors
        grad_input = grad_output * (1 / (1 + 10 * torch.abs(i))).pow(2)
        return grad_input


spike_fn = SurrogateSpike.apply


class LIFLayer(nn.Module):
    """
       A stateful recurrent cell that models a Leaky Integrate-and-Fire (LIF)
       neuron with a high degree of biological plausibility.

       This layer maintains an internal state consisting of the membrane potential (V),
       a refractory counter (R), and a spike adaptation bias (B). It processes one
       timestep of a sequence at a time.

       Key features include:
       - **Leaky Integration:** Membrane potential decays over time with a learnable
         time constant (tau).
       - **Soft Reset:** After firing, the potential is reset towards a learnable
         resting state with a learnable strength.
       - **Spike-Frequency Adaptation:** The firing threshold dynamically increases
         with recent activity, promoting sparse firing.
       - **Absolute Refractory Period:** A neuron is guaranteed to be silent for a
         fixed number of timesteps after firing.
       - **Dual Output Mode:** Can output rich analog signals (via SFU) for standard
         ANN tasks or discrete binary spikes for SNN tasks.

       Args:
           input_size (int): The number of features in the input at each timestep.
           hidden_size (int): The number of LIF neurons in the layer.
           refractory_steps (int, optional): The duration of the absolute refractory
                                            period in timesteps. Defaults to 3.
       """

    def __init__(self, input_size: int, hidden_size: int, refractory_steps: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.refractory_steps = float(refractory_steps)
        # Core layers (created on default device/dtype)
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.sfu = SFU(hidden_size)
        # Learnable parameters for neuron dynamics
        self.reset_factor_param = nn.Parameter(torch.randn(hidden_size))
        self.firing_threshold = nn.Parameter(torch.full((hidden_size,), 0.5))
        self.adaptation_tau_param = nn.Parameter(torch.randn(hidden_size))
        self.leak_tau_param = nn.Parameter(torch.randn(hidden_size))
        self.V_rest = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, state, return_spike=False):
        """
            Performs a single forward step, processing one timestep of a sequence.

            Args:
                   x_t (Tensor): Input tensor for the current timestep, shape (batch_size, input_size).
                   state (Tuple): A tuple containing the hidden state from the previous timestep
                                  (V_prev, R_prev, B_prev).
                   return_spike (bool, optional): If True, the output is a binary spike tensor.
                                                  If False, the output is a continuous analog
                                                  signal from the SFU. Defaults to False.

            Returns:
                   A tuple containing:
                   - output (Tensor): The output of the layer for the current timestep.
                   - new_state (Tuple): The updated hidden state (V_t, R_t, B_t) for the next timestep.
        """
        # Unpacking the state from the previous timestep
        V_prev, R_prev, B_prev = state

        is_refractory = (R_prev > 0)

        # Soft Reset: Blend between previous potential and resting potential
        reset_factor = torch.sigmoid(self.reset_factor_param)
        reset_value = reset_factor * self.V_rest + (1 - reset_factor) * V_prev

        # Calculate which neurons spiked in the previous step for the reset logic
        S_prev = spike_fn(V_prev - (self.firing_threshold + B_prev))

        # Spike-Frequency Adaptation: Effective threshold increases with recent activity
        V_after_spike = V_prev * (1 - S_prev) + reset_value * S_prev

        # Leaky Integration
        # The potential integrates the new input. This is the core update.
        leak_alpha = torch.exp(-F.softplus(self.leak_tau_param))
        V_t = leak_alpha * V_after_spike.detach() + self.linear_in(x_t)

        # Updating Adaptation State
        # The adaptation state for this step decays and is influenced by the previous spike.
        adaptation_alpha = torch.exp(-F.softplus(self.adaptation_tau_param))
        B_t = adaptation_alpha * B_prev.detach() + S_prev

        # Soft Reset: Blend between previous potential and resting potential
        effective_threshold = self.firing_threshold + B_t

        # A neuron can only fire if it is NOT in a refractory period.
        # Spike Generation for the current timestep
        S_t = (1 - is_refractory.float()) * spike_fn(V_t - effective_threshold)

        # Update Refractory Counter for the next timestep
        R_t = torch.relu(R_prev.detach() - 1) + self.refractory_steps * S_t

        # Determine final output (analog or spike)
        # The analog output is a function of the final potential V_t.
        y_t_analog = self.sfu(V_t)

        # The output is silenced if the neuron is in a refractory period.
        output = (1 - is_refractory.float()) * (S_t if return_spike else y_t_analog)

        # Returning the output and the new state for the next timestep
        return output, (V_t, R_t, B_t)

    def init_state(self, batch_size: int, device: torch.device):
        """Initializes V, R, and B states to zero."""
        dtype = self.V_rest.dtype
        V0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        R0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        B0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return V0, R0, B0


class LIFRnn(nn.Module):
    """
       A complete Recurrent Neural Network built by stacking one or more LIFLayers.

       This module handles the processing of entire sequences of data by iterating
       through the time dimension and managing the hidden states of its internal layers.
       It is designed to be device-agnostic; create the model first, then move it to
       the desired device using `.to(device)`.

       Args:
           input_size (int): The number of features in the input sequence.
           output_size (int): The number of features in the final output.
           hidden_layers_config (List[int]): A list of integers defining the number
                                              of neurons in each hidden LIFLayer.
           refractory_steps (int, optional): The refractory period for all layers.
                                            Defaults to 3.
       """

    def __init__(self, input_size: int, output_size: int, hidden_layers_config: List[int], **kwargs):
        super().__init__()
        self.lif_layers = nn.ModuleList()
        layer_input_size = input_size
        for hidden_size in hidden_layers_config:
            self.lif_layers.append(LIFLayer(layer_input_size, hidden_size, **kwargs))
            layer_input_size = hidden_size
        last_hidden_size = hidden_layers_config[-1]
        self.fc_out = nn.Linear(last_hidden_size, output_size)

    def forward(self, x: torch.Tensor, return_spike: bool = False) -> torch.Tensor:
        """
            Processes a batch of sequences through the stacked LIFLayers.

            Args:
                   x (Tensor): The input tensor of shape (batch_size, sequence_length, input_size).
                   return_spike (bool, optional): If True, all layers will output binary spikes.
                                                  Defaults to False.

            Returns:
                   The output tensor from the final layer at the final timestep, after the
                   output projection. Shape: (batch_size, output_size).
        """
        batch_size, sequence_length, _ = x.shape
        device = x.device
        states = [layer.init_state(batch_size, device) for layer in self.lif_layers]
        outputs_over_time = []
        for t in range(sequence_length):
            x_t = x[:, t, :]
            for i, layer in enumerate(self.lif_layers):
                x_t, new_state = layer(x_t, states[i], return_spike)
                states[i] = new_state
            outputs_over_time.append(x_t)
        output_sequence = torch.stack(outputs_over_time, dim=1)
        return self.fc_out(x_t)


# ==============================================================================
# SECTION 2: The Classifier Wrappers and Benchmark Framework
# ==============================================================================

class RNNClassifier(nn.Module):
    """A generic classifier that wraps a sequential feature extractor."""

    def __init__(self, rnn_backbone, hidden_size, num_classes):
        super().__init__()
        self.rnn = rnn_backbone
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        features = self.rnn(x)
        return self.classifier(features)


class GRUExtractor(nn.Module):
    """A wrapper for nn.GRU to match the LIFRnn interface."""

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, final_hidden_state = self.gru(x)
        return final_hidden_state[-1]


def set_seed(seed: int):
    """Sets a random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_single_trial(model_name: str, model_class: nn.Module, model_args: dict, device: torch.device,
                     train_loader, val_loader, epochs, lr):
    """Runs a single training and evaluation trial for a given model."""
    print(f"  --- Starting Trial for {model_name} (LR={lr}) ---")
    model = model_class(**model_args).to(device)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())

        epoch_accuracy = accuracy_score(all_true, all_preds)
        val_accuracies.append(epoch_accuracy)

    return val_accuracies


def run_final_benchmark():
    """Main function to orchestrate the hyperparameter sweep and final comparison."""
    # --- 1. Configuration ---
    print("--- Setting up Final Benchmark ---")
    set_seed(42)
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS = 16, 64, 2
    NUM_CLASSES, SEQ_LEN, BATCH_SIZE = 2, 50, 128
    EPOCHS = 20  # A reasonable number of epochs for a sweep
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # --- 2. Meaningful Synthetic Data ---
    print("Generating meaningful synthetic data...")
    dataset = torch.utils.data.TensorDataset(
        torch.randn(BATCH_SIZE * 20, SEQ_LEN, INPUT_SIZE),
        (torch.randn(BATCH_SIZE * 20, SEQ_LEN, INPUT_SIZE).mean(dim=(1, 2)) > 0).long()
    )
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)),
                                                                         len(dataset) - int(0.8 * len(dataset))])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- 3. Establish GRU Baseline ---
    print("\n--- Establishing GRU Baseline ---")
    gru_backbone_args = {'input_size': INPUT_SIZE, 'hidden_size': HIDDEN_SIZE, 'num_layers': NUM_LAYERS}
    gru_classifier_args = {'rnn_backbone': GRUExtractor(**gru_backbone_args), 'hidden_size': HIDDEN_SIZE,
                           'num_classes': NUM_CLASSES}
    gru_learning_curve = run_single_trial(
        "GRU (Baseline)", RNNClassifier, gru_classifier_args, device, train_loader, val_loader, EPOCHS, lr=1e-3
    )
    gru_final_accuracy = gru_learning_curve[-1]
    print(f"--- GRU Baseline Final Accuracy: {gru_final_accuracy:.2%} ---")

    # --- 4. Run Hyperparameter Sweep for LIFRnn ---
    lif_learning_rates_to_test = [3e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    best_lif_accuracy = 0
    best_lif_lr = None
    best_lif_curve = None
    lif_results_by_lr = {}

    print("\n--- Starting Hyperparameter Sweep for LIFRnn ---")
    lif_rnn_args = {'input_size': INPUT_SIZE, 'output_size': HIDDEN_SIZE,
                    'hidden_layers_config': [HIDDEN_SIZE] * NUM_LAYERS}
    lif_classifier_args = {'rnn_backbone': LIFRnn(**lif_rnn_args), 'hidden_size': HIDDEN_SIZE,
                           'num_classes': NUM_CLASSES}

    for lr in lif_learning_rates_to_test:
        set_seed(42)  # Use the same seed for each LR trial for a fair comparison
        learning_curve = run_single_trial(
            f"LIFRnn", RNNClassifier, lif_classifier_args, device, train_loader, val_loader, EPOCHS, lr=lr
        )
        final_accuracy = learning_curve[-1]
        lif_results_by_lr[lr] = final_accuracy
        if final_accuracy > best_lif_accuracy:
            best_lif_accuracy = final_accuracy
            best_lif_lr = lr
            best_lif_curve = learning_curve

    # --- 5. Print Final Report ---
    print("\n\n" + "=" * 35 + " FINAL BENCHMARK SUMMARY " + "=" * 35)
    print("\n--- GRU Baseline Performance ---")
    print(f"  Final Validation Accuracy: {gru_final_accuracy:.2%}")

    print("\n--- LIFRnn Performance by Learning Rate ---")
    for lr, acc in lif_results_by_lr.items():
        print(f"  LR: {lr:<9} -> Final Validation Accuracy: {acc:.2%}")

    print("\n--- Optimal Configuration ---")
    print(f"  Best LIFRnn Learning Rate: {best_lif_lr}")
    print(f"  Best LIFRnn Accuracy:      {best_lif_accuracy:.2%}")

    print("\n" + "=" * 95)

    # --- 6. Plot the Best LIFRnn vs. GRU ---
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, EPOCHS + 1), gru_learning_curve, marker='s', linestyle='--',
             label=f'GRU (Baseline) - Final Acc: {gru_final_accuracy:.2%}')
    if best_lif_curve:
        plt.plot(range(1, EPOCHS + 1), best_lif_curve, marker='o', linestyle='-',
                 label=f'Best LIFRnn (LR={best_lif_lr}) - Final Acc: {best_lif_accuracy:.2%}')

    plt.title("LIFRnn (Optimal LR) vs. GRU Baseline: Learning Curves", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.xticks(range(0, EPOCHS + 1, 2))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.ylim(bottom=0.45, top=1.02)  # Start y-axis at 45%
    plt.show()


#################################
#################################
#################################
#################################
#################################
#################################
#################################


# ==============================================================================
# SECTION 2: The Classifier Wrapper and Baseline Model
# ==============================================================================

class RNNClassifier(nn.Module):
    """A generic classifier that wraps a sequential feature extractor."""

    def __init__(self, rnn_backbone, hidden_size, num_classes):
        super().__init__()
        self.rnn = rnn_backbone
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.is_lif = "LIFRnn" in rnn_backbone.__class__.__name__

    def forward(self, x):
        if self.is_lif:
            # Your LIFRnn returns the final hidden state of the last layer
            features = self.rnn(x)
        else:  # GRU
            # GRU returns (output_sequence, final_hidden_state_per_layer)
            _, final_hidden_state = self.rnn(x)
            # Take the hidden state of the final layer for multi-layer GRUs
            features = final_hidden_state[-1]

        return self.classifier(features)


# ==============================================================================
# SECTION 3: The Rigorous Benchmark Framework
# ==============================================================================

def set_seed(seed: int):
    """Sets a random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_single_trial(model_name: str, model_class: nn.Module, model_args: dict, device: torch.device,
                     train_loader, val_loader, epochs, lr):
    """Runs a single training and evaluation trial for a given model."""
    print(f" Starting Trial for {model_name} ")
    model = model_class(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_accuracies = []

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation loop
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())

        epoch_accuracy = accuracy_score(all_true, all_preds)
        val_accuracies.append(epoch_accuracy)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch + 1}/{epochs}, Val Accuracy: {epoch_accuracy:.2%}")

    train_time = time.time() - start_time
    final_f1 = f1_score(all_true, all_preds, average="weighted")

    return val_accuracies, final_f1, train_time


def run_advanced_benchmark():
    """
    Main function to orchestrate the multi-trial benchmark.
    """
    # --- 1. Configuration ---
    print("--- Setting up Advanced Benchmark ---")
    set_seed(42)

    INPUT_SIZE = 16
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    NUM_CLASSES = 2  # Binary classification: is the mean positive or negative?
    SEQ_LEN = 50
    BATCH_SIZE = 128
    EPOCHS = 20
    LR = 5e-05
    NUM_TRIALS = 5  # Run 5 times for statistical significance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # --- 2. Meaningful Synthetic Data ---
    print("Generating meaningful synthetic data...")
    num_samples = BATCH_SIZE * 20
    X_data = torch.randn(num_samples, SEQ_LEN, INPUT_SIZE)
    # The label depends on the mean of the entire sequence
    y_data = (X_data.mean(dim=(1, 2)) > 0).long()

    dataset = torch.utils.data.TensorDataset(X_data, y_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- 3. Models to Benchmark ---

    # Baseline: GRU
    gru_backbone_args = {'input_size': INPUT_SIZE, 'hidden_size': HIDDEN_SIZE, 'num_layers': NUM_LAYERS,
                         'batch_first': True}
    gru_classifier_args = {'rnn_backbone': nn.GRU(**gru_backbone_args), 'hidden_size': HIDDEN_SIZE,
                           'num_classes': NUM_CLASSES}

    # Your Model: LIFRnn
    lif_rnn_args = {'input_size': INPUT_SIZE, 'output_size': HIDDEN_SIZE,
                    'hidden_layers_config': [HIDDEN_SIZE] * NUM_LAYERS}
    lif_classifier_args = {'rnn_backbone': LIFRnn(**lif_rnn_args), 'hidden_size': HIDDEN_SIZE,
                           'num_classes': NUM_CLASSES}

    models_to_test = {
        "LIFRnn": {'class': RNNClassifier, 'args': lif_classifier_args},
        "GRU (Baseline)": {'class': RNNClassifier, 'args': gru_classifier_args},
    }

    # --- 4. Run All Trials ---
    all_results = {}
    all_learning_curves = {}

    for name, model_info in models_to_test.items():
        trial_accuracies = []
        trial_f1s = []
        trial_times = []

        for i in range(NUM_TRIALS):
            print(f"\n--- Running Trial {i + 1}/{NUM_TRIALS} for {name} ---")
            # Set a different seed for each trial to test robustness
            set_seed(42 + i)
            learning_curve, f1, train_time = run_single_trial(
                name, model_info['class'], model_info['args'], device, train_loader, val_loader, EPOCHS, LR
            )
            trial_accuracies.append(learning_curve[-1])  # Final accuracy
            trial_f1s.append(f1)
            trial_times.append(train_time)
            if i == 0:  # Store the learning curve from the first trial for plotting
                all_learning_curves[name] = learning_curve

        all_results[name] = {
            "Accuracy": f"{np.mean(trial_accuracies):.2%} ± {np.std(trial_accuracies):.2%}",
            "F1-Score": f"{np.mean(trial_f1s):.4f} ± {np.std(trial_f1s):.4f}",
            "Train Time (s)": f"{np.mean(trial_times):.2f} ± {np.std(trial_times):.2f}"
        }

    # --- 5. Print Report ---
    print("\n\n" + "=" * 35 + " ADVANCED BENCHMARK SUMMARY " + "=" * 35)
    header = f"| {'Model':<20} | {'Final Accuracy (Mean ± Std)':<30} | {'Final F1-Score (Mean ± Std)':<32} | {'Avg Train Time (s)':<20} |"
    print(header)
    print(f"|{'-' * 22}|{'-' * 32}|{'-' * 34}|{'-' * 22}|")
    for name, metrics in all_results.items():
        row = f"| {name:<20} | {metrics['Accuracy']:<30} | {metrics['F1-Score']:<32} | {metrics['Train Time (s)']:<20} |"
        print(row)
    print("=" * 116)

    # --- 6. Plot Learning Curves ---
    plt.figure(figsize=(10, 6))
    for name, curve in all_learning_curves.items():
        plt.plot(range(1, EPOCHS + 1), curve, marker='o', linestyle='-', label=name)

    plt.title("Validation Accuracy Learning Curves (Trial 1)", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.xticks(range(0, EPOCHS + 1, 5))
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0.4)  # Start y-axis at 40% for better visibility
    plt.show()


if __name__ == '__main__':
    run_advanced_benchmark()
