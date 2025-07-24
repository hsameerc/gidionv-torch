# File: final_benchmark_parity.py
# Description: The definitive benchmark script comparing the LIFRnn against a GRU
#              baseline on the Temporal Parity Task, a true test of sequential memory.

import torch
import torch.nn as nn
import time
import numpy as np
import random
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from typing import List

from src.lib.core.lifmodels import DualStateRNN


# ==============================================================================

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
    """A wrapper for nn.GRU to match the LIFRnn interface (returns last hidden state)."""

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
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch + 1}/{epochs}, Val Accuracy: {epoch_accuracy:.2%}")

    return val_accuracies


def run_parity_benchmark():
    """Main function to orchestrate the benchmark on the Temporal Parity Task."""
    # Configuration
    print("Setting up Temporal Parity Benchmark")
    set_seed(42)
    INPUT_SIZE = 1  # Each timestep is just a single value
    HIDDEN_SIZE = 128  # A smaller hidden size is sufficient for this task
    NUM_LAYERS = 8
    NUM_CLASSES = 2  # The parity is either odd (1) or even (0)
    SEQ_LEN = 50
    BATCH_SIZE = 256
    EPOCHS = 30  # This task is often learned quickly
    NUM_TRIALS = 1  # We can use 1 trial for the hyperparameter sweep

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # --- 2. The Temporal Parity Task Data ---
    print("Generating Temporal Parity Task data...")
    num_samples = BATCH_SIZE * 40
    # Input: A sequence of +1s and -1s
    X_data = torch.randint(0, 2, (num_samples, SEQ_LEN, INPUT_SIZE)).float() * 2 - 1
    # Target: 1 if the number of +1s is odd, 0 if even.
    y_data = (torch.sum(X_data > 0, dim=(1, 2)) % 2 == 1).long()
    # print(X_data)
    # print(y_data)
    dataset = torch.utils.data.TensorDataset(X_data, y_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- 3. Establish GRU Baseline ---
    # print("\n--- Establishing GRU Baseline ---")
    # gru_backbone_args = {'input_size': INPUT_SIZE, 'hidden_size': HIDDEN_SIZE, 'num_layers': NUM_LAYERS}
    # gru_classifier_args = {'rnn_backbone': GRUExtractor(**gru_backbone_args), 'hidden_size': HIDDEN_SIZE,
    #                        'num_classes': NUM_CLASSES}
    # gru_learning_curve = run_single_trial(
    #     "GRU (Baseline)", RNNClassifier, gru_classifier_args, device, train_loader, val_loader, EPOCHS, lr=1e-3
    # )
    # gru_final_accuracy = gru_learning_curve[-1]
    # print(f"--- GRU Baseline Final Accuracy: {gru_final_accuracy:.2%} ---")

    # --- 4. Run Hyperparameter Sweep for LIFRnn ---
    lif_learning_rates_to_test = [1e-2,]
    best_lif_accuracy = 0
    best_lif_lr = None
    best_lif_curve = None
    lif_results_by_lr = {}

    print("\n--- Starting Hyperparameter Sweep for LIFRnn ---")
    lif_rnn_args = {'input_size': INPUT_SIZE, 'output_size': HIDDEN_SIZE,
                     'hidden_layers_config': [HIDDEN_SIZE]
                    }
    lif_classifier_args = {'rnn_backbone': DualStateRNN(**lif_rnn_args), 'hidden_size': HIDDEN_SIZE,
                           'num_classes': NUM_CLASSES}

    for lr in lif_learning_rates_to_test:
        set_seed(42)
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
    print("\n\n" + "=" * 35 + " TEMPORAL PARITY BENCHMARK SUMMARY " + "=" * 35)
    print("\n--- GRU Baseline Performance ---")
    # print(f"  Final Validation Accuracy: {gru_final_accuracy:.2%}")

    print("\n--- LIFRnn Performance by Learning Rate ---")
    for lr, acc in lif_results_by_lr.items():
        print(f"  LR: {lr:<9} -> Final Validation Accuracy: {acc:.2%}")

    print("\n--- Optimal Configuration ---")
    print(f"  Best LIFRnn Learning Rate: {best_lif_lr}")
    print(f"  Best LIFRnn Accuracy:      {best_lif_accuracy:.2%}")

    print("\n" + "=" * 95)

    # --- 6. Plot the Best LIFRnn vs. GRU ---
    plt.figure(figsize=(12, 7))
    # plt.plot(range(1, EPOCHS + 1), gru_learning_curve, marker='s', linestyle='--',
    #          label=f'GRU (Baseline) - Final Acc: {gru_final_accuracy:.2%}')
    if best_lif_curve:
        plt.plot(range(1, EPOCHS + 1), best_lif_curve, marker='o', linestyle='-',
                 label=f'LIFRnn (LR={best_lif_lr}) - Final Acc: {best_lif_accuracy:.2%}')

    plt.title("LIFRnn vs. GRU on Temporal Parity Task", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.xticks(range(0, EPOCHS + 1, 5))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.ylim(bottom=0.45, top=1.02)
    plt.show()


if __name__ == '__main__':
    # Make sure to have your lif_rnn.py file in the same directory or in your python path
    # from lif_rnn import LIFRnn
    run_parity_benchmark()