import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps

plt.ion()


def all_plot(df, fig, axs):
    """
    Clears and redraws the plots on the existing figure.
    """
    for ax_row in axs:
        for ax in ax_row:
            ax.clear()

    df = df.dropna(how='all')
    for col in ['step', 'epoch', 'loss', 'val_loss', 'perplexity', 'lr', 'grad_norm']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    val_df = df.dropna(subset=['val_loss', 'perplexity']).copy()
    mem_cols = sorted([col for col in df.columns if col.startswith('mem_weight_')])

    if 'loss' in df.columns and not df['loss'].isnull().all():
        axs[0, 0].plot(df['step'], df['loss'].rolling(window=10, min_periods=1).mean(),
                       label='Training Loss (Smoothed)', color='blue', alpha=0.7)
        axs[0, 0].set_title("Training Loss (10-step Rolling Avg)")
        axs[0, 0].set_xlabel("Step")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].legend()
        axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Step vs. Perplexity
    if not val_df.empty:
        axs[0, 1].plot(val_df['step'], val_df['perplexity'].rolling(window=1, min_periods=1).mean(), label='Perplexity',
                       color='purple', marker='o',
                       linestyle='-')
        axs[0, 1].set_title("Validation Perplexity")
        axs[0, 1].set_xlabel("Step")
        axs[0, 1].set_ylabel("Perplexity")
        axs[0, 1].legend()
        axs[0, 1].grid(True, linestyle='--', alpha=0.6)
        axs[0, 1].set_yscale('log')  # Perplexity often best viewed on a log scale

    # Plot 3: Step vs. Validation Loss
    if not val_df.empty:
        axs[1, 0].plot(val_df['step'], val_df['val_loss'].rolling(window=1, min_periods=1).mean(),
                       label='Validation Loss', color='red', marker='o',
                       linestyle='-')
        axs[1, 0].set_title("Validation Loss")
        axs[1, 0].set_xlabel("Step")
        axs[1, 0].set_ylabel("Loss")
        axs[1, 0].legend()
        axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    # Plot 4: Learning Rate over Steps
    if 'lr' in df.columns and not df['lr'].isnull().all():
        axs[1, 1].plot(df['step'], df['lr'].rolling(window=10, min_periods=1).mean(), label='Learning Rate',
                       color='green')
        axs[1, 1].set_title("Learning Rate Schedule")
        axs[1, 1].set_xlabel("Step")
        axs[1, 1].set_ylabel("Learning Rate")
        axs[1, 1].legend()
        axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    # Plot 5: Gradient Norm (The new plot)
    if 'grad_norm' in df.columns and not df['grad_norm'].isnull().all():
        rolling_window = max(1, len(df) // 100)
        axs[2, 0].plot(df['step'], df['grad_norm'].rolling(window=1, min_periods=1).mean(),
                       label=f'Gradient Norm ({rolling_window}-step avg)', color='green')
        axs[2, 0].set_title("Gradient Norm (Smoothed)")
        axs[2, 0].set_xlabel("Step")
        axs[2, 0].set_ylabel("L2 Norm")
        axs[2, 0].set_yscale('log')
        axs[2, 0].legend()
        axs[2, 0].grid(True, linestyle='--', alpha=0.6)
    else:
        axs[2, 0].axis('off')

    # Plot 6: Memory Weights
    if mem_cols:
        mem_df = df[mem_cols].fillna(0.0)
        steps = df['step']

        viridis = colormaps['viridis']
        lin_space_series = pd.Series(range(len(mem_cols))) / (len(mem_cols) - 1)
        colors = viridis(lin_space_series.values)

        axs[2, 1].stackplot(
            steps,
            [mem_df[col] for col in mem_cols],
            labels=mem_cols,
            colors=colors,
            alpha=0.85,
            edgecolor='k', linewidth=0.3
        )

        dominant_streams = mem_df.idxmax(axis=1, skipna=True).fillna("Unknown")
        dominant_numeric = dominant_streams.map({name: i for i, name in enumerate(mem_cols)})

        axs[2, 1].plot(
            steps,
            dominant_numeric,
            linestyle='--', color='black', linewidth=1,
            label='Dominant Stream (Index)'
        )

        axs[2, 1].set_title("Memory Stream Weight Distribution", fontsize=11)
        axs[2, 1].set_xlabel("Step", fontsize=10)
        axs[2, 1].set_ylabel("Proportion", fontsize=10)
        axs[2, 1].set_ylim(0, 1.01)
        axs[2, 1].grid(True, linestyle='--', alpha=0.4)
        axs[2, 1].legend(loc='upper left', fontsize='x-small', ncol=2, frameon=False)

    else:
        axs[2, 1].axis('off')

    fig.suptitle('Training Progress Overview', fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    fig.canvas.draw()
    fig.canvas.flush_events()


def launch_log_plot(destination: str):
    DEST_LOG = Path(destination)
    REFRESH_INTERVAL_SECS = 30
    last_size = -1

    fig, axs = plt.subplots(3, 2, figsize=(15, 12))

    print("Live Plot Monitor Started (Interactive Mode)")
    print("Close the plot window to exit.")

    while True:
        if not plt.fignum_exists(fig.number):
            print("Plot window closed. Exiting monitor.")
            break

        try:
            if DEST_LOG.exists():
                new_size = DEST_LOG.stat().st_size
                if new_size != last_size:
                    last_size = new_size
                    print(f"[{time.strftime('%H:%M:%S')}] ✅ File changed. Redrawing plot...")
                    data_file = pd.read_csv(DEST_LOG)
                    all_plot(data_file, fig, axs)  # Pass the existing figure and axes
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] ℹ️ No change.")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ❌ Log not found.")

        except Exception as e:
            print(f"❌ Error: {e}")
        plt.pause(REFRESH_INTERVAL_SECS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Gidion Plot.")
    parser.add_argument('--path', type=str, default="research/models/gidionv_multi_memory/gidionv_multi_memory.csv",
                        help="Path to a Log file to override defaults.")
    args = parser.parse_args()
    launch_log_plot(args.path)
