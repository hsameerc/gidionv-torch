# Project GidionV: A Memory-Augmented Transformer Architecture

**Project GidionV** is an end-to-end deep learning framework for building, training, and interacting with a novel,
memory-augmented transformer model. This project was undertaken as an exercise in first-principles engineering to create
an intelligent agent that overcomes the static knowledge limitations of conventional language models.

The entire system, from the lowest-level neural network layers to the production-grade training pipeline, has been built
from scratch using NumPy/CuPy, providing complete control and a deep understanding of the underlying mechanics.

The name is a tribute to the sentient AI from DC Comics' *The Flash*, embodying the project's goal of creating an AI
that can process, reason with, and dynamically utilize vast amounts of information.

## Core Philosophy: The Augmented Brain

Standard language models possess a powerful but static **parametric memory** (their trained weights). GidionV enhances
this with a dynamic, external **non-parametric memory** (a FAISS vector database). This hybrid architecture allows
GidionV to:

- **Ground Responses in Fact:** By retrieving and referencing specific documents, it reduces hallucinations and provides
  verifiable answers.
- **Learn Instantly:** New facts can be added to its external memory without costly re-training.
- **Maintain Context:** It can remember previous turns in a conversation and handle follow-up questions.
- **Specialize On-Demand:** Its expertise can be dynamically altered by providing it with different memory sources (
  e.g., legal documents vs. technical manuals).

## The Architecture: `AugmentedTransformer`

The core of the project is a custom Encoder-Decoder transformer designed for memory injection.

1. **Memory Encoder:** A stack of standard `TransformerEncoderBlock`s that reads a set of retrieved memory vectors and
   processes them into a rich, contextualized `memory_context`.
2. **Augmented Decoder:** A stack of custom `AugmentedDecoderBlock`s. At every layer, this decoder uses a *
   *cross-attention mechanism** to "look at" the `memory_context`, allowing the external knowledge to influence the
   generation of each new token.
3. **From-Scratch Components:** Every part of this architecture was built and rigorously unit-tested in isolation,
   including `MultiHeadAttention`, `LayerNorm`, `DynamicReluNetwork`, and a from-scratch `AdamW` optimizer.

## Project Structure

The repository is organized into a clean, modular structure:

- `run.py`: The main interactive CLI for chatting with a trained GidionV model.
- `train.py`: The unified script for both large-scale pre-training and fine-tuning.
- `finetune.py`: (Optional) A dedicated script for the fine-tuning phase.
- `validate.py`: A script for running validation tests on a trained model.
- `configs/`: Contains all JSON configuration files for different training and inference scenarios.
- `sample-data/`: Holds small `.jsonl` and `.txt` files for testing the pipelines.
- `src/`: The main source code directory.
    - `lib/core/`: Contains all the fundamental, from-scratch neural network components.
    - `lib/memory/`: Contains the high-level cognitive components: `FaissMemoryCore`, `IntentRouter`, and the
      `MemoryController`.
    - `lib/transformer/`: Contains the top-level `GidionAugmentedTransformer` assembly.
    - `utils/, data, loaders`: Includes helper utilities for data loading, saving/loading models, plotting, and training.
    - `test/`: A comprehensive suite of unit tests for every component, ensuring mathematical correctness and
      robustness.

## Getting Started

### 1. Installation

Clone the repository and install the required dependencies. It is highly recommended to use a virtual environment.

```bash
git clone [your-repo-url]
cd [your-repo-name]
pip install -r requirements.txt
```

Note: This project can run in either CPU (NumPy) or GPU (CuPy) mode. Set the USE_NUMPY_CPU flag at the top of the
relevant source files to switch between modes.

### 2. Training

The training process is typically a multi-stage endeavor.
**Stage 1: Pre-training (on a large .txt corpus)**
This stage builds the model's foundational understanding of language.

```bash
# First, ensure your config file (e.g., configs/pretrain.json for jsonl and  configs/gidion.json for txt corpus) is correct
# Then, launch the training script
python train.py --config configs/gidion.json
#OR
python train.py --config configs/pretrain.json
#based on your training file format
```

**Stage 2: Instruction Fine-tuning (on a .jsonl dataset)**
This stage teaches the pre-trained model how to be a helpful assistant. It requires a dataset where each line is a JSON
object, like:
{"source": "### INSTRUCTION: ...", "target": "..."}

```bash
# This assumes you have a fine-tuning config pointing to your .jsonl file
# and that you have a pre-trained checkpoint to resume from.
python train.py --config configs/finetune.json
```

### 3. Inference: Chatting with GidionV
Once you have a trained and fine-tuned model, you can interact with it using the main run script.

```bash
# Point the config to your best fine-tuned model checkpoint
python run.py --config configs/gidion.json
```

The CLI supports several commands:
 - `/help`: Displays available commands.
 - `/learn`: Manually teaches GidionV a new Q&A pair, which is instantly added to its memory.
 - `/memor`y [N]: Inspects the N most recent memories.
 - `/exit:` Shuts down the application gracefully, saving the memory state.

### 4. Model Optimization (Optional)
The framework includes an advanced feature to compress the trained model's weights using SVD for faster inference.

```bash
# Apply 95% rank compression at runtime
python run.py --config configs/gidion.json --lora_rank 0.95

# Apply a fixed rank of 128
python run.py --config configs/gidion.json --lora_rank 128
```

### Development Method
A core philosophy of this project was to build "with AI, not just on AI." Throughout the development, contemporary LLMs were used as a symbiotic partner for debugging complex mathematics, brainstorming architectural ideas, and accelerating the implementation of from-scratch components, mirroring the iterative, tool-assisted engineering process.
This project stands as a complete, end-to-end case study in modern AI development, from theoretical architecture to a fully functional and intelligent agent.


