# GidionV — Multi-Stream Memory-Augmented Transformer

**GidionV** is a research framework for training and experimenting with novel memory-augmented language models built from scratch in PyTorch. It implements two original transformer architectures that go beyond standard context-window RAG by natively fusing external knowledge streams directly inside the decoder via cross-attention.

> Named after the sentient AI from DC Comics' *The Flash* — an entity that processes and dynamically retrieves vast knowledge in real-time.

---

## Why GidionV?

Standard LLMs are limited by their context window. Bolt-on RAG pipelines (LangChain, LlamaIndex) append retrieved documents to the prompt, causing quadratic attention costs and "lost in the middle" degradation.

GidionV processes external knowledge **in parallel** to the main sequence, via dedicated memory encoder paths and cross-attention fusion inside each decoder block. The model learns to *query* external streams the same way it attends to its own hidden state.

**Proven behaviours (empirically verified):**
- The model can reproduce information *only* present in the memory stream — not in the main prompt or training label (see [Memory Swap Test](#running-the-memory-swap-test))
- Multi-stream synthesis: the model correctly assembles an answer from facts split across 3 independent parallel streams
- Gated expert routing: the MoE variant learns which expert stream is relevant and suppresses the others

---

## Architectures

### 1. Multi-Memory Transformer (`MultiMemoryTransformer`)
Each memory stream gets its own dedicated cross-attention layer per decoder block. Streams are blended using learned `fusion_weights` parameters.

```
Input Tokens ──► Token Embedding ──► Decoder Stack ──► LM Head ──► Logits
                                          │
     Stream 0 ──► Memory Encoder ──► CrossAttn_0 ─┐
     Stream 1 ──► Memory Encoder ──► CrossAttn_1 ─┼─ weighted sum ──► residual
     Stream N ──► Memory Encoder ──► CrossAttn_N ─┘
```

**Best for:** Fixed small number of streams where each stream always matters equally.

---

### 2. Memory-of-Experts Transformer (`MemoryOfExpertsTransformer`)
A gating attention mechanism scores each expert stream and dynamically routes to the most relevant ones. All streams are fused into a single context before cross-attention — far more compute-efficient at scale.

```
Input Tokens ──► Token Embedding ──► Decoder Stack ──► LM Head ──► Logits
                                          │
     Stream 0 ──► LIF Memory Encoder ─┐
     Stream 1 ──► LIF Memory Encoder ─┼─ Gating Attention ──► fused context ──► CrossAttn ──► residual
     Stream N ──► LIF Memory Encoder ─┘
```

**Best for:** Many streams where only 1-2 are relevant to any given query. Scales efficiently.

The **LIF Memory Encoder** uses a biologically-inspired Leaky Integrate-and-Fire RNN variant for sequence encoding instead of a standard transformer encoder, making it more parameter-efficient.

---

## Project Structure

```
gidionv-torch/
│
├── train.py                        # Main training entry point
├── run.py                          # Interactive CLI chat interface
├── sanity_check_finetune.py        # Quick inference test (accepts config path as arg)
├── plot.py                         # Training loss/metric plotting utility
├── prepare_finetune_v2.py          # Transfer pretrain → finetune checkpoint (Multi-Memory)
├── prepare_finetune_moe_v1.py      # Transfer pretrain → finetune checkpoint (MoE)
├── prepare_scaled_pretrain_checkpoint.py  # Scale up pretrain checkpoint utility
│
├── experiments/                    # Research scripts and architecture validation tests
│   ├── swap_memory_test_v2.py      # Definitive memory stream swap test (PASS proven)
│   ├── ablation_memory_test.py     # Memory stream ablation study
│   ├── overfit_test.py             # Basic architecture overfit test
│   ├── overfit_memory_test.py      # Single-stream memory overfit test
│   └── overfit_multi_memory_test.py # Multi-stream memory overfit test
│
├── configs/
│   ├── pretrain_v2.json            # Multi-Memory pre-training config (~120M params)
│   ├── finetune_v2.json            # Multi-Memory fine-tuning config
│   ├── pretrain_moe_v1.json        # MoE pre-training config (~70M params)
│   ├── finetune_moe_v1.json        # MoE fine-tuning config
│   └── tokenizers/
│       └── bpe_v2_50200_vocab.json # Custom BPE tokenizer (50,200 vocab)
│
├── src/
│   ├── lib/
│   │   ├── core/
│   │   │   ├── attention.py                    # Multi-Head Attention (causal + cross)
│   │   │   ├── ffn.py                          # Dynamic Feed-Forward Network
│   │   │   ├── lif_rnn.py                      # Leaky Integrate-and-Fire RNN
│   │   │   ├── lif_memory_encoder.py           # LIF-based memory encoder (used in MoE)
│   │   │   ├── memory_encoder.py               # Transformer-based memory encoder (used in Multi-Memory)
│   │   │   ├── maf_decoder_block.py            # Memory-Attention Fusion Decoder Block (Multi-Memory)
│   │   │   ├── hiearchical_fusion_decoder_block.py  # Hierarchical Gating Decoder Block (MoE)
│   │   │   ├── positional_encoding.py          # Sinusoidal positional encoding
│   │   │   └── hf_tokenizer_wrapper.py         # HuggingFace tokenizer wrapper
│   │   └── transformer/
│   │       ├── multi_memory_transformer.py     # Multi-Memory Transformer (full model)
│   │       └── memory_of_experts_transformer.py # Memory-of-Experts Transformer (full model)
│   │
│   ├── trainer/
│   │   ├── trainer.py              # Main training loop (AMP, grad accum, recovery)
│   │   └── datafactory.py          # Dataset factory (pretrain / finetune / online / local)
│   │
│   ├── streamers/
│   │   ├── pretrain_streamer.py    # Streaming pre-train data (RedPajama, C4 via HuggingFace)
│   │   └── finetune_streamer.py    # Streaming fine-tune data (Dolly-15k via HuggingFace)
│   │
│   ├── loaders/
│   │   └── finetune_loader.py      # Prompt formatting utilities
│   │
│   └── data/
│       └── saver_loader.py         # Checkpoint save/load + config-driven model selection
│
└── research/
    └── models/                     # Saved checkpoints and training logs (gitignored)
```

---

## Getting Started

### 1. Installation

```bash
git clone https://github.com/your-username/gidionv-torch
cd gidionv-torch
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

**Requirements:** Python 3.11+, PyTorch 2.7+, CUDA GPU strongly recommended.

---

### 2. Pre-Training

Training streams data directly from HuggingFace (RedPajama, C4 for pre-train; Dolly-15k for fine-tune). No manual data download required.

**Multi-Memory Transformer (~120M params):**
```bash
python train.py --config configs/pretrain_v2.json --source online
```

**Memory-of-Experts Transformer (~70M params):**
```bash
python train.py --config configs/pretrain_moe_v1.json --source online
```

Training logs (loss, val_loss, perplexity, lr, grad_norm, memory weights) are saved as CSV files inside `research/models/<model_name>/`.

---

### 3. Instruction Fine-Tuning

Once pre-training loss has plateaued, transfer the checkpoint and start fine-tuning on Dolly-15k:

**Multi-Memory:**
```bash
python prepare_finetune_v2.py
python train.py --config configs/finetune_v2.json --source online --type finetune
```

**Memory-of-Experts:**
```bash
python prepare_finetune_moe_v1.py
python train.py --config configs/finetune_moe_v1.json --source online --type finetune
```

---

### 4. Sanity Check (Inference)

Run a quick Q&A test against any checkpoint:

```bash
# Multi-Memory fine-tuned model
python sanity_check_finetune.py configs/finetune_v2.json

# MoE fine-tuned model
python sanity_check_finetune.py configs/finetune_moe_v1.json
```

---

### 5. Switching Architectures

Architecture selection is config-driven via the `MODEL_ARCHITECTURE` key:

```json
"MODEL_ARCHITECTURE": "moe"          // Uses MemoryOfExpertsTransformer
"MODEL_ARCHITECTURE": "multi_memory"  // Uses MultiMemoryTransformer (default)
```

---

## Running the Memory Swap Test

This test empirically proves that the memory stream architecture is causally reading from external streams at inference time (not just memorizing from training labels).

```bash
python swap_memory_test_v2.py
```

**How it works:**
1. The model is trained with a different random code on every step (e.g., `XAJI0`, `L1CQ9`, `WAOUA`...). It cannot memorize any single code.
2. At inference time, a brand new code is injected into the memory stream.
3. If the model outputs the new code — it is reading *live* from the memory stream.

**Result achieved:**
```
Test Code (never seen in training): 'EUSWR'
[WITH memory = 'EUSWR']: The secret code is 'EUSWR'.   ✓ PASS
[WITHOUT memory]:         The secret code is 'FXW'.
```

---

## Custom Tokenizer

GidionV uses a custom BPE tokenizer trained specifically for this project with special instruction tokens:

| Token | ID | Purpose |
|---|---|---|
| `</INST>` / `<eos>` | 0 | End of sequence |
| `<pad>` | 1 | Padding |
| `<ASSISTANT>` | 2 | Assistant turn marker |
| `<bos>` | 7 | Beginning of sequence |
| `<USER>` | 5 | User turn marker |

Prompt format for fine-tuning:
```
<bos><USER><INST> {instruction} </INST><ASSISTANT> {response}<eos>
```

---

## Key Design Decisions

- **No external RAG pipeline:** Knowledge streams are fused inside the decoder, not prepended to the context window.
- **Parallel memory encoding:** Memory streams are encoded independently and simultaneously — the O(N²) attention cost does not scale with the number of streams.
- **Pre-computable memory:** Memory stream encodings can be computed once and cached, making inference fast even for large knowledge bases.
- **Config-driven everything:** Switch architectures, data sources, and training types purely through JSON configs.
- **Automatic recovery:** The trainer detects NaN/Inf gradients and automatically reloads the last clean checkpoint.

---

## Acknowledgements

Built with PyTorch. Training data from HuggingFace Datasets (RedPajama, C4, Dolly-15k).
