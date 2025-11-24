# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements "Minimalist Concept Erasure in Generative Models" (ICML 2024), a method for removing unwanted concepts from diffusion models (Stable Diffusion, SDXL, Flux, etc.) through sparse weight masking. The core approach learns binary masks on attention, feedforward, and normalization layers to selectively erase concepts while preserving model capabilities.

Package name: `diffsolver`

## Development Commands

### Installation and Setup
```bash
# Install package in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[core,test,notebook]"

# Setup pre-commit hooks
make precomit
```

### Testing
```bash
# Run all tests
pytest -s
# Or use make
make test
```

### Code Quality
```bash
# Run pre-commit checks (formatting, linting)
make format
```

### Training
```bash
# Direct training with config file
python scripts/train.py --cfg configs/flux.yaml

# Training with Accelerate (for distributed training)
accelerate launch scripts/train.py --cfg configs/flux.yaml

# Using Makefile shortcuts
make accvisual cfg=flux  # Uses CONFIG_DIR and RESULTS_DIR from .env

# Generate hyperparameter sweep configs
make gen

# Run hyperparameter sweep
make run
```

### Cleaning
```bash
# Remove cache files, logs, and generated configs
make clean
```

## Architecture Overview

### Core Components

**Models (`src/diffsolver/models/`)**
- Base class: `DiffusionModelForCheckpointing` defines the interface for gradient checkpointing during training
- Model-specific implementations: `flux.py`, `sd2.py`, `sd3.py`, `sdxl.py`, `dit.py`
- Each model extends diffusers pipelines and implements three key methods:
  - `inference_preparation_phase()`: Setup and encoding phase
  - `inference_denoising_step()`: Single denoising iteration (for gradient checkpointing)
  - `inference_aft_denoising()`: Post-denoising processing (VAE decode)

**Hooks System (`src/diffsolver/hooks/`)**
- `BaseHooker`: Abstract base class for all hookers with mask management (save/load/binarize)
- `CrossAttentionExtractionHook`: Masks attention heads using custom attention processors
- `FeedForwardHooker`: Masks feedforward layer channels
- `NormHooker`: Masks normalization layer channels
- `LinearLayerHooker`: For older SD models (SD1/SD2)
- `init_hooker()`: Factory function that initializes appropriate hookers based on model type
- Masking strategies: `sigmoid`, `hard_discrete` (hard concrete distribution), `binary`, `continues2binary`

**Training Loop (`scripts/train.py`)**
- Uses gradient checkpointing to save memory by splitting forward pass into preparation + denoising steps
- Two-phase backpropagation:
  1. Compute loss on final output, backprop to last latent
  2. Iterate through timesteps backward, computing gradients for lambda masks
- Loss components:
  - Reconstruction loss: Preserve model capabilities on neutral prompts
  - Intermediate loss: Erase target concepts by matching "deconcept" latents
- Supports distributed training via Accelerate

**Data (`src/diffsolver/data/`)**
- `PromptImageDataset`: Loads prompt-image pairs with pre-computed latents
- `unlearn_template.py`: Templates for concept erasure (e.g., nudity, copyright)
- `adversarial_promptdatasets.py`: Adversarial prompt datasets for robustness testing

**Configuration System**
- Uses OmegaConf for YAML-based configs
- Key config sections:
  - `data`: Dataset paths, batch size, concept to erase
  - `trainer`: Model type, learning rates (attn_lr, ff_lr, n_lr), masking strategy, num_intervention_steps
  - `loss`: Loss function choices and weights (beta for reconstruction)
  - `logger`: W&B or CSV logging, output directories
  - `accelerator`: Distributed training settings

### Model Type Differences

- **Flux/SD3**: Use transformer-based architecture → hooks on attention, feedforward, and norm layers
- **SD1/SD2/SDXL**: Use UNet architecture → hooks on linear layers

### Baselines

The `baselines/` directory contains implementations of comparison methods:
- **ESD**: Erased Stable Diffusion
- **CA**: Concept Ablation
- **SLD**: Safe Latent Diffusion
- **FlowEdit**: Flow-based editing
- **EAP**: Embedding-based approach
- **Ring-a-Bell**: Adversarial prompt generation

## Key Concepts

**Lambda (λ) Masks**: Learnable parameters that control which model components are masked. During training, lambda values are optimized; during inference, they're binarized to create sparse masks.

**Gradient Checkpointing**: The training uses a custom checkpointing scheme that splits the diffusion process into steps, storing intermediate latents and recomputing forward passes during backward pass to save memory.

**Deconcept Prompts**: Neutral prompts used to generate target latents for erasing unwanted concepts while preserving general capabilities.

**Hook Registration**: Masks are applied by registering forward hooks on PyTorch modules. The hooks intercept activations and apply learned masks before passing to the next layer.

## Configuration Notes

- Configs are in `configs/` directory with model-specific YAML files
- The training script expects a `.env` file with `PYTHON`, `RESULTS_DIR`, and `CONFIG_DIR` variables for Makefile commands
- Pre-trained lambda masks are stored in `models/flux/` (e.g., `attn.pt`, `ff.pt`, `norm.pt`)
- Set `trainer.attn_lr`, `trainer.ff_lr`, `trainer.n_lr` to 0 to disable training for specific components

## Running Single Tests

```bash
# Run specific test file
pytest tests/test_hooks.py -s

# Run specific test function
pytest tests/test_hooks.py::test_function_name -s
```

## Important Implementation Details

- Lambda masks use float32 precision even when model is bf16/fp16 for numerical stability
- The `hard_discrete` masking uses hard concrete distribution for differentiable discrete masks
- Dataset filtering can be enabled with `data.with_fg_filter` to remove prompts with low foreground ratios
- Binarization supports both local (per-layer) and global (across-model) sparsity constraints
