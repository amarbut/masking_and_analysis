# Weight Distributions and Overparameterization

Exploratory analysis of weight distributions and masking in pre-trained transformer language models, corresponding to Chapter 4 of:

> **Searching for Nooks and Crannies: Geometric and Mechanistic Perspectives on Transformer Language Model Interpretability**
> Anna C. Marbut. PhD dissertation, University of Montana (2026).

## Overview

This repository investigates what happens inside transformer language models at the weight level before, during, and after pre-training. The central finding is that training scale and hyperparameter choices affect weight distributions far more than training task does, and that pre-trained weights remain surprisingly close to their initialized values even after extensive training. This connects to the lottery ticket hypothesis and raises questions about which weights are actually doing meaningful work.

The work is organized around three related threads:

**Pre-training Dynamics.** How do pre-training data scale, training task, and hyperparameter configuration shape the distribution of weights across transformer layers?

**Overparameterization.** Pre-trained weights remain surprisingly close to initialized values even after extensive training. Which weights are actually doing meaningful work? Can we use weight movement patterns to identify and mask the rest?

**Initialization.** Building on work showing that non-standard pre-training processes can produce surprisingly strong benchmarking performance, how do different model initializations and pre-training configurations affect downstream benchmark performance throughout training?

These observations connect to a broader interpretability framework: if weight distributions encode something meaningful about a model's representational capacity, then the geometry of weight space may itself be predictive of downstream performance, and targeted initialization strategies may be able to produce favorable latent geometry before linguistic training begins.

## Repository Structure

### `initialization_and_GLUE/`
Scripts exploring how initialization and pre-training configuration affect GLUE benchmark performance.

```
baseline-pretraining/   BabyLM baseline pre-training codebase (reference implementation)
compare_glue.py         Compare GLUE results across model configurations and masking runs
glue_parse.py           Parse and aggregate GLUE evaluation outputs
data_shuffle.py         Dataset preparation utilities
```

### `overparameterization/`
Implementation of weight masking experiments and masked pre-training.

```
mask_and_train.py       Main training script — parameterized, HPC-compatible masked
                        weight pre-training. Accepts all masking strategies via argparse.
BabyLM_mask.py          Lightweight masking utility for BabyLM-scale experiments
```

`mask_and_train.py` supports five masking strategies:

| Strategy | Masks weights based on... |
|---|---|
| `raw_value` | Smallest absolute weight values |
| `movement` | Weights that changed least from initialization |
| `magnitude` | Smallest magnitude change from initialization |
| `direction_mask` | Weights that moved toward zero (partial zeroing) |
| `direction_all` | Weights that moved toward zero (full zeroing) |

### `visualize_and_explore/`
Analysis and visualization of weight distributions across training scales and checkpoints.

```
weight_explore.py          Compare weight distributions across model sizes
                           (10M, 100M token, and base RoBERTa)
weight_movement_viz.py     Visualize how weights move during pre-training
                           (ModelSeries class loads checkpoint sequences)
Init_explore.ipynb         Exploratory notebook: weight distributions at initialization
```

## Usage

**Run masked weight pre-training:**
```bash
python overparameterization/mask_and_train.py \
    --trained_model <path_to_trained_model> \
    --initial_model <path_to_init_model> \
    --dataset babyLM-10M \
    --output_dir <output_path> \
    --mask_strategy movement \
    --cutoff_perc 0.7
```

**Compare GLUE results across runs:**
```bash
python initialization_and_GLUE/compare_glue.py
```

Experiments were run on a SLURM HPC cluster; adapt resource requests and local paths as needed.

## Key Dependencies

```
torch
transformers
datasets
numpy
matplotlib
seaborn
```
