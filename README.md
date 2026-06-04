# masking_and_analysis

Exploratory analysis of weight distributions and masking in pre-trained transformer language models, connected to Chapter 4 of:

> **Searching for Nooks and Crannies: Geometric and Mechanistic Perspectives on Transformer Language Model Interpretability**  
> Anna C. Marbut. PhD dissertation, University of Montana (2026).

This work investigates how pre-training scale and hyperparameters shape the distribution of weights across transformer encoder layers, and whether lottery-ticket-style masking can be used to re-train smaller models with masked weight initialization.

## Research Questions

- How do weight distributions differ between models trained on 10M vs. 100M tokens?
- Do different pre-training scales agree on which weights are "important" (as measured by magnitude, movement, and direction from initialization)?
- Can a model trained from masked weights (retaining a subset of weights from a pre-trained model) recover competitive downstream performance?

## Repository Structure

```
mask_and_train.py          Main training script — parameterized HPC-compatible implementation
                           of masked weight pre-training. Accepts all settings via argparse.

masking.py                 Exploratory prototype of the masking approach. Documents the initial
                           build_freeze_dict() and FreezeTrainer implementations, and the
                           mask overlap analysis comparing 10M and 100M models.

weight_movement_viz.py     Visualization of weight movement across training checkpoints.
                           ModelSeries class loads a series of checkpoints and computes
                           statistics on how weights move during pre-training.

full_model_weight_dist.py  Compare global weight distributions across pre-trained model families
                           (RoBERTa, BERT, GPT-2, T5).

unmask_mask_weights.py     Analysis of which weights survive masking and how their distributions
                           change after masked re-training.

BabyLM_mask.py             BabyLM-specific masking experiments.
roberta_pretrain.py        RoBERTa pre-training utilities.

mask_train.slurm           SLURM job script for HPC cluster runs.
run_glue.sh                Script to run GLUE evaluation on trained models.

Freeze_trial.ipynb         Notebook exploring early freeze/mask trials.
Init_explore.ipynb         Notebook exploring weight distributions at initialization.
unmask_mask_wgt_explore.ipynb  Notebook exploring weight changes under masking.
```

## Masking Strategies

`mask_and_train.py` supports five masking strategies:

| Strategy | Masks weights with... |
|---|---|
| `raw_value` | Smallest absolute weight values |
| `movement` | Weights that changed least from initialization |
| `magnitude` | Weights with smallest magnitude change from initialization |
| `direction_mask` | Weights that moved toward zero (partial zeroing) |
| `direction_all` | Weights that moved toward zero (full zeroing) |

## Usage

```bash
python mask_and_train.py \
    --trained_model <path_to_trained_model> \
    --initial_model <path_to_init_model> \
    --dataset babyLM-10M \
    --output_dir <output_path> \
    --mask_strategy movement \
    --cutoff_perc 0.7
```

## Key Dependencies

```
torch
transformers
datasets
numpy
matplotlib
seaborn
```
