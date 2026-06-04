"""
Exploratory prototype: weight masking and custom training with frozen weights.

This file documents the initial prototype of the masking and freezing approach,
developed before the full parameterized implementation in mask_and_train.py.
It is structured for interactive cell-by-cell execution (Spyder / VS Code).

The core abstractions here — `build_freeze_dict` and `FreezeTrainer` — were
subsequently cleaned up and made configurable in mask_and_train.py, which is
the canonical implementation used in the HPC experiments.

The mask overlap analysis at the end (cells 2–3) explores the research question:
do 10M and 100M token models agree on which weights should be masked, under
different masking strategies?
"""

from transformers import Trainer
import torch
from typing import Dict, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


# ---------------------------------------------------------------------------
# Core masking implementation (prototype; see mask_and_train.py for full version)
# ---------------------------------------------------------------------------

def build_freeze_dict(model, mask_strategy, cutoff_perc, init_model):
    """Build dictionaries of zero and freeze masks for encoder weight matrices.

    Parameters
    ----------
    model : RobertaForMaskedLM
        Trained model to mask.
    mask_strategy : str
        One of 'raw_value', 'movement', 'direction'.
    cutoff_perc : float
        Quantile cutoff (0–1) for masking.
    init_model : RobertaForMaskedLM
        Untrained reference model (used for movement and direction strategies).

    Returns
    -------
    dict
        Maps parameter name to boolean mask tensor (True = keep).
    """
    freeze_dict = dict()
    for name, param in model.named_parameters():
        if name[8:15] == "encoder" and name[-6:] == "weight" and name[-16:-7] != "LayerNorm":
            if mask_strategy == "raw_value":
                cutoff = torch.quantile(torch.abs(param), cutoff_perc)
                zero_mask = torch.abs(param) > cutoff

            elif mask_strategy == "movement":
                init_param = init_model.state_dict()[name]
                param_change = torch.abs(init_param - param)
                cutoff = torch.quantile(param_change, cutoff_perc)
                zero_mask = param_change > cutoff

            elif mask_strategy == "direction":
                init_param = init_model.state_dict()[name]
                zero_mask = torch.abs(init_param) - torch.abs(param) > 0

            freeze_dict[name] = zero_mask
    return freeze_dict


class FreezeTrainer(Trainer):
    """Custom Trainer that zeroes gradients for masked weights each step."""

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward()
        for name, param in model.named_parameters():
            if name in freeze_dict:
                param.grad *= freeze_dict[name]
        return loss.detach()


# ---------------------------------------------------------------------------
# Cell 1: Mask overlap analysis — do 10M and 100M models agree on what to mask?
#
# Requires:
#   m_10M  = RobertaForMaskedLM trained on 10M tokens
#   m_100M = RobertaForMaskedLM trained on 100M tokens
#   m_init = untrained RobertaForMaskedLM (random init)
# ---------------------------------------------------------------------------
#%%

# m_10M = RobertaForMaskedLM.from_pretrained("<path to 10M model>")
# m_100M = RobertaForMaskedLM.from_pretrained("<path to 100M model>")
# m_init = RobertaForMaskedLM.from_pretrained("<path to init model>")

# zraw_10M  = build_freeze_dict(m_10M,  "raw_value", 0.7, m_init)
# zraw_100M = build_freeze_dict(m_100M, "raw_value", 0.7, m_init)
# zmove_10M  = build_freeze_dict(m_10M,  "movement", 0.7, m_init)
# zmove_100M = build_freeze_dict(m_100M, "movement", 0.7, m_init)
# zmag_10M   = build_freeze_dict(m_10M,  "magnitude", 0.7, m_init)
# zmag_100M  = build_freeze_dict(m_100M, "magnitude", 0.7, m_init)

# ---------------------------------------------------------------------------
# Cell 2: Compute per-parameter mask overlap percentages
# ---------------------------------------------------------------------------
#%%

# ovlp_perc = {}
# for param in zraw_10M:
#     ovlp_perc[param] = {
#         "raw":       float(torch.sum(zraw_10M[param]  == zraw_100M[param])  / zraw_10M[param].numel()),
#         "movement":  float(torch.sum(zmove_10M[param] == zmove_100M[param]) / zmove_10M[param].numel()),
#         "magnitude": float(torch.sum(zmag_10M[param]  == zmag_100M[param])  / zmag_10M[param].numel()),
#     }
# ovlp_df = pd.DataFrame(ovlp_perc)
# p = "roberta.encoder.layer."
# ovlp_df.columns = [t.removeprefix(p)[:-7] for t in ovlp_df.columns]

# ---------------------------------------------------------------------------
# Cell 3: Visualize mask overlap by layer
# ---------------------------------------------------------------------------
#%%

# f, axes = plt.subplots(4, 3, figsize=(20, 15))
# axes = axes.flatten()
# plt.subplots_adjust(hspace=0.2)
# plt.suptitle("Mask Overlap: 10M vs 100M Models", fontsize=30)
# for idx, i in enumerate(range(0, len(ovlp_df.columns), 6)):
#     layer = ovlp_df.iloc[:, i:i+6].copy()
#     layer.columns = [re.sub(r'^\d{1,2}\.', '', t) for t in layer.columns]
#     layer.T.plot(kind="bar", ax=axes[idx])
#     axes[idx].set_title(f"Layer {idx}", fontsize="xx-large")
#     if idx == 6:
#         axes[idx].set_ylabel("Proportion Mask Overlap", fontsize=25)
#     if idx not in [9, 10, 11]:
#         axes[idx].set_xticks([])
#     if idx != 2:
#         axes[idx].legend().set_visible(False)
# plt.show()
