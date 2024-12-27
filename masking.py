#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:51:33 2024

@author: anna
"""

#import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, LineByLineTextDataset

#%%
#daniel's code for freezing weights within a standard training loop
# m_10M = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_10M/hf_20")
# m_100M = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_100M/hf_20")
# m_base = RobertaForMaskedLM.from_pretrained("roberta-base")

# # Note that we're ignoring the bias for this example...
# layer = model.roberta.encoder.layer[0].attention.self.query

# # Create masks
# zero_mask = torch.abs(layer.weight.data) > 0.02
# freeze_mask = torch.rand_like(zero_mask) > 0.5

# # we want to freeze zero_masked values also
# freeze_mask = torch.logical_and(freeze_mask, zero_mask)

# # Zero mask
# layer.weight.data *= zero_mask

# for step in training_steps:
# 	# do training stuff

# 	loss.backward()
# 	# freeze mask
# 	layer.weight.grad *= freeze_mask
# 	op.step()
# 	op.zero_grad()

#%%
#trial training with custom trainer/masking/freezing

from transformers import Trainer
import torch
from typing import Dict, Union, Any
from datasets import load_dataset

def build_freeze_dict(model, mask_strategy, cutoff_perc, init_model):
    freeze_dict = dict()
    for name, param in model.named_parameters():
        if name[8:15] == "encoder" and name[-6:] == "weight" and name[-16:-7] != "LayerNorm":
            if mask_strategy == "raw_value":
                cutoff = torch.quantile(torch.abs(param), cutoff_perc)
                zero_mask = torch.abs(param) > cutoff
            
            elif mask_strategy == "movement":
                init_param = init_model.state_dict()[name]
                param_change = torch.abs(init_param-param)
                cutoff = torch.quantile(param_change, cutoff_perc)
                zero_mask = torch.abs(param_change) > cutoff
                
            elif mask_strategy == "direction":
                init_param = init_model.state_dict()[name]
                #is the final value closer to zero than the initial value
                zero_mask = torch.abs(init_param)-torch.abs(param) > 0
            freeze_dict[name] = zero_mask
    return freeze_dict

class FreezeTrainer(Trainer):
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]):
        #custom training step that includes zeroing the gradient for masked weights
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward()
        
        for name, param in model.named_parameters():
            if name in freeze_dict:
                param.grad *= freeze_dict[name] #hard-coded, not sure how to pass a new argument through the Trainer...
        return loss.detach()

#build out tiny model
config = m_base.config
config.num_attention_heads = 2
config.num_hidden_layers= 2
m = RobertaForMaskedLM(config = config)
m2 = RobertaForMaskedLM(config = config)

#build out masks
freeze_dict = build_freeze_dict(m, "raw_value", 0.7, m2)
freeze_dict_move = build_freeze_dict(m, "movement", 0.7, m2)
freeze_dict_dir = build_freeze_dict(m, "direction", 0.7, m2)

#apply mask to weights
for name, param in m.named_parameters():
    if name in freeze_dict:
        p = param.detach()
        p *= freeze_dict[name]
        param = p
        
#double check that masking worked        
# for name, param in m.named_parameters():
#     if name in freeze_dict:
#         print(param)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
datacollator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=0.15,
                )

training_args = TrainingArguments(
    output_dir="./freeze_trial",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True)

# can't use babyLM library in spyder d/t python version
# dataset = load_dataset(path = "./baseline-pretraining/src/babylm_baseline_train/datasets/babyLM_for_hf.py",
#                        name = 'babyLM-10M',
#                        split = "train")

dataset = LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path="/media/anna/Samsung_T5/Initialization/BabyLM/datasets/babyLM_10M/aochildes.train",
    block_size = 128)



trainer = FreezeTrainer(
    model = m,
    args = training_args,
    data_collator=datacollator,
    train_dataset=dataset)

trainer.train()
#%%
import numpy as np

m_10M = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_10M/hf_20")
m_100M = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_100M/hf_20")
m_init = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_init")


zraw_10M, fraw_10M = build_freeze_dict(m_10M, "raw_value", 0.7, m_init)
zraw_100M, fraw_100M = build_freeze_dict(m_100M, "raw_value", 0.7, m_init)

ovlp_raw_10M100M = dict()
for param in zraw_10M:
    ovlp = zraw_10M[param] == zraw_100M[param]
    ovlp_raw_10M100M[param] = ovlp

zmove_10M, fmove_10M = build_freeze_dict(m_10M, "movement", 0.7, m_init)
zmove_100M, fmove_100M = build_freeze_dict(m_100M, "movement", 0.7, m_init)

ovlp_move_10M100M = dict()
for param in zmove_10M:
    ovlp = zmove_10M[param] == zmove_100M[param]
    ovlp_move_10M100M[param] = ovlp
    
zmag_10M, fmag_10M = build_freeze_dict(m_10M, "magnitude", 0.7, m_init)
zmag_100M, fmag_100M = build_freeze_dict(m_100M, "magnitude", 0.7, m_init)

ovlp_mag_10M100M = dict()
for param in zmag_10M:
    ovlp = zmag_10M[param] == zmag_100M[param]
    ovlp_mag_10M100M[param] = ovlp
    
ovlp_perc = dict()
for param in ovlp_raw_10M100M:
    ovlp_perc[param] = dict()
    r = ovlp_raw_10M100M[param]
    r_perc = torch.sum(r)/r.numel()
    ovlp_perc[param]["raw"] = float(r_perc)
    
    mo = ovlp_move_10M100M[param]
    mo_perc = torch.sum(mo)/mo.numel()
    ovlp_perc[param]["movement"] = float(mo_perc)
    
    ma = ovlp_mag_10M100M[param]
    ma_perc = torch.sum(ma)/ma.numel()
    ovlp_perc[param]["magnitude"] = float(ma_perc)

ovlp_df = pd.DataFrame(ovlp_perc)

p = "roberta.encoder.layer."
colnames = [t.removeprefix(p)[:-7] for t in list(ovlp_perc.keys())]
ovlp_df.columns=colnames

f, axes = plt.subplots(4,3, figsize = (20,15))
axes = axes.flatten()
#f.tight_layout()
plt.subplots_adjust(hspace = 0.2)
plt.suptitle("Mask Overlap 10M to 100M Models", fontsize = 30)

for idx, i in enumerate(range(0,len(ovlp_df.columns),6)):
    layer = ovlp_df.iloc[:,i:i+6]
    layer.columns = [re.sub(r'^\d{1,2}\.','',t) for t in layer.columns]
    layer = layer.T
    layer.plot(kind = "bar", ax = axes[idx])
    axes[idx].set_title(f"Layer {idx}", fontsize = "xx-large")
    if idx == 6:
        axes[idx].set_ylabel("Proportion Mask Overlap", fontsize = 25)
    if idx not in [9,10,11]:
        axes[idx].set_xticks([])
    if idx != 2:
        axes[idx].legend().set_visible(False)
    
    
    
    
