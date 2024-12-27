#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:23:36 2024

@author: anna
"""
from transformers import RobertaForMaskedLM
import numpy as np

def mask_weights(model, cutoff = 0.7):
    masks = dict()
    for name, param in model.named_parameters():
        
        if name[8:15] == "encoder" and name[-6:] == "weight":
            print(name)
            # print(param)
            
            param_np = param.detach().numpy()
            if param_np.ndim == 1:
                param_np = param_np[np.newaxis,:]
            layer_mask = np.ones(param_np.shape)
            for idx, n in enumerate(param_np):
                c = np.sum([1 for i in n if i >=cutoff])
                #print(c)
                if c == 0:
                    layer_mask[idx] = np.zeros(n.shape)
            masks[name] = layer_mask
    return masks
            
m_baby = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/roberta-baby/hf_20")

baby_mask = mask_weights(m_baby)

m_full = RobertaForMaskedLM.from_pretrained("roberta-base")
full_mask = mask_weights(m_full)

for param in full_mask:
    s = np.sum(full_mask[param])
    if s > 0 and param[-16:] != "LayerNorm.weight":
        
        print(param, ":", str(s/768))

for param in baby_mask:
    s = np.sum(baby_mask[param])
    if s > 0:
        print(param, ":", str(s/768))
        
#%%
#largest final value mask

def large_mask(model, percent = 0.5):
    masks = dict()
    for name, param in model.named_parameters():
        
        if name[8:15] == "encoder" and name[-6:] == "weight" and param[-16:] != "LayerNorm.weight":
            print(name)
            # print(param)
            
            param_np = param.detach().numpy()
            if param_np.ndim == 1:
                param_np = param_np[np.newaxis,:]
            layer_mask = np.ones(param_np.shape)
            mx = np.max(param_np)
            #TODO: how to take top x% weights? quantile function?
            for idx, n in enumerate(param_np):
                
            masks[name] = layer_mask
    return masks