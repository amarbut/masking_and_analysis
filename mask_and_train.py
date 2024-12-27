#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:02:36 2024

@author: anna
"""

from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, LineByLineTextDataset, RobertaConfig
from transformers import Trainer
import torch
from typing import Dict, Union, Any
from datasets import load_dataset
import argparse
from babylm_baseline_train.datasets import babyLM

def build_freeze_dict(model, mask_strategy, cutoff_perc, init_model):
    freeze_dict = dict()
    zero_dict = dict()
    if cutoff_perc > 0:
        for name, param in model.named_parameters():
            if name[8:15] == "encoder" and name[-6:] == "weight" and name[-16:-7] != "LayerNorm":
                if mask_strategy == "raw_value":
                    cutoff = torch.quantile(torch.abs(param), cutoff_perc)
                    zero_mask = torch.abs(param) > cutoff
                    freeze_mask = zero_mask
                
                elif mask_strategy == "movement":
                    init_param = init_model.state_dict()[name]
                    param_change = torch.abs(param-init_param)
                    cutoff = torch.quantile(param_change, cutoff_perc)
                    zero_mask = param_change > cutoff
                    freeze_mask = zero_mask
                    
                elif mask_strategy == "magnitude":
                    init_param = init_model.state_dict()[name]
                    mag_change = torch.abs(param)-torch.abs(init_param)
                    cutoff = torch.quantile(mag_change, cutoff_perc)
                    zero_mask = mag_change > cutoff
                    freeze_mask = zero_mask
                    
                elif mask_strategy == "direction_mask":
                    #freeze values based on raw value
                    cutoff = torch.quantile(torch.abs(param), cutoff_perc)
                    freeze_mask = torch.abs(param) > cutoff 
                    
                    #zero values based on movement toward zero
                    init_param = init_model.state_dict()[name]
                    #is the final value closer to zero than the initial value
                    zero_mask = torch.abs(init_param)-torch.abs(param) > 0
                    #only zero if also frozen
                    zero_mask = torch.logical_and(torch.logical_not(freeze_mask), torch.logical_not(zero_mask))
                    
                elif mask_strategy == "direction_all":
                    #freeze values based on raw value
                    cutoff = torch.quantile(torch.abs(param), cutoff_perc)
                    freeze_mask = torch.abs(param) > cutoff 
                    
                    #zero all values based on movement toward zero
                    init_param = init_model.state_dict()[name]
                    #is the final value closer to zero than the initial value
                    zero_mask = torch.abs(init_param)-torch.abs(param) > 0
                    
                elif mask_strategy == "random":
                    zero_mask = torch.rand_like(param) > cutoff_perc
                    freeze_mask = zero_mask
                    
                freeze_dict[name] = freeze_mask
                zero_dict[name] = zero_mask
    return zero_dict, freeze_dict

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model', help = 'location of trained model', required = True)
    parser.add_argument('--initial_model', help = 'location of untrained model', required = True)
    parser.add_argument('--dataset', help = 'babyLM-10M, babyLM-100M', required = True)
    parser.add_argument('--output_dir', help = 'save location', required = True)
    parser.add_argument('--epochs', help = 'number of epochs to train', required = False, type = int, default = 20)
    parser.add_argument('--mask_strategy', help = 'raw_value, movement, direction_mask, direction_all', required = True)
    parser.add_argument('--cutoff_perc', help = '% of weights to remove', type = float, required = True)
    parser.add_argument('--batch_size', help = 'training batch size', required = False, type = int, default = 128)
    parser.add_argument('--lr', help = 'training learning rate', required = False, type = float, default = 1e-4)
    parser.add_argument('--wgt_decay', help = 'training weight decay', required = False, type = float, default = 0.1)
    parser.add_argument('--checkpoint_model', help = 'location of checkpoint', required = False, default = None)
    parser.add_argument('--model_10M', help = 'location of 10M token model', required = False, default = None)
    parser.add_argument('--cutoff_10M', help = '% of weights removed from first mask', type = float, required = False, default = 0)
    parser.add_argument('--save_strategy', help = 'epoch or steps', required = False, default = "steps")
    parser.add_argument('--max_save', help = 'max checkpoints to save when saving by steps', required = False, type = int, default = None)
    parser.add_argument('--save_steps', help = 'number of iterations for checkpoints if save_strategy=steps', required = False, type = int, default = 10000)
    
        
    kwargs = vars(parser.parse_args())
    
    print("loading models")
    model = RobertaForMaskedLM.from_pretrained(kwargs['trained_model'])
    init =  RobertaForMaskedLM.from_pretrained(kwargs['initial_model'])
    if kwargs['checkpoint_model'] is not None:
        ckpt = RobertaForMaskedLM.from_pretrained(kwargs['checkpoint_model'])
    else:
        ckpt = None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    init.to(device)
    
    #apply mask to init model if running through a second mask    
    if kwargs["model_10M"] is not None:
        print("applying 10M token mask")
        m_10M = RobertaForMaskedLM.from_pretrained(kwargs['model_10M'])
        m_10M.to(device)
        z_dict, f_dict = build_freeze_dict(init, kwargs['mask_strategy'], kwargs['cutoff_10M'], m_10M)
        for name, param in init.named_parameters():
            if name in z_dict:
                p = param.detach()
                p *= z_dict[name]
                param = p
    
    print("building mask")
    zero_dict, freeze_dict = build_freeze_dict(model, kwargs['mask_strategy'], kwargs['cutoff_perc'], init)
    
    print("applying mask")
    for name, param in init.named_parameters():
        if name in zero_dict:
            p = param.detach()
            p *= zero_dict[name]
            param = p
    
    print("prepping tokenizer, collator, and training args")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    datacollator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=True,
                    mlm_probability=0.15,
                    )
    
    training_args = TrainingArguments(
        output_dir= kwargs['output_dir'],
        overwrite_output_dir=True,
        num_train_epochs= kwargs['epochs'],
        per_device_train_batch_size= kwargs['batch_size'],
        learning_rate= kwargs['lr'],
        weight_decay= kwargs['wgt_decay'],
        save_strategy= kwargs['save_strategy'],
        save_steps=kwargs['save_steps'],
        save_total_limit= kwargs['max_save'] if kwargs['save_strategy']=='steps' else None,
        prediction_loss_only=True)
    
    # dataset = LineByLineTextDataset(
    #     tokenizer = tokenizer,
    #     file_path= kwargs['dataset'],
    #     block_size = 128)
    
    print("building dataset")

    if kwargs['dataset'] == 'babyLM-10M':
        dataset = babyLM.get_babyLM_10M(seq_len=128, tokenizer=None, just_dataset=False)
    elif kwargs['dataset'] == 'babyLM-100M':
        dataset = babyLM.get_babyLM_100M(seq_len=128, tokenizer=None, just_dataset=False)
    elif kwargs['dataset'] == 'babyLM-1B':
        dataset = babyLM.get_babyLM_1B(seq_len=128, tokenizer=None, just_dataset=False)
    else:
        raise NotImplementedError("Unknown dataset")
    
    trainer = FreezeTrainer(
        model = ckpt if ckpt is not None else init,
        args = training_args,
        data_collator=datacollator,
        train_dataset=dataset)
    
    print("begin training")
    trainer.train()
