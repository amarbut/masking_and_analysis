#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:48:42 2024

@author: anna
"""

from transformers import RobertaForMaskedLM, RobertaConfig, RobertaTokenizer, RobertaModel
from tokenizers import ByteLevelBPETokenizer
import os

config = RobertaConfig(hidden_size = 512, 
                       num_attention_heads=4, 
                       num_hidden_layers=4,
                       intermediate_size=2048)
model = RobertaForMaskedLM(config)

start = "/media/anna/Samsung_T5/initialization/BabyLM/train_10M"
paths = [start+"/"+f for f in os.listdir(start)]
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(paths, vocab_size=50265, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model("babylm-base")
model.save_pretrained("babylm-base")

model.num_parameters()


model = RobertaModel.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/babylm-base")
tokenizer = RobertaTokenizer.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/babylm-base")


config = RobertaConfig(hidden_size = 512, 
                       num_attention_heads=1, 
                       num_hidden_layers=1,
                       intermediate_size=512)
model = RobertaForMaskedLM(config)

model.save_pretrained("babylm-test")
start = "/media/anna/Samsung_T5/initialization/BabyLM/datasets/train_10M"
paths = [start+"/"+f for f in os.listdir(start)]
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(paths, vocab_size=50265, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model("scripts/babylm-test")
