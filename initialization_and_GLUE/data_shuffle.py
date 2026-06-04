#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:59:03 2024

@author: anna
"""
import os
from sklearn.feature_extraction.text import CountVectorizer
import random


folder_loc = "/media/anna/Samsung_T5/Initialization/BabyLM/datasets/babylm_10M/"
tokenizer = CountVectorizer().build_tokenizer()
dataset = dict()

for d in os.listdir(folder_loc):
    if d[-5:] == "train":
        dataset[d] = []
        with open(folder_loc+d, "r") as f:
            for line in f.readlines():
                dataset[d].append(tokenizer(line))
                
corpus = []
for d in dataset:
    for l in dataset[d]:
        corpus.extend(l)

random.shuffle(corpus)            

shuffle_sent_loc = "/media/anna/Samsung_T5/Initialization/BabyLM/datasets/shuffle_sent/"
shuffle_corp_loc = "/media/anna/Samsung_T5/Initialization/BabyLM/datasets/shuffle_corp/"

i = 0
for d in dataset:
    sent_save = []
    corp_save = []
    for s in dataset[d]:
        l = len(s)
        random.shuffle(s)
        sent = " ".join(s)
        corp = " ".join(corpus[i:i+l])
        i+=l
        if l > 1:
            sent_save.append(sent)
        corp_save.append(corp)
    with open(shuffle_sent_loc+d, "w") as sf:
        for s in sent_save:
            sf.write(s+"\n")
    with open(shuffle_corp_loc+d, "w") as cf:
        for c in corp_save:
            cf.write(c+"\n")
        