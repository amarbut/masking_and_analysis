#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:43:23 2024

@author: anna
"""

from transformers import BertForMaskedLM, RobertaForMaskedLM

m1 = BertForMaskedLM.from_pretrained("aajrami/bert-rand-base")
m2 = RobertaForMaskedLM.from_pretrained("aajrami/bert-rand-base")


#%%
# confirm ratios of test (validation) dataset labels
from datasets import load_dataset
import json
import numpy as np
from collections import Counter

glue_tasks = list(rand.index.values)

valid_hists = dict()

for t in glue_tasks:
    if t != "average":
        
        labels = []
        fname = f"/home/anna/Documents/PhD/initialization/evaluation-pipeline-2024/evaluation_data/glue_filtered/{t}.valid.jsonl"
        with open(fname, "r") as f:
            for l in f.readlines():
                d = json.loads(l)
                label = d["label"]
                labels.append(label)    
        total = len(labels)
        h = {label:count/total for label, count in Counter(labels).items()}
        valid_hists[t] = h
     
        
#%%
# explore embedding distribution of baby rand and ascii models

m_rand = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/rand/hf_20")
m_ascii = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/ascii/hf_20")
m_sent = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/shuffle-sentence/hf_20")
m_baby = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/roberta-baby/hf_20")
seq = pickle.load(open("/media/anna/Samsung_T5/manifolds/sample_sequences.pkl", "rb"))

emb_rand = build_samples(seq, m_rand, tokenizer = "roberta-base", model_type = "roberta")
emb_ascii = build_samples(seq, m_ascii, tokenizer = "roberta-base", model_type = "roberta")
emb_sent = build_samples(seq, m_sent, tokenizer = "roberta-base", model_type = "roberta")
emb_baby = build_samples(seq, m_baby, tokenizer = "roberta-base", model_type = "roberta")

EEE_rand = EEE(emb_rand)
EEE_ascii = EEE(emb_ascii)
EEE_sent = EEE(emb_sent)
EEE_baby = EEE(emb_baby)

#%%
#explore nn distribution of baby rand and ascii models
import matplotlib.pyplot as plt
from collections import Counter

emb_rand = pickle.load(open("/media/anna/Samsung_T5/Initialization/rand_sample_space.pkl", "rb"))
emb_ascii = pickle.load(open("/media/anna/Samsung_T5/Initialization/ascii_sample_space.pkl", "rb"))
emb_sent = pickle.load(open("/media/anna/Samsung_T5/Initialization/sent_sample_space.pkl", "rb"))
emb_baby = pickle.load(open("/media/anna/Samsung_T5/Initialization/baby_sample_space.pkl", "rb"))


def nn_dist(emb):
    d = len(emb[0])
    emb = np.array(emb)
    index = faiss.IndexFlatL2(d)
    index.add(emb)
    D,I = index.search(emb, 2)
    return(D,I)

nn_rand = nn_dist(emb_rand)
nn_ascii = nn_dist(emb_ascii)
nn_sent = nn_dist(emb_sent)
nn_baby = nn_dist(emb_baby)

nn_dict = {"nn_rand": nn_rand,
           "nn_ascii":nn_ascii,
           "nn_sent":nn_sent,
           "nn_baby":nn_baby}

pickle.dump(nn_dict, open("nn_distances_dict", "wb"))

index_rand = Counter([j[1] for j in nn_rand[1]])
index_ascii = Counter([j[1] for j in nn_ascii[1]])
index_sent = Counter([j[1] for j in nn_sent[1]])
index_baby = Counter([j[1] for j in nn_baby[1]])

dist_rand = Counter([j[1] for j in nn_rand[0]])
dist_ascii = Counter([j[1] for j in nn_ascii[0]])
dist_sent = Counter([j[1] for j in nn_sent[0]])
dist_baby = Counter([j[1] for j in nn_baby[0]])


f, axes = plt.subplots(2,2, figsize = (10,7))
#f.tight_layout()
plt.subplots_adjust(hspace = 0.25)   
plt.suptitle("NN Distances", fontsize = "xx-large")
axes[0,0].set_title("Rand: EEE=0.9974")
axes[0,0].hist([j[1] for j in nn_rand[0]], bins = 30)

axes[1,0].set_title("ASCII: EEE=0.9946")
axes[1,0].hist([j[1] for j in nn_ascii[0]], bins = 30)

axes[0,1].set_title("Shuffle-Sentence EEE=0.80")
axes[0,1].hist([j[1] for j in nn_sent[0]], bins = 30)

axes[1,1].set_title("RoBERTa-Baby EEE=0.7903")
axes[1,1].hist([j[1] for j in nn_baby[0]], bins = 30)

#%%
#explore model weight distributions
m_rand = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/rand/hf_20")
m_ascii = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/ascii/hf_20")
m_sent = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/shuffle-sentence/hf_20")
m_baby = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/roberta-baby/hf_20")
m_full = RobertaForMaskedLM.from_pretrained("roberta-base")

params_rand = []
flat_rand = []
for l in list(range(12)):
    layer = []    
    for name, param in m_rand.named_parameters():
        if name[8:15] == "encoder" and name[-6:] == "weight" and (name[22:24] == str(l)+"." or name[22:25] == str(l)+"."):
            print(name)
            # print(param)
            param_np = param.detach().numpy()
            if param_np.ndim > 1:
                layer.extend(param_np)
            else:
                layer.append(param_np)
    params_rand.append(layer)
for layer in params_rand:
    flattened = [i for p in layer for i in p]
    flat_rand.append(flattened)
    
params_ascii = []
flat_ascii = []
for l in list(range(12)):
    layer = []    
    for name, param in m_ascii.named_parameters():
        if name[8:15] == "encoder" and name[-6:] == "weight" and (name[22:24] == str(l)+"." or name[22:25] == str(l)+"."):
            print(name)
            # print(param)
            param_np = param.detach().numpy()
            if param_np.ndim > 1:
                layer.extend(param_np)
            else:
                layer.append(param_np)
    params_ascii.append(layer)
for layer in params_ascii:
    flattened = [i for p in layer for i in p]
    flat_ascii.append(flattened)
    
params_sent = []
flat_sent = []
for l in list(range(12)):
    layer = []    
    for name, param in m_sent.named_parameters():
        if name[8:15] == "encoder" and name[-6:] == "weight" and (name[22:24] == str(l)+"." or name[22:25] == str(l)+"."):
            print(name)
            # print(param)
            param_np = param.detach().numpy()
            if param_np.ndim > 1:
                layer.extend(param_np)
            else:
                layer.append(param_np)
    params_sent.append(layer)
for layer in params_sent:
    flattened = [i for p in layer for i in p]
    flat_sent.append(flattened)
    
params_baby = []
flat_baby = []
for l in list(range(12)):
    layer = []    
    for name, param in m_baby.named_parameters():
        if name[8:15] == "encoder" and name[-6:] == "weight" and (name[22:24] == str(l)+"." or name[22:25] == str(l)+"."):
            print(name)
            # print(param)
            param_np = param.detach().numpy()
            if param_np.ndim > 1:
                layer.extend(param_np)
            else:
                layer.append(param_np)
    params_baby.append(layer)
for layer in params_baby:
    flattened = [i for p in layer for i in p]
    flat_baby.append(flattened)
    
params_full = []
flat_full = []
for l in list(range(12)):
    layer = []    
    for name, param in m_full.named_parameters():
        if name[8:15] == "encoder" and name[-6:] == "weight" and (name[22:24] == str(l)+"." or name[22:25] == str(l)+"."):
            print(name)
            # print(param)
            param_np = param.detach().numpy()
            if param_np.ndim > 1:
                layer.extend(param_np)
            else:
                layer.append(param_np)
    params_full.append(layer)
for layer in params_full:
    flattened = [i for p in layer for i in p]
    flat_full.append(flattened)
 
import seaborn as sns

# #def weight_hist(params):
# for layer in params_rand:
#     flattened = [i for p in layer for i in p]
#     #hist = np.histogram(flattened, bins = 30)
    
#     sns.kdeplot(flattened, label = str(idx))
#     plt.show()
#         #plt.bar(hist[1][:-1], height=hist[0], width=np.diff(hist[1]), align = "edge",alpha = 0.5, label = str(idx))
#         #plt.hist(flattened, bins = 30, alpha = 0.5, label = str(idx))
#     #plt.legend()
#     plt.show()
    
# weight_hist(params_rand)

def weight_hist(flattened_params, model_name, log = False, filter_high = False):
    cmap = plt.get_cmap('tab20')
    for idx, layer in enumerate(flattened_params):
        #sns.kdeplot(layer, label = str(idx))
        # if log == True:
        #     layer = np.log1p(layer)
        if filter_high == True:
            layer = [i for i in layer if i < 0.5]
        counts, edges = np.histogram(layer, bins = 100)
        if log == True:
            counts = np.log1p(counts)
        centers = (edges[:-1]+edges[1:])/2
        plt.plot(centers, counts, label = str(idx), color = cmap(idx))
        #plt.bar(hist[1][:-1], height=hist[0], width=np.diff(hist[1]), align = "edge",alpha = 0.5, label = str(idx))
    plt.title(model_name)
    if log == True:
        plt.ylabel("Log Counts")
    else:
        plt.ylabel("Counts")
    plt.xlabel("Weight Value")
    plt.legend(title = "Layer")
    plt.show()
    
def vector_norm_hist(params, model_name, log = False):
    for idx, layer in enumerate(params):
        layer_norms = [np.linalg.norm(i) for i in layer]
        counts, edges = np.histogram(layer_norms, bins = 100)
        if log == True:
            counts = np.log1p(counts)
        centers = (edges[:-1]+edges[1:])/2
        plt.plot(centers, counts, label = str(idx))
        #plt.bar(hist[1][:-1], height=hist[0], width=np.diff(hist[1]), align = "edge",alpha = 0.5, label = str(idx))
    plt.title(model_name)
    if log == True:
        plt.ylabel("Log Counts")
    else:
        plt.ylabel("Counts")
    plt.xlabel("Node Vector Norm")
    plt.legend(title = "Layer")
    plt.show()

weight_hist(flat_rand, "Random Digit", log = True, filter_high = True)
weight_hist(flat_ascii, "ASCII", log = True, filter_high = True)
weight_hist(flat_sent, "Shuffled Sentence", log = True, filter_high = True)
weight_hist(flat_baby, "RoBERTa BabyLM", log = True, filter_high = True)
weight_hist(flat_full, "RoBERTa-Base", log = True, filter_high = True)

vector_norm_hist(params_rand, "Random Digit", log = True)
vector_norm_hist(params_ascii, "ASCII", log = True)
vector_norm_hist(params_sent, "Shuffled Sentence", log = True)
vector_norm_hist(params_baby, "RoBERTa BabyLM", log = True)
vector_norm_hist(params_full, "RoBERTa-Base", log = True)

def node_wgt_count(params, model_name, log = False):
    all_counts = []
    for idx, layer in enumerate(params):
        wgt_counts = []
        for n in layer:
            c = np.sum([1 for i in n if i >=0.7])
            wgt_counts.append(c)
        ctr = Counter(wgt_counts)
        all_counts.append(ctr)
        keys = [str(k) for k in ctr.keys()]
        hts = [np.log1p(h) for h in ctr.values()]
        plt.bar(keys, hts, alpha = 0.5, label = str(idx))
        # counts, edges = np.histogram(wgt_counts)
        # if log == True:
        #     counts = np.log1p(counts)
        # centers = (edges[:-1]+edges[1:])/2
        # plt.plot(centers, counts, label = str(idx))
        #plt.bar(hist[1][:-1], height=hist[0], width=np.diff(hist[1]), align = "edge",alpha = 0.5, label = str(idx))
    plt.title(model_name)
    if log == True:
        plt.ylabel("Log Node Counts")
    else:
        plt.ylabel("Node Counts")
    plt.xlabel("Weight Values >= 0.7")
    plt.legend(title = "Layer")
    plt.show()
    return all_counts
            
wgt_ct_rand = node_wgt_count(params_rand, "Random Digit", log = True)
wgt_ct_ascii = node_wgt_count(params_ascii, "ASCII", log = True)
wgt_ct_sent = node_wgt_count(params_sent, "Shuffled Sentence", log = True)
wgt_ct_baby = node_wgt_count(params_baby, "RoBERTa BabyLM", log = True)
wgt_ct_full = node_wgt_count(params_full, "RoBERTa-Base", log = True)

