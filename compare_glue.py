#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:35:47 2024

@author: anna
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re

#alternative models run
glue_folder = "/media/anna/Samsung_T5/Initialization/glue_results/all_results/"

#masked models run
#glue_folder = "/media/anna/Samsung_T5/Initialization/glue_results/all_mask_results/"
# glue_folder = "/media/anna/Samsung_T5/Initialization/glue_results/all_mask_results_qqpacc/"

glue_dict = dict()
for m in os.listdir(glue_folder):
    m_name = m[:-5]
    r = json.load(open(glue_folder+m, "r"))
    glue_dict[m_name]=r

glue_tasks = [#'cola',
             'mnli',
             #'mnli-mm',
             'mrpc',
             'qnli',
             'qqp',
             'rte',
             'sst2']
    
for m in glue_dict:
    for e in glue_dict[m]:
        glue_dict[m][e]["glue_avg"] = np.mean([glue_dict[m][e][t]for t in glue_dict[m][e] if t in glue_tasks])

# for alt models    
model_list = ["rt_shuffle-index",
              "rt_shuffle-sentence",
              "rt_shuffle-corpus",
              "rt_ascii",
              "rerun_rt_rand",
              "roberta-baby",
              "rerun_rand",
              "original_rand",
              "stand_norm",
              "rt_rand"]

#for mask models
# model_list = ["Roberta_10M",
#               "Roberta_10M_2",
#               "Roberta_10M_3",
#               "Roberta_10M_init2",
#               "Roberta_10M_init3",
#               "Magnitude_10M",
#               "Magnitude_10M_2",
#               "Magnitude_10M_3",
#               "Magnitude_10M_init2",
#               "Magnitude_10M_init3",
#               # "Movement_10M",
#               # "Direction_all_10M",
#               # "Direction_mask_10M",
#               "Raw_10M",
#               "Raw_10M_2",
#               "Raw_10M_3",
#               "Raw_10M_init2",
#               "Raw_10M_init3",
#               ]

#visualize superglue performance curves
# plt.figure(figsize=(12,8))
# plt.xticks(list(range(0,21)))
# plt.xlabel("Epoch")
# plt.ylabel("Avg (Super)GLUE Score")
# plt.title("(Super)GLUE Performance during pre-training on BabyLM 10M")
# for m in model_list:
#     x = []
#     y = []
#     for e in glue_dict[m]:
#         e_int = int(e[3:])
#         x.append(e_int)
#         y.append(glue_dict[m][e]["average"])
#     p = list(zip(x,y))
#     p.sort()
#     x_sort, y_sort = zip(*p)
#     plt.plot(x_sort,y_sort, label = m)

# plt.legend()    
# plt.show()

#exclude super tasks
cmap = plt.get_cmap('tab20')
plt.figure(figsize=(12,8))
plt.xticks(list(range(0,21)))
plt.xlabel("Epoch")
plt.ylabel("Avg GLUE Score")
plt.title("GLUE Performance during pre-training on BabyLM 10M")
for idx, m in enumerate(model_list):
    x = []
    y = []
    for e in glue_dict[m]:
        e_int = int(e[3:])
        # e_int = int(float(re.search("epoch_(.+)",e).group(1).replace("_", ".")))
        x.append(e_int)
        y.append(glue_dict[m][e]["glue_avg"])
    p = list(zip(x,y))
    p.sort()
    x_sort, y_sort = zip(*p)
    plt.plot(x_sort,y_sort, label = m, color = cmap(idx))

plt.legend()    
plt.show()


#%%
#explore results
import pandas as pd
order = ["hf_"+str(i) for i in range(0,21)]

rand = pd.DataFrame(glue_dict["rt_rand"])[order]
asci = pd.DataFrame(glue_dict["rt_ascii"])[order]
sent = pd.DataFrame(glue_dict["rt_shuffle-sentence"])[order]
norm = pd.DataFrame(glue_dict["normal_init"])[order]

#%%
#explore weird rand performance
from transformers import RobertaForMaskedLM

rand1 = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/rt_rand/hf_4")
rand2 = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/rt_rand/hf_6")
rand3 = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/rt_rand/hf_11")

params1 = []
for name, param in rand1.named_parameters():
    if name[8:15] == "encoder" and name[-6:] == "weight":
        params1.append(param)
        
params2 = []
for name, param in rand2.named_parameters():
    if name[8:15] == "encoder" and name[-6:] == "weight":
        params2.append(param)    
        
params3 = []
for name, param in rand3.named_parameters():
    if name[8:15] == "encoder" and name[-6:] == "weight":
        params3.append(param)  
        
        
[params1[i]==params3[i] for i in range(len(params1))]

#%%
new_rand = pd.DataFrame({"hf_2": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.38461539149284363, "average": 0.41391131118346736}, "hf_19": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_4": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_3": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.38461539149284363, "average": 0.4553024423018141}, "hf_12": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_11": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_18": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_8": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_13": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_7": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_1": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.31611067056655884, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.38461539149284363, "average": 0.4133565382429184}, "hf_5": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.42082986055762006}, "hf_17": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_16": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_10": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_15": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_9": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_14": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_20": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}, "hf_6": {"mnli": 0.33190709352493286, "rte": 0.46043166518211365, "sst2": 0.5183486342430115, "mrpc": 0.8104956268221575, "multirc": 0.5754950642585754, "mnli-mm": 0.32221317291259766, "cola": 0.0, "qqp": 0.0, "boolq": 0.6403669714927673, "qnli": 0.5091508030891418, "wsc": 0.6153846383094788, "average": 0.4348903336213433}})
new_rand = rand[["hf_1", "hf_6", "hf_14", "hf_20"]]
