#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:39:10 2024

@author: anna
"""
from transformers import RobertaForMaskedLM
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

m_10M = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_10M/hf_20")
m_100M = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_100M/hf_20")
m_base = RobertaForMaskedLM.from_pretrained("roberta-base")

params_10M = []
flat_10M = []
for l in list(range(12)):
    layer = []    
    for name, param in m_10M.named_parameters():
        if name[8:15] == "encoder" and name[-6:] == "weight" and name[-16:-7] != "LayerNorm" and (name[22:24] == str(l)+"." or name[22:25] == str(l)+"."):
            print(name)
            # print(param)
            param_np = param.detach().numpy()
            if param_np.ndim > 1:
                layer.extend(param_np)
            else:
                layer.append(param_np)
    params_10M.append(layer)
for layer in params_10M:
    flattened = [i for p in layer for i in p]
    flat_10M.append(flattened)
    
params_100M = []
flat_100M = []
for l in list(range(12)):
    layer = []    
    for name, param in m_100M.named_parameters():
        if name[8:15] == "encoder" and name[-6:] == "weight" and name[-16:-7] != "LayerNorm"  and (name[22:24] == str(l)+"." or name[22:25] == str(l)+"."):
            print(name)
            # print(param)
            param_np = param.detach().numpy()
            if param_np.ndim > 1:
                layer.extend(param_np)
            else:
                layer.append(param_np)
    params_100M.append(layer)
for layer in params_100M:
    flattened = [i for p in layer for i in p]
    flat_100M.append(flattened)
    
params_base = []
flat_base = []
for l in list(range(12)):
    layer = []    
    for name, param in m_base.named_parameters():
        if name[8:15] == "encoder" and name[-6:] == "weight" and name[-16:-7] != "LayerNorm"  and (name[22:24] == str(l)+"." or name[22:25] == str(l)+"."):
            print(name)
            # print(param)
            param_np = param.detach().numpy()
            if param_np.ndim > 1:
                layer.extend(param_np)
            else:
                layer.append(param_np)
    params_base.append(layer)
for layer in params_base:
    flattened = [i for p in layer for i in p]
    flat_base.append(flattened)
    
#%%    
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
            counts = np.log10([i+1 for i in counts])
        centers = (edges[:-1]+edges[1:])/2
        plt.plot(centers, counts, label = str(idx), color = cmap(idx))
        #plt.bar(hist[1][:-1], height=hist[0], width=np.diff(hist[1]), align = "edge",alpha = 0.5, label = str(idx))
    plt.title(model_name)
    if log == True:
        plt.ylabel("Log10 Counts")
    else:
        plt.ylabel("Counts")
    plt.xlabel("Weight Value")
    plt.legend(title = "Layer")
    plt.show()
    
weight_hist(flat_10M, "Roberta_10M", log = True)
weight_hist(flat_100M, "Roberta_100M", log = True)
weight_hist(flat_base, "Roberta_base", log = True)

def node_wgt_count(params, model_name, log = False):
    all_counts = []
    for idx, layer in enumerate(params):
        wgt_counts = []
        for n in layer:
            c = np.sum([1 for i in n if i >0.1])
            wgt_counts.append(c)
        ctr = Counter(wgt_counts)
        all_counts.append(ctr)
        keys = [str(k) for k in ctr.keys()]
        hts = [np.log1p(h) for h in ctr.values()]
        # plt.bar(keys, hts, alpha = 0.5, label = str(idx))
    #     counts, edges = np.histogram(wgt_counts)
    #     if log == True:
    #         counts = np.log10([i+1 for i in counts])
    #     centers = (edges[:-1]+edges[1:])/2
    #     plt.plot(centers, counts, label = str(idx))
    #     #plt.bar(hist[1][:-1], height=hist[0], width=np.diff(hist[1]), align = "edge",alpha = 0.5, label = str(idx))
    # plt.title(model_name)
    # if log == True:
    #     plt.ylabel("Log10 Node Counts")
    # else:
    #     plt.ylabel("Node Counts")
    # plt.xlabel("Weight Values >= 0.7")
    # plt.legend(title = "Layer")
    # plt.show()
    return all_counts

wgt_ct_10M = node_wgt_count(params_10M, "Roberta_10M", log = True)

#%%
#build out dictionaries of weight counts by node/layer over 0.1 in abs final value
wgt_dict_10M = dict()
for i in range(12):
    wgt_dict_10M[f"layer_{i}"] = dict()
    for name, param in m_10M.named_parameters():
        if name[8:15] == "encoder" and name[-6:] == "weight" and name[-16:-7] != "LayerNorm" and (name[22:24] == str(i)+"." or name[22:25] == str(i)+"."):
            print(name)
            param_np = param.detach().numpy()
            c = []
            if param_np.ndim > 1:
                for v in param_np:
                    c_v = np.sum([1 for w in v if abs(w) > 0.1])
                    c.append(c_v)
            else:
                c_v = np.sum([1 for w in param_np if abs(w) > 0.1])
                c.append(c_v)
            wgt_dict_10M[f"layer_{i}"][name] = c

# pickle.dump(wgt_dict_10M, open("wgt_dict_10M.pkl", "wb"))
# wgt_dict_10M = pickle.load(open("wgt_dict_10M.pkl", "rb")))

wgt_dict_100M = dict()
for i in range(12):
    wgt_dict_100M[f"layer_{i}"] = dict()
    for name, param in m_100M.named_parameters():
        if name[8:15] == "encoder" and name[-6:] == "weight" and name[-16:-7] != "LayerNorm" and (name[22:24] == str(i)+"." or name[22:25] == str(i)+"."):
            print(name)
            param_np = param.detach().numpy()
            c = []
            if param_np.ndim > 1:
                for v in param_np:
                    c_v = np.sum([1 for w in v if abs(w) > 0.1])
                    c.append(c_v)
            else:
                c_v = np.sum([1 for w in param_np if abs(w) > 0.1])
                c.append(c_v)
            wgt_dict_100M[f"layer_{i}"][name] = c

# pickle.dump(wgt_dict_100M, open("wgt_dict_100M.pkl", "wb"))
# wgt_dict_100M = pickle.load(open("wgt_dict_100M.pkl", "rb")))
            
wgt_dict_base = dict()
for i in range(12):
    wgt_dict_base[f"layer_{i}"] = dict()
    for name, param in m_base.named_parameters():
        if name[8:15] == "encoder" and name[-6:] == "weight" and name[-16:-7] != "LayerNorm" and (name[22:24] == str(i)+"." or name[22:25] == str(i)+"."):
            print(name)
            param_np = param.detach().numpy()
            c = []
            if param_np.ndim > 1:
                for v in param_np:
                    c_v = np.sum([1 for w in v if abs(w) > 0.1])
                    c.append(c_v)
            else:
                c_v = np.sum([1 for w in param_np if abs(w) > 0.1])
                c.append(c_v)
            wgt_dict_base[f"layer_{i}"][name] = c

# pickle.dump(wgt_dict_base, open("wgt_dict_base.pkl", "wb"))
# wgt_dict_base = pickle.load(open("wgt_dict_base.pkl", "rb")))

#%%
# visualize weight counts
plot_lookup = [[0,0], [0,1], [0,2],
               [1,0], [1,1], [1,2],
               [2,0], [2,1], [2,2],
               [3,0], [3,1], [3,2]]

def wgt_dict_viz(wgt_dict, model_name, filename, log = False):
    f, axes = plt.subplots(4,3, figsize = (20,15))
    #f.tight_layout()
    plt.subplots_adjust(hspace = 0.2)
    plt.suptitle(model_name, fontsize = 30)
    for idx, layer in enumerate(wgt_dict):
        print(layer)
        row,col = plot_lookup[idx]
        l = layer.removeprefix("layer_")
        p = "roberta.encoder.layer."+l+"."
        for t in wgt_dict[layer]:
            t_name = t.removeprefix(p)[:-7]
            print(t_name)
            y = wgt_dict[layer][t]
            y.sort(reverse = True)
            if log == True:
                y = [np.log10(i+1) for i in y]
            x = np.linspace(0,100, len(y))
            axes[row,col].plot(x, y, label = t_name)
        axes[row,col].set_title(layer, fontsize = "xx-large")
        axes[row,col].set_xticks([])
        if idx == 10:
            axes[row,col].set_xlabel("Sorted Nodes", fontsize = 25)
        if idx == 6:
            if log == True:
                axes[row,col].set_ylabel('log10 # weights > |0.1| per node', fontsize = 25)
            else:
                axes[row,col].set_ylabel('# weights > |0.1| per node', fontsize = 25)
        if idx == 2:
            axes[row,col].legend(fontsize="x-large")
        # axes[row,col].legend()
        # axes[row,col].show(block = False)
    plt.savefig(filename, format = "pdf")

wgt_dict_viz(wgt_dict_10M, "Roberta_10M", "wgt_dist_01cutoff_10M.pdf")
wgt_dict_viz(wgt_dict_100M, "Roberta_100M", "wgt_dist_01cutoff_100M.pdf", log = True)
wgt_dict_viz(wgt_dict_base, "Roberta_base", "wgt_dist_01cutoff_base.pdf", log = True)      
    

#%%
m_init = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_init")

params_init = []
flat_init = []
for l in list(range(12)):
    layer = []    
    for name, param in m_init.named_parameters():
        if name[8:15] == "encoder" and name[-6:] == "weight" and name[-16:-7] != "LayerNorm"  and (name[22:24] == str(l)+"." or name[22:25] == str(l)+"."):
            print(name)
            # print(param)
            param_np = param.detach().numpy()
            if param_np.ndim > 1:
                layer.extend(param_np)
            else:
                layer.append(param_np)
    params_init.append(layer)
for layer in params_init:
    flattened = [i for p in layer for i in p]
    flat_init.append(flattened)
    
#just for fun
weight_hist(flat_init, "Roberta_untrained", log = True)

#%%
#raw value magnitude change in weights

def mag_change(init_weights, trained_weights):
    weight_change = []
    for i, layer in enumerate(init_weights):
        layer_change = []
        for j, param in enumerate(layer):
            param_change = param - trained_weights[i][j]
            layer_change.append(param_change)
        weight_change.append(layer_change)
    flat_change = []
    for layer in weight_change:
        flattened = [i for p in layer for i in p]
        flat_change.append(flattened)
    return weight_change, flat_change

change_10M, flat_ch_10M = mag_change(params_init, params_10M)
change_100M, flat_ch_100M = mag_change(params_init, params_100M)
change_base, flat_ch_base = mag_change(params_init, params_base)

weight_hist(flat_ch_10M, "Roberta_10M Magnitude Change", log = True)
weight_hist(flat_ch_100M, "Roberta_100M Magnitude Change", log = True)
weight_hist(flat_ch_base, "Roberta_base Magnitude Change", log = True)

#%%
#movement toward or away from 0
#TODO: fix to be more efficient--can I avoid looping through every single weight?

def dir_change(init_weights, trained_weights):
    weight_change = []
    for i, layer in enumerate(init_weights):
        layer_change = []
        print("layer", i)
        for j, param in enumerate(layer):
            #print(j)
            mult = param*trained_weights[i][j]
            sub = abs(param)-abs(trained_weights[i][j])
            
            mult_bin = [0 if w<0 else 1 for w in mult]
            sub_bin = [0 if w<0 else 1 for w in sub]
            
            param_dir = [a*b for a,b in zip(mult_bin, sub_bin)]
            layer_change.append(param_dir)
        weight_change.append(layer_change)
    dir_perc = []
    k = 0
    for layer in weight_change:
        print("calc_perc", k)
        p_perc = []
        for p in layer:
            l = sum([1 for w in p])
            s = sum([1-w for w in p])
            p_perc.append(s/l)
        dir_perc.append(p_perc)
        k+=1
    return weight_change, dir_perc

dir_10M, dir_p_10M = dir_change(params_init, params_10M)
weight_hist(dir_p_10M, "Roberta_10M Direction Change", log = True)

dir_100M, dir_p_100M = dir_change(params_init, params_100M)
weight_hist(dir_p_100M, "Roberta_100M Direction Change", log = True)

dir_base, dir_p_base = dir_change(params_init, params_base)
weight_hist(dir_p_base, "Roberta_base Direction Change", log = True)
#%%
#clean up functions to create weight dictionaries and visualize
#can use dicts to build out masks later?

def wgt_dict_build(model, mask_strategy, cutoff_perc, init_model = None):
    wgt_dict = dict()
    for i in range(12):
        wgt_dict[f"layer_{i}"] = dict()
        for name, param in model.named_parameters():
            if name[8:15] == "encoder" and name[-6:] == "weight" and name[-16:-7] != "LayerNorm" and (name[22:24] == str(i)+"." or name[22:25] == str(i)+"."):
                print(name)
                param_np = param.detach().numpy()
                c = []
                if mask_strategy == "raw_value":
                    abs_arr = np.abs(param_np)
                    cutoff = np.percentile(abs_arr, cutoff_perc)
                    for v in abs_arr:
                        c_v = np.sum([1 for w in v if w > cutoff])
                        c.append(c_v)
                elif mask_strategy == "movement":
                    init_param = init_model.state_dict()[name].detach().numpy()
                    param_change = np.abs(init_param-param_np)
                    cutoff = np.percentile(param_change, cutoff_perc)
                    for v in param_change:
                        c_v = np.sum([1 for w in v if w > cutoff])
                        c.append(c_v)
                elif mask_strategy == "direction":
                    init_param = init_model.state_dict()[name].detach().numpy()
                    
                    for idx, v in enumerate(param_np):
                        #if product is negative >> weight change >> not moving to zero
                        mult = init_param[idx]*v
                        #or if abs difference is negative >> weight growth
                        sub = np.abs(init_param[idx])-np.abs(v)
                        
                        mult_bin = [0 if w<0 else 1 for w in mult]
                        sub_bin = [0 if w<0 else 1 for w in sub]
                        # 0 = away from origin >> substract ones from total to get # zeros
                        c_v = len(v) - np.sum([a*b for a,b in zip(mult_bin, sub_bin)])
                        c.append(c_v)
                wgt_dict[f"layer_{i}"][name] = c
                
def wgt_dict_viz(wgt_dict, model_name, mask_strategy, cutoff_perc, filename, log = False):
    f, axes = plt.subplots(4,3, figsize = (20,15))
    #f.tight_layout()
    plt.subplots_adjust(hspace = 0.2)
    plt.suptitle(model_name, fontsize = 30)
    for idx, layer in enumerate(wgt_dict):
        print(layer)
        row,col = plot_lookup[idx]
        l = layer.removeprefix("layer_")
        p = "roberta.encoder.layer."+l+"."
        for t in wgt_dict[layer]:
            t_name = t.removeprefix(p)[:-7]
            print(t_name)
            y = wgt_dict[layer][t]
            y.sort(reverse = True)
            if log == True:
                y = [np.log10(i+1) for i in y]
            x = np.linspace(0,100, len(y))
            axes[row,col].plot(x, y, label = t_name)
        axes[row,col].set_title(layer, fontsize = "xx-large")
        axes[row,col].set_xticks([])
        if idx == 10:
            axes[row,col].set_xlabel("Sorted Nodes", fontsize = 25)
        if idx == 6:
            #TODO: make labels dynamic
            if mask_strategy == "raw_value":
                m = f"> ||"
            elif mask_strategy == "movement":
                m = ""
            if log == True:
                axes[row,col].set_ylabel('log10 # weights > |0.1| per node', fontsize = 25)
            else:
                axes[row,col].set_ylabel('# weights > |0.1| per node', fontsize = 25)
        if idx == 2:
            axes[row,col].legend(fontsize="x-large")
        # axes[row,col].legend()
        # axes[row,col].show(block = False)
    plt.savefig(filename, format = "pdf")