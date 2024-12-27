#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:20:29 2024

@author: anna
"""
import numpy as np
from transformers import RobertaForMaskedLM
import matplotlib.pyplot as plt
import seaborn as sns

w_mag = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/mask_10Mto100M_mag/checkpoint-186340")
g_mag = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/glue/mask_10Mto100M_mag/prune_70/seed_12/qqp")
w_rand = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/mask_100M_rand/checkpoint-186340")
g_rand = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/glue/mask_100M_rand/qqp")


def wgt_dict_build(model, glue_model, cutoff_perc):
    # wgt_dict = {"masked":dict(), "unmasked":dict()}
    glue_dict = {"masked":{"raw":dict(), "move":dict()}, "unmasked":{"raw":dict(), "move":dict()}}
    for i in range(12):
        # wgt_dict["masked"][f"layer_{i}"] = dict()
        glue_dict["masked"]["raw"][f"layer_{i}"] = dict()
        glue_dict["masked"]["move"][f"layer_{i}"] = dict()
        # wgt_dict["unmasked"][f"layer_{i}"] = dict()
        glue_dict["unmasked"]["raw"][f"layer_{i}"] = dict()
        glue_dict["unmasked"]["move"][f"layer_{i}"] = dict()
        for name, param in model.named_parameters():
            if name[8:15] == "encoder" and name[-6:] == "weight" and name[-16:-7] != "LayerNorm" and (name[22:24] == str(i)+"." or name[22:25] == str(i)+"."):
                print(name)
                param_np = param.detach().numpy()
                glue_param = glue_model.state_dict()[name].detach().numpy()
                #calculate movement of all weights after glue training
                param_change = np.abs(param_np-glue_param)
                cutoff = np.percentile(param_change, cutoff_perc)
                #make lists for masked/unmasked weight vectors
                w_m, g_m, c_m, w_u, g_u, c_u = [],[],[],[],[],[]
                for idx, v in enumerate(param_np):
                    #gather masked values and add to lists
                    # w_m_v = [w for w in v if w==0]
                    g_m_v = [g for g,w in zip(glue_param[idx],v) if w ==0 ]
                    c_m_v = [ch for ch,w in zip(param_change[idx],v) if w == 0 and ch >= cutoff]
                    
                    # w_m.append(w_m_v)
                    g_m.append(g_m_v)                    
                    c_m.append(c_m_v)
                    
                    #gather unmasked values and add to lists
                    # w_u_v = [w for w in v if w!=0]
                    g_u_v = [g for g,w in zip(glue_param[idx],v) if w !=0 ]
                    c_u_v = [ch for ch,w in zip(param_change[idx],v) if w != 0 and ch >= cutoff]
                    
                    # w_u.append(w_u_v)
                    g_u.append(g_u_v)                    
                    c_u.append(c_u_v)
                
                # wgt_dict["masked"][f"layer_{i}"][name] = w_m
                # wgt_dict["unmasked"][f"layer_{i}"][name] = w_u
                glue_dict["masked"]["raw"][f"layer_{i}"][name] = g_m
                glue_dict["unmasked"]["raw"][f"layer_{i}"][name] = g_u
                glue_dict["masked"]["move"][f"layer_{i}"][name] = c_m
                glue_dict["unmasked"]["move"][f"layer_{i}"][name] = c_u
    return glue_dict
#%%
#mag_glue = wgt_dict_build(w_mag, g_mag, 0)
rand_glue = wgt_dict_build(w_rand, g_rand, 0)

# import pickle
# pickle.dump(mag_wgts, open("magnitude_wgts_dict.pkl", "wb"))
# pickle.dump(mag_glue, open("magnitude_glue_wgt_dict.pkl", "wb"))
#%%
plot_lookup = [[0,0], [0,1], [0,2],
               [1,0], [1,1], [1,2],
               [2,0], [2,1], [2,2],
               [3,0], [3,1], [3,2]]
              
                
def wgt_dict_viz(glue_dict, model_name, filename, plot, log = False):
    x_move = max([max(i) for layer in glue_dict["unmasked"]["move"].values() 
                  for sl in layer.values() 
                  for i in sl ])
    x_raw_max = max([max(i) for layer in glue_dict["unmasked"]["raw"].values() 
                  for sl in layer.values() 
                  for i in sl ])
    x_raw_min = min([min(i) for layer in glue_dict["unmasked"]["raw"].values() 
                  for sl in layer.values() 
                  for i in sl ])
    for m in ["masked", "unmasked"]:
        for viz in ["raw", "move"]:
            f, axes = plt.subplots(4,3, figsize = (20,15))
            #f.tight_layout()
            plt.subplots_adjust(hspace = 0.2)
            plt.suptitle(model_name+" "+m+" "+viz, fontsize = 30)
            for idx, layer in enumerate(glue_dict[m][viz]):
                print(m, viz, layer)
                row,col = plot_lookup[idx]
                l = layer.removeprefix("layer_")
                p = "roberta.encoder.layer."+l+"."
                for t in glue_dict[m][viz][layer]:
                    t_name = t.removeprefix(p)[:-7]
                    print(t_name)
                    if plot == "density":
                        data = [i for v in glue_dict[m][viz][layer][t] for i in v]
                        sns.kdeplot(data, ax=axes[row,col],label=t_name)
                            
                    if plot == "hist":
                        data = [i for v in glue_dict[m][viz][layer][t] for i in v]
                        y, x = np.histogram(data, bins = 30)
                        x = (x[:-1]+x[1:])/2
                        if log == True:
                            y = [np.log10(i+1) for i in y]
                        axes[row,col].plot(x, y, label = t_name)
                if viz == "raw":
                    axes[row,col].set_xlim(left = x_raw_min, right = x_raw_max)
                elif viz == "move":
                    axes[row,col].set_xlim(left = 0, right = x_move)

                axes[row,col].set_title(layer, fontsize = "xx-large")
                if idx == 10:
                    if viz == "raw":
                        axes[row,col].set_xlabel("Weight Value", fontsize = 25)
                    elif viz == "move":
                        axes[row,col].set_xlabel("Absolute Weight Change", fontsize = 25)
                if idx == 6:
                    if plot == "density":
                        axes[row,col].set_ylabel("Density", fontsize = 25)
                    elif plot == "hist":
                        if log == True:
                            axes[row,col].set_ylabel("Log10 Counts", fontsize = 25)
                        else:
                            axes[row,col].set_ylabel("Raw Counts", fontsize = 25)
                if idx == 2:
                    axes[row,col].legend(fontsize="x-large")
                # axes[row,col].legend()
                # axes[row,col].show(block = False)
            fname = filename+"_"+m+"_"+viz+".pdf"
            plt.savefig(fname, format = "pdf")
#%%
 
#wgt_dict_viz(mag_glue, "10Mto100M Magnitude","10Mto100M_mag", plot = "hist", log = True)
wgt_dict_viz(rand_glue, "100M Random","100M_rand", plot = "hist", log = True)

#%%

move_um = sum([len(i) for layer in mag_glue["unmasked"]["move"].values() 
              for sl in layer.values() 
              for i in sl ])

move_m = sum([len(i) for layer in mag_glue["masked"]["move"].values() 
              for sl in layer.values() 
              for i in sl ])

move_um_r = sum([len(i) for layer in rand_glue["unmasked"]["move"].values() 
              for sl in layer.values() 
              for i in sl ])

move_m_r = sum([len(i) for layer in rand_glue["masked"]["move"].values() 
              for sl in layer.values() 
              for i in sl ])

raw_um = sum([len(i) for layer in mag_glue["unmasked"]["raw"].values() 
              for sl in layer.values() 
              for i in sl ])

raw_m = sum([len(i) for layer in mag_glue["masked"]["raw"].values() 
              for sl in layer.values() 
              for i in sl ])

raw_um_r = sum([len(i) for layer in rand_glue["unmasked"]["raw"].values() 
              for sl in layer.values() 
              for i in sl ])

raw_m_r = sum([len(i) for layer in rand_glue["masked"]["raw"].values() 
              for sl in layer.values() 
              for i in sl ])

#%%

w_mag_masked = 0
w_mag_unmasked = 0
w_mag_total = 0

for name, param in w_mag.named_parameters():
    if name[8:15] == "encoder" and name[-6:] == "weight" and name[-16:-7] != "LayerNorm":
        
        print(name)
        param_np = param.detach().numpy()
        for v in param_np:
            mask = sum([1 for w in v if w == 0])
            unmask = sum([1 for w in v if w != 0])
            total = sum([1 for w in v])
            
            w_mag_masked += mask
            w_mag_unmasked += unmask
            w_mag_total += total
                    
