from transformers import RobertaForMaskedLM
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os
from copy import deepcopy
import re
import json

class ModelSeries:
    def __init__(self, model_folder, init_model, layerwise = False):
        self.model_folder = model_folder
        self.model_name = model_folder.split("/")[-1]
        self.init_model = init_model
        self.num_chk = len(os.listdir(model_folder))
        
        #dictate order to navigate epochs based on number of checkpoints
        epoch_map = {20: [f"epoch_{i}_0" for i in range(1,21)] +["final"],
                     41: [f"epoch_{i//2}_{int(round((i/2)-int(i/2),1)*10)}" for i in range (1,41)]+["final"],
                     61: [f"epoch_{i//3}_{int(round((i/3)-int(i/3),1)*10)}" for i in range (1,61)]+["final"],
                     }
        self.epoch_list = epoch_map[self.num_chk]
        self.min_chk = self.epoch_list[0]
        
        #load training state for final saved model
        if self.model_name ==  "roberta_baby":
            self.ts_dict = None
        else:
            if os.path.exists(self.model_folder+"/final"):
                    fname = self.model_folder+"/final/trainer_state.json"
            else:
                fname = self.model_folder+"/epoch_20_0/trainer_state.json"
            
            with open(fname, "r") as f:
                self.ts_dict = json.load(f)
        
    
        # if layerwise == True:
            
        #     self.layer_dict_folder = f'{model_folder}/layer_dicts'
        #     #check if the layerwise dictionaries already exist
        #     if os.path.exists(self.layer_dict_folder) and len(os.listdir(self.layer_dict_folder)==12):
        #         continue
        #     else:
        #         #TODO build out layerwise model dictionaries
        #         os.mkdir(self.layer_dict_folder)
        #         self.layer_dict_build()
        # else:
        #     self.layer_dict_folder = None
    
        # def model_metric_calc(self, metric):
        #     '''
        #     compile model-level metrics for each epoch to be shown on a plot together with other models
        #     '''
        #     metric_dict = dict()
        #     for epoch in os.listdir(self.model_folder):
        #         if epoch == self.min_chk:
        #             m = RobertaForMaskedLM.from_pretrained(self.init_model)
        #             #TODO: METRIC STUFF FOR EPOCH 0
        #         m = RobertaForMaskedLM.from_pretrained(f'{self.model_folder}/{epoch}')
        #         #TODO: execute model_level metric calcs
    
        #     return metric_dict        

    def change_metric_calc(self, m, m_init, agg_level, metric):
        '''
        agg_level: "weight", "head", "layer"; defines the level at which the metric is aggregated
        metric: defines the metric to be calculated
        '''
        
        metric_data = []
        
        if agg_level == 'weight':
            #WEIGHT LEVEL METRIC LIMITED TO ABSOLUTE CHANGE
            for name, param in m.named_parameters():
                if ("encoder" in name 
                    and "weight" in name 
                    and "LayerNorm" not in name):
                    print(name)
                    p = param.data
                    p_init = m_init.state_dict()[name].data
                    # met = torch.abs(p.flatten() - p_init.flatten()) / torch.norm(p_init.flatten()) #represent as proportion of previous weight value
                    met = torch.abs(p.flatten()-p_init.flatten())
                    
                    metric_data.extend(met.tolist())
                    
        elif agg_level == 'parameter':
            for name, param in m.named_parameters():
                if ("encoder" in name 
                    and "weight" in name 
                    and "LayerNorm" not in name):
                    print(name)
                    p = param.data
                    p_init = m_init.state_dict()[name].data
                    
                    met = metric(p, p_init)
                    metric_data.extend(met)
                                        
                        
        elif agg_level == 'head':
            for i in range(12):
                hidden_size = m.config.hidden_size
                num_heads = m.config.num_attention_heads
                head_dim = hidden_size // num_heads
                
                p = []
                p_init = []
                for name, param in m.named_parameters():
                    if ("encoder" in name 
                        and "weight" in name 
                        and ("query" in name or "value" in name or "key" in name) 
                        and (name[22:24] == str(i)+"." or name[22:25] == str(i)+".")):
                        print(name)
                        
                        p.append(param.data.view(num_heads, head_dim, -1))
                        p_init.append(m_init.state_dict()[name].data.view(num_heads, head_dim, -1))
                p = torch.cat(p, dim=-1)
                p_init = torch.cat(p_init, dim=-1)
                
                met = metric(p, p_init)
                metric_data.extend(met.tolist())
                
        elif agg_level == 'layer':
            for i in range(12):
                p = []
                p_init = []
                for name, param in m.named_parameters():
                    if ("encoder" in name 
                        and "weight" in name 
                        and "LayerNorm" not in name 
                        and (name[22:24] == str(i)+"." or name[22:25] == str(i)+".")):
                        print(name)
                        
                        p.append(param.flatten())
                        p_init.append(m_init.state_dict()[name].flatten())
                p = torch.cat(p)
                p_init = torch.cat(p_init)
                
                met = metric(p, p_init)
                metric_data.append(met.tolist())
        
        return metric_data
                


    def change_viz_build(self, metric_name, agg_level, viz_type, filename, plot_title, lr_overlay = False, loss_overlay = False):
        '''
        agg_level: "weight", "head", "layer"; defines the level at which the metric is aggregated
        metric: defines the metric to be plotted
        viz_type: "error" or "ridge"; defines whether to view metric as a line with error bars per epoch, or a ridgeline plot with one ridge per epoch 
        '''
        if lr_overlay and loss_overlay:
            print("Can only add one model metric overlay to plot")
            return
        if viz_type == "ridge" and (lr_overlay or loss_overlay):
            print("Model metric overlay only available for error bar plot")
            return
        if self.ts_dict == None and (lr_overlay or loss_overlay):
            print("Model metrics not available")
            return
        
        
        metric_dict = {"weight_movement":self.weight_movement,
                       "cosine_similarity": self.cosine_similarity}
        
        metric = metric_dict[metric_name]
        m = None
        
        if viz_type == "error":
            
            #store tuple of distribution data for each checkpoint's metric dict
            met_dist = []
        
            for epoch in self.epoch_list:
                if epoch in os.listdir(self.model_folder):
                    print(epoch)
                    
                    if epoch != "final":
                        epoch_num = float(re.search(r"epoch_(.*)", epoch).group(1).replace("_", "."))
                    else:
                        epoch_num = 20.1
                    
                    if epoch == self.min_chk:
                        m_init = RobertaForMaskedLM.from_pretrained(self.init_model) #use untrained model for first datapoint
                    else:
                        m_init = deepcopy(m) #use previous checkpoint's model for all following datapoints
                    print("loading model")
                    m = RobertaForMaskedLM.from_pretrained(f'{self.model_folder}/{epoch}')
                    
                    print("calculating metric")
                    metric_data = self.change_metric_calc(m, m_init, agg_level, metric)
                    
                    met_med = np.median(metric_data)
                    met_low = np.quantile(metric_data, 0.25)
                    met_high = np.quantile(metric_data, 0.75)
                    
                    met_dist.append((epoch_num, met_med, met_low, met_high))
                
            print("visualizing data")
            #split up checkpoint data for visualization
            met_dist = sorted(met_dist, key = lambda t: t[0])
            
            epoch_nums = [c[0] for c in met_dist]
            meds = [c[1] for c in met_dist]
            low = [c[1]-c[2] for c in met_dist]
            high = [c[3]-c[1] for c in met_dist]
            
            fig, ax1 = plt.subplots(constrained_layout = True)
            
            ax1.errorbar(epoch_nums, meds, yerr=[low,high], capsize=5, label = metric_name, color = "blue")
            ax1.set_ylabel(metric_name, color = "blue", fontsize = "large")
            ax1.set_xlabel("Epoch", fontsize = "large")
            ax1.tick_params(axis="y", colors = "blue")
            ax1.grid(False)
            
            if lr_overlay:
                
                lr_epochs = [h["epoch"] for h in self.ts_dict["log_history"]]
                lr = [h["learning_rate"] for h in self.ts_dict["log_history"]]
                
                ax2 = ax1.twinx()
                ax2.plot(lr_epochs, lr, label = "learning rate", color = "red")
                ax2.set_ylabel("learning_rate", color = "red", fontsize = "large")
                ax2.tick_params(axis = "y", colors = "red")
                ax2.grid(False)
                
            if loss_overlay:
                
                loss_epochs = [h["epoch"] for h in self.ts_dict["log_history"]]
                loss = [h["loss"] for h in self.ts_dict["log_history"]]
                
                ax2 = ax1.twinx()
                ax2.plot(loss_epochs, loss, label = "loss", color = "red")
                ax2.set_ylabel("loss", color = "red", fontsize = "large")
                ax2.tick_params(axis = "y", colors = "red")
                ax2.grid(False)
                
            
            #plt.suptitle(plot_title, )
            plt.title(plot_title, fontsize = "xx-large")
            plt.savefig(filename, bbox_inches = "tight", pad_inches = 0.02)
            plt.show()
            
            
                
        elif viz_type == "ridge":
            
            #visualize distribution of metric data over each checkpoint as a ridgline
            fig, axes = plt.subplots(self.num_chk, 1, figsize = (6,20), sharex=True, constrained_layout = True)
            sns.set(style="whitegrid")
            
            for idx,epoch in enumerate(self.epoch_list):
                if epoch in os.listdir(self.model_folder):
                    print(epoch)
                    if epoch != "final":
                        epoch_num = float(re.search(r"epoch_(.*)", epoch).group(1).replace("_", "."))
                    else:
                        epoch_num = 20.1
                    
                    if epoch == self.min_chk:
                        m_init = RobertaForMaskedLM.from_pretrained(self.init_model) #use untrained model for first datapoint
                    else:
                        m_init = deepcopy(m) #use previous checkpoint's model for all following datapoints
                    print("loading model")
                    m = RobertaForMaskedLM.from_pretrained(f'{self.model_folder}/{epoch}')
                    
                    print("calculating metric")
                    metric_data = self.change_metric_calc(m, m_init, agg_level, metric)
                    
                    print("visualizing data")
                    sns.kdeplot(metric_data, ax = axes[idx], fill = True, color="skyblue", linewidth=1.5)
                    axes[idx].set_title(epoch)
            plt.suptitle(self.model_name+": "+agg_level+" level")
            plt.xlabel(metric_name)
            # plt.tight_layout()
            plt.savefig(filename, bbox_inches = "tight", pad_inches=0.02)
            plt.show()      
                


    # MODEL CHANGE METRICS
    def weight_movement(self, p, p_init):
        #calculate L2 norm of difference between checkpoints
        
        dims = tuple(range(1, p.dim())) #calculate over all dims other than first
        met = torch.norm(p-p_init, dim=dims)
        return met

    def cosine_similarity(self, p, p_init):
        #calculate the cosine similarity between checkpoints
        
        reshaped_p = p.view(p.shape[0], -1) #flatten structure after first dimension
        reshaped_p_init = p_init.view(p_init.shape[0], -1)
        met =F.cosine_similarity(reshaped_p, reshaped_p_init, dim=1)
        return met


    # #MODEL LEVEL METRICS
    # def glue_score(self, glue_folder):
    #     #visualize the model glue score over all checkpoints          
            

    # def glue_time(self, glue_folder, glue_task):
    #     #visualize the training time for glue_task over all checkpoints


    # #OTHER METRIC
    # #TODO: does this need its own viz_build function?
    # def connectivity(self, viz_level):
    #     #visualize the distribution of in/out nodes at the relevant level
    #     #is this a model-level statistic or a layer-level statistic? Or something else? Different than the model progress viz because not comparing between epochs
    #     #how to quantify whether the nodes follow a power law? apply mask?
        
    # def layer_dict_build(self):
    #     #BUILD OUT LAYER WISE DICTS FOR ALL EPOCHS

#%%

raw = ModelSeries("/media/easystore/initialization_all_epochs/roberta_10M", "/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_init")

raw.change_viz_build("weight_movement", "weight", "error")
raw.change_viz_build("weight_movement", "head", "error")
raw.change_viz_build("weight_movement", "layer", "error", loss_overlay = True)

raw.change_viz_build("cosine_similarity", "head", "error")
                            

raw.change_viz_build("weight_movement", "weight", "ridge")
raw.change_viz_build("weight_movement", "layer", "ridge")

#%%

baby = ModelSeries("/media/easystore/initialization_all_epochs/roberta_baby", "/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_init")

baby.change_viz_build("cosine_similarity", "head", "error")

#%%

clr = ModelSeries("/media/easystore/initialization_all_epochs/roberta_10M_clr", "/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_init")

clr.change_viz_build("cosine_similarity", "head", "error", lr_overlay = True)

#%%
clr = ModelSeries("/media/anna/easystore/initialization_all_epochs/roberta_100M_clr", "/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_init")

clr.change_viz_build("cosine_similarity", "head", "error",filename = "hdcos_roberta_100M.pdf", lr_overlay = False, plot_title = "HF Training 100M")
#%%
clr = ModelSeries("/media/anna/easystore/initialization_all_epochs/roberta_baby", "/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_init")

clr.change_viz_build("cosine_similarity", "head", "error",filename = "hdcos_babylm_10M.pdf", lr_overlay = False, plot_title = "BabyLM 10M")
#%%
clr = ModelSeries("/media/anna/easystore/initialization_all_epochs/roberta_100M_clr", "/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_init")

clr.change_viz_build("weight_movement", "weight", "error",filename = "wgtmv_roberta_100M.pdf", lr_overlay = False, plot_title = "HF Training 100M")
#%%
clr = ModelSeries("/media/anna/easystore/initialization_all_epochs/roberta_baby", "/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_init")

clr.change_viz_build("weight_movement", "weight", "error",filename = "wgtmv_babylm_10M.pdf", lr_overlay = False, plot_title = "BabyLM 10M")
#%%
clr = ModelSeries("/media/anna/easystore/initialization_all_epochs/roberta_10M_clr", "/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_init")

clr.change_viz_build("cosine_similarity", "head", "error",filename = "hdcos_roberta_10M_LR.pdf", lr_overlay = True, plot_title = "HF Training 10M")
#%%
clr = ModelSeries("/media/anna/easystore/initialization_all_epochs/roberta_10M_clr", "/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_init")

clr.change_viz_build("weight_movement", "weight", "error",filename = "wgtmv_roberta_10M.pdf", lr_overlay = False, plot_title = "HF Training 10M")

#%%
clr = ModelSeries("/media/anna/easystore/initialization_all_epochs/roberta_10M_clr", "/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_init")

clr.change_viz_build("weight_movement", "weight", "error",filename = "wgtmv_roberta_10M_loss.pdf", loss_overlay = True, plot_title = "HF Training 10M")
#%%
fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)                                                                                     
                                                                                                                                                              
for folder, label, color in [                                                                                                                               
    ("/media/anna/easystore/initialization_all_epochs/roberta_100M_clr", "Constant LR", "#4c8be0"),                                                         
    ("/media/anna/easystore/initialization_all_epochs/roberta_100M",     "Linear LR",   "#e07b39"),                                                         
]:                                                                                                                                                          
    with open(folder + "/final/trainer_state.json") as f:                                                                                                   
        log = [h for h in json.load(f)["log_history"] if "loss" in h and "learning_rate" in h]                                                              
    epochs = [h["epoch"] for h in log]                                                                                                                      
    ax_loss.plot(epochs, [h["loss"] for h in log], color=color, label=label)
    ax_lr.plot(epochs,   [h["learning_rate"] for h in log], color=color, label=label)                                                                       
                
ax_loss.set_ylabel("Loss")                                                                                                                                  
ax_lr.set_ylabel("Learning rate")
ax_lr.set_xlabel("Epoch")                                                                                                                                   
ax_loss.legend()
fig.suptitle("100M training: loss and LR over epochs")
fig.tight_layout()                                                                                                                                          
plt.show()
#%%
fig, ax_loss = plt.subplots(figsize=(8, 4))                                                                                                                 
ax_lr = ax_loss.twinx()                                                                                                                                     
                                                                                                                                                              
models = [      
    ("/media/anna/easystore/initialization_all_epochs/roberta_100M_clr", "Constant LR", "#4c8be0"),
    ("/media/anna/easystore/initialization_all_epochs/roberta_100M",     "Linear LR",   "#e07b39"),                                                         
]
                                                                                                                                                            
for folder, label, color in models:                                                                                                                         
    with open(folder + "/final/trainer_state.json") as f:
        log = [h for h in json.load(f)["log_history"] if "loss" in h and "learning_rate" in h]                                                              
    epochs = [h["epoch"] for h in log]
    ax_loss.plot(epochs, [h["loss"] for h in log], color=color, linestyle="-",  label=f"{label} — loss")                                                    
    ax_lr.plot(  epochs, [h["learning_rate"] for h in log], color=color, linestyle="--", label=f"{label} — LR", alpha=0.6)                                  
                                                                                                                                                            
ax_loss.set_yscale("log")                                                                                                                                   
ax_loss.set_ylabel("Loss (log scale)", fontsize = "large")                                                                                                                      
ax_loss.set_xlabel("Epoch", fontsize = "large")                                                                                                                                 
ax_lr.set_ylabel("Learning rate", fontsize = "large")
ax_lr.grid(False)                                                                                                                                           
                
lines1, labels1 = ax_loss.get_legend_handles_labels()                                                                                                       
lines2, labels2 = ax_lr.get_legend_handles_labels()
ax_loss.legend(lines1 + lines2, labels1 + labels2, fontsize = "large", framealpha=0.9)                                                                              
                                                                                                                                                            
fig.suptitle("100M training: loss and learning rate", fontsize = "xx-large")
fig.tight_layout()                                                                                                                                          
plt.savefig("100M_loss_lr_comparison.pdf", format="pdf")
plt.show()      
#%%
def load_log(folder):
    for subfolder in ["final", "epoch_20_0"]:
        path = os.path.join(folder, subfolder, "trainer_state.json")                                                                                        
        if os.path.exists(path):
            with open(path) as f:                                                                                                                           
                return [h for h in json.load(f)["log_history"] if "loss" in h and "learning_rate" in h]                                                     
    raise FileNotFoundError(f"No trainer_state.json found in {folder}")
                                                                                                                                                            
fig, ax_loss = plt.subplots(figsize=(8, 4))
ax_lr = ax_loss.twinx()                                                                                                                                     
                
models = [
    ("/media/anna/easystore/initialization_all_epochs/roberta_10M_clr", "Constant LR", "#4c8be0"),
    ("/media/anna/easystore/initialization_all_epochs/roberta_10M",     "Linear LR",   "#e07b39"),                                                          
]
                                                                                                                                                            
for folder, label, color in models:                                                                                                                         
    log = load_log(folder)
    epochs = [h["epoch"] for h in log]                                                                                                                      
    ax_loss.plot(epochs, [h["loss"] for h in log], color=color, linestyle="-",  label=f"{label} — loss")
    ax_lr.plot(  epochs, [h["learning_rate"] for h in log], color=color, linestyle="--", label=f"{label} — LR", alpha=0.6)                                  
 
ax_loss.set_yscale("log")                                                                                                                                   
ax_loss.set_ylabel("Loss (log scale)", fontsize = "large")
ax_loss.set_xlabel("Epoch", fontsize = "large")                                                                                                                                 
ax_lr.set_ylabel("Learning rate", fontsize = "large")
ax_lr.grid(False)

lines1, labels1 = ax_loss.get_legend_handles_labels()                                                                                                       
lines2, labels2 = ax_lr.get_legend_handles_labels()
ax_loss.legend(lines1 + lines2, labels1 + labels2,  fontsize = "large", framealpha=0.9)                                                                              
                
fig.suptitle("10M training: loss and learning rate", fontsize = "xx-large")                                                                                                        
fig.tight_layout()
plt.savefig("10M_loss_lr_comparison.pdf", format="pdf")                                                                                                     
plt.show()      
                                 