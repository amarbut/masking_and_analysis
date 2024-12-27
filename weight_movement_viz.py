from transformers import RobertaForMaskedLM
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from copy import deepcopy

class ModelSeries(self, model_folder, init_model, layerwise = False):
    self.model_folder = model_folder
    self.init_model = init_model

    if layerwise == True:
        #TODO: is there a reason to compile and save layer_dicts, where all epochs for a single layer are stored together to iterate through? Yes, if we use the subplot_level="layer", would need to visualize all of the epochs for a single layer before moving on to the next layer.
        self.layer_dict_folder = f'{model_folder}/layer_dicts'
    else:
        self.layer_dict_folder = None

    def model_metric_calc(self, metric):
        '''
        compile model-level metrics for each epoch to be shown on a plot together with other models
        '''
        metric_dict = dict()
        for epoch in os.listdir(self.model_folder):
            if epoch == "epoch_1_0":
                m = RobertaForMaskedLM.from_pretrained(self.init_model)
                #TODO: METRIC STUFF FOR EPOCH 0
            m = RobertaForMaskedLM.from_pretrained(f'{self.model_folder}/{epoch}')
            #TODO: execute model_level metric calcs

        return metric_dict        

    def change_metric_calc(self, viz_level, agg_level, metric):
        '''
        agg_level: "weight", "head", "layer"; defines the level at which the metric is aggregated
        metric: defines the metric to be plotted
        '''

        #iterate through epoch models one at a time
        for epoch in os.listdir(self.model_folder):

            #model change metrics won't include an epoch_0 entry
            if epoch == "epoch_1_0":
                m_init = RobertaForMaskedLM.from_pretrained(self.init_model)
            else:
                #replace m_init with previous epoch's model
                m_init = deepcopy(m)
            m = RobertaForMaskedLM.from_pretrained(f'{self.model_folder}/{epoch}')
            
            metric_data = []
            
            if agg_level == 'weight':
                #WEIGHT LEVEL METRIC LIMITED TO ABSOLUTE CHANGE
                for name, param in m.named_parameters():
                    if ("encoder" in name 
                        and "weight" in name 
                        and "layer_norm" not in name):
                        p = param.data
                        p_init = m_init.state_dict()[name].data
                        
                        met = torch.abs(p.flatten()-p_init.flatten())
                        metric_data.extend(met)
                        
            elif agg_level == 'parameter':
                for name, param in m.named_parameters():
                    if ("encoder" in name 
                        and "weight" in name 
                        and "layer_norm" not in name):
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
                    metric_data.extend(met)
                    
            elif agg_level == 'layer':
                for i in range(12):
                    p = []
                    p_init = []
                    for name, param in m.named_parameters():
                        if ("encoder" in name 
                            and "weight" in name 
                            and "layer_norm" not in name 
                            and (name[22:24] == str(i)+"." or name[22:25] == str(i)+".")):
                            
                            p.append(param.flatten())
                            p_init.append(m_init.state_dict()[name].flatten())
                    p = torch.cat(p)
                    p_init = torch.cat(p_init)
                    
                    met = metric(p, p_init)
                    metric_data.extend(met)
                
                            
                    



    def viz_build(metric_data, viz_type):
        '''
        metric_dict: dictionary with metrics calculated at the specific agg level
        viz_type: "error" or "ridge"; defines whether to view metric as a line with error bars per epoch, or a ridgeline plot with one ridge per epoch 
        '''



    # MODEL CHANGE METRICS
    def weight_movement(self, p, p_init):
        #calculate L2 norm of difference between checkpoints at the relevant level
        



    def cosine_similarity(self, viz_level):
        #visualize the cosine similarity between checkpoints at the relevant level


    #MODEL LEVEL METRICS
    def glue_score(self, glue_folder):
        #visualize the model glue score over all checkpoints

    def avg_grad(self):
        #visualize the average gradient over all checkpoints

    def glue_time(self, glue_folder, glue_task):
        #visualize the training time for glue_task over all checkpoints


    #OTHER METRIC
    #TODO: does this need its own viz_build function?
    def connectivity(self, viz_level):
        #visualize the distribution of in/out nodes at the relevant level
        #is this a model-level statistic or a layer-level statistic? Or something else? Different than the model progress viz because not comparing between epochs
        #how to quantify whether the nodes follow a power law? apply mask?




                            
