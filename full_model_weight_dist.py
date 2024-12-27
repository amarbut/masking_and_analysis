from transformers import RobertaForMaskedLM, BertForMaskedLM, GPT2Model, T5Model

roberta = RobertaForMaskedLM.from_pretrained("roberta-base")
bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
gpt = GPT2Model.from_pretrained("gpt2")
t5 = T5Model.from_pretrained("t5-base")
#%%

total_sum = 0.0
total_squared_sum = 0.0
total_elements = 0

for name, param in roberta.named_parameters():
    if "encoder" in name and "weight" in name and "LayerNorm" not in name:
        print(name)
        weights = param.data
        total_sum += weights.sum().item()  # Accumulate the sum of weights
        total_squared_sum += (weights ** 2).sum().item()  # Accumulate the sum of squared weights
        total_elements += weights.numel()  # Count the total number of elements

rob_mean = total_sum / total_elements
rob_stddev = ((total_squared_sum / total_elements) - (rob_mean ** 2)) ** 0.5
# %%

total_sum = 0.0
total_squared_sum = 0.0
total_elements = 0

for name, param in bert.named_parameters():
    if "encoder" in name and "weight" in name and "LayerNorm" not in name:
        print(name)
        weights = param.data
        total_sum += weights.sum().item()  # Accumulate the sum of weights
        total_squared_sum += (weights ** 2).sum().item()  # Accumulate the sum of squared weights
        total_elements += weights.numel()  # Count the total number of elements

bert_mean = total_sum / total_elements
bert_stddev = ((total_squared_sum / total_elements) - (rob_mean ** 2)) ** 0.5

#%%
total_sum = 0.0
total_squared_sum = 0.0
total_elements = 0

for name, param in gpt.named_parameters():
    if "weight" in name and "ln" not in name:
        print(name)
        weights = param.data
        total_sum += weights.sum().item()  # Accumulate the sum of weights
        total_squared_sum += (weights ** 2).sum().item()  # Accumulate the sum of squared weights
        total_elements += weights.numel()  # Count the total number of elements

gpt_mean = total_sum / total_elements
gpt_stddev = ((total_squared_sum / total_elements) - (rob_mean ** 2)) ** 0.5
#%%
total_sum = 0.0
total_squared_sum = 0.0
total_elements = 0

for name, param in t5.named_parameters():
    if "encoder" in name and "weight" in name and "layer_norm" not in name:
        print(name)
        weights = param.data
        total_sum += weights.sum().item()  # Accumulate the sum of weights
        total_squared_sum += (weights ** 2).sum().item()  # Accumulate the sum of squared weights
        total_elements += weights.numel()  # Count the total number of elements

t5_mean = total_sum / total_elements
t5_stddev = ((total_squared_sum / total_elements) - (rob_mean ** 2)) ** 0.5

#%%
import torch

rob_init = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/HPC_store/models/roberta_init")
rob_2B = RobertaForMaskedLM.from_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/roberta_1B/checkpoint-2870140")

wgt_change = []
chunk_size = 100000

for name, param in rob_init.named_parameters():
    if "encoder" in name and "weight" in name and "layer_norm" not in name:
        print(name)
        i_weights = param.data
        t_weights = rob_2B.state_dict()[name].data
        param_change = torch.abs(t_weights-i_weights)

wgt_change = torch.cat(wgt_change)

# percentile_90 = torch.quantile(wgt_change, 0.90).item()
# percentile_95 = torch.quantile(wgt_change, 0.95).item()
# percentile_99 = torch.quantile(wgt_change, 0.99).item()

k90 = int(0.90 * len(wgt_change))  # Calculate the index for the 90th percentile
k95 = int(0.95 * len(wgt_change))
k99 = int(0.99 * len(wgt_change))

# Use topk to find the k-th largest value, which approximates the 90th percentile
top_90_values = torch.topk(-wgt_change, k90)
top_95_values = torch.topk(-wgt_change, k95)
top_99_values = torch.topk(-wgt_change, k99)

percentile_90_approx = top_90_values.values[-1].item() 
percentile_95_approx = top_95_values.values[-1].item() 
percentile_99_approx = top_99_values.values[-1].item() 
