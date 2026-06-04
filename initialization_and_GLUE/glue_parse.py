#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:27:49 2024

@author: anna
"""

import os
import json
import numpy as np
import argparse

glue_dict = {'cola': "eval_mcc",
             'mnli': "eval_accuracy",
             'mnli-mm': "eval_accuracy",
             'mrpc': "eval_f1",
             'qnli': "eval_accuracy",
             'qqp': "eval_f1",
             'rte': "eval_accuracy",
             'sst2': "eval_accuracy",
             'boolq': "eval_accuracy",
             'multirc': "eval_accuracy",
             'wsc': "eval_accuracy"}

def glue_parse(model_folder):
    
    results = dict()
    for epoch in os.listdir(model_folder):
        folder_loc = model_folder+"/" + epoch
        results[epoch] = dict()
        for task in os.listdir(folder_loc):
            if task != "all_results.json":
                r_name = glue_dict[task]
                task_dir = folder_loc+"/"+task
                for filename in os.listdir(task_dir):
                    if filename == "all_results.json":
                        file_loc = task_dir+"/"+filename
                        with open(file_loc, "r") as f:
                            r = json.load(f)
                            results[epoch][task] = r[r_name]
        avg = np.mean(list(results[epoch].values()))
        results[epoch]['average'] = avg
    
    with open(model_folder+"/all_results.json", "w") as save_file:
        json.dump(results, save_file)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', help = 'Folder holding glue folders for all epochs', default = '', required = False)
    args = vars(parser.parse_args())
    
    glue_parse(**args)
    