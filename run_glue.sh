#!/bin/bash

for m in ascii rand roberta_s1 shuffle-corpus shuffle-sentence shuffle_index
do
	m_dir=${m}/hf_20
	nohup bash finetune_model.sh $m_dir

done