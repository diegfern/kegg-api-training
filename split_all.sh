#!/bin/zsh
declare -a models=("bepler" "esm1b" "fasttext" "glove" "plus_rnn" "prottrans_bert_bfd")
for model in ${models[@]}
	python split_data.py ./kegg-data/encoded/ ${model} response 30 ./kegg-data/
done
for i in {0..7}
	python split_data.py ./kegg-data/encoded/ Group_${i}_encoding response 30 ./kegg-data/
done