#!/bin/zsh
declare -a models=("bepler" "esm1b" "fasttext" "glove")
declare -a algorythms=("knn" "nb" "rf" "svm")
for model in ${models[@]};
do
  for algorythm in ${algorythms[@]};
  do
	  python generate_model.py ./kegg-data/ ${model} ${algorythm} ./kegg-data/
	  python test_model.py ./kegg-data/ ${model} ${algorythm} ./kegg-data/
	done
done