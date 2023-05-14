#!/bin/zsh
for i in {0..7}
do
	python random_forest.py data/Group_${i}_encoding_FFT.csv None data/results.csv response 30 1 2
done
