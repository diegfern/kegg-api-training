import pandas as pd
import numpy as np
import sys
import os

data_path = sys.argv[1]
base_name = sys.argv[2]
encoded_path = sys.argv[3]
encoding_model = sys.argv[4]
export_path = sys.argv[5]
batch_data = int(sys.argv[6])

print("Loading data")
dataframe_sequences = pd.read_csv(f"{data_path}{base_name}.csv")
if batch_data == 0:
    with np.load(f"{encoded_path}{encoding_model}_encoding.npz") as data:
        encoded_sequences = data['arr_0']
else:
    with np.load(f"{encoded_path}{encoding_model}_encoding{batch_data}.npz") as data:
        encoded_sequences = data['arr_0']

embedding_dimensions = {'bepler': 121,
                        'esm': 1280,
                        'esm1b': 1280,
                        'fasttext': 512,
                        'glove': 512,
                        'one_hot_encoding': 21,
                        'plus_rnn': 1024,
                        'prottrans_albert_bfd': 4096,
                        'prottrans_bert_bfd': 1024,
                        'prottrans_t5_xl_u50': 1024,
                        'prottrans_xlnet_uniref100': 1024,
                        'seqvec': 1024,
                        'word2vec': 512}
header = ['p_{}'.format(i) for i in range(embedding_dimensions[encoding_model])]

print("Processing data")
encoded_data = []
if batch_data == 0:
    for idx, row in dataframe_sequences.iterrows():
        encoded_data.append(encoded_sequences[row['index']])
else:
    i = batch_data
    for idx, row in dataframe_sequences.iterrows():
        encoded_data.append(encoded_sequences[row['index']])

dataframe_encoded = pd.DataFrame(encoded_data, columns=header)
dataframe_sequences = pd.concat([dataframe_sequences, dataframe_encoded], axis=1)

print("Saving data")
if not os.path.exists(export_path):
    os.makedirs(export_path)
dataframe_sequences.to_csv(f'{export_path}data_{encoding_model}.csv', index=False)