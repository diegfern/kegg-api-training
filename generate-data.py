import pandas as pd
import numpy as np
import sys
import os
# Existen 2 opciones:
# 1. La data esta en 1 npz. Ejecutar:
#   ../kegg-api-data-tools/data-sampled/ sequences ../bio-embeddings/kegg-data-encoded/ fasttext ./kegg-data/encoded/
# 2. La data esta particionada en n npz. Ejecutar:
#   ../kegg-api-data-tools/data-sampled/ sequences ../bio-embeddings/kegg-data-encoded/ esm1b ./kegg-data/encoded/ 3866683 1 4
#   ../kegg-api-data-tools/data-sampled/ sequences ../bio-embeddings/kegg-data-encoded/ esm1b ./kegg-data/encoded/ 3866683 2 4
#   ../kegg-api-data-tools/data-sampled/ sequences ../bio-embeddings/kegg-data-encoded/ esm1b ./kegg-data/encoded/ 3866683 3 4
#   ../kegg-api-data-tools/data-sampled/ sequences ../bio-embeddings/kegg-data-encoded/ esm1b ./kegg-data/encoded/ 3866683 4 4

data_path = sys.argv[1]
base_name = sys.argv[2]
encoded_path = sys.argv[3]
encoding_model = sys.argv[4]
export_path = sys.argv[5]
total_sequences = 0
n_file = 0
total_files = 0


print("Loading data")
dataframe_sequences = pd.read_csv(f"{data_path}{base_name}.csv")
if len(sys.argv) <= 6:
    with np.load(f"{encoded_path}{encoding_model}_encoding.npz") as data:
        encoded_sequences = data['arr_0']
else:
    n_file = int(sys.argv[7])
    with np.load(f"{encoded_path}{encoding_model}_encoding{n_file}.npz") as data:
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
idx_diff = 0
idx_split = 0
encoded_data = []
if len(sys.argv) <= 6:
    for idx, row in dataframe_sequences.iterrows():
        encoded_data.append(encoded_sequences[row['index']])
        idx_split = row['index']
else:
    total_sequences = int(sys.argv[6])
    total_files = int(sys.argv[8])
    idx_diff = int(total_sequences*(n_file-1)/total_files)
    for idx, row in dataframe_sequences.iterrows():
        if row['index'] < idx_diff:
            continue
        if (row['index'] - idx_diff) < len(encoded_sequences):
            encoded_data.append(encoded_sequences[row['index'] - idx_diff])
            idx_split = row['index']
        else:
            break


if not os.path.exists(f"{export_path}{encoding_model}.csv"):
    dataframe_sequences = dataframe_sequences.loc[dataframe_sequences['index'] <= idx_split]
    dataframe_encoded = pd.DataFrame(encoded_data, columns=header)
    dataframe_sequences = pd.concat([dataframe_sequences, dataframe_encoded], axis=1)

    print("Saving data")
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    dataframe_sequences.to_csv(f'{export_path}{encoding_model}.csv', index=False)
else:
    print("Copying into existing data")
    dataframe_sequences = dataframe_sequences.loc[
         (dataframe_sequences['index'] >= idx_diff) & (dataframe_sequences['index'] <= idx_split)]

    dataframe_already_encoded = pd.read_csv(f"{export_path}{encoding_model}.csv")

    dataframe_encoded = pd.DataFrame(encoded_data, columns=header)
    dataframe_sequences.reset_index(drop=True, inplace=True)
    dataframe_encoded.reset_index(drop=True, inplace=True)
    dataframe_sequences = pd.concat([dataframe_sequences, dataframe_encoded], axis=1)

    print("Saving data")
    export_dataframe = dataframe_already_encoded.append(dataframe_sequences, ignore_index=True)
    export_dataframe.to_csv(f'{export_path}{encoding_model}.csv', index=False)


