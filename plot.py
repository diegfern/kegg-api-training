import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data_path = sys.argv[1]

results = pd.read_csv(f"{data_path}results.csv")
results = results.drop(columns=["min_ec","max_ec"])
results = results.sort_values(by=["test_size","data"])

normal = results.loc[~results['data'].str.contains("FFT")].reset_index()
fft = results.loc[results['data'].str.contains("FFT")].reset_index()

labels = results['test_size'].unique()

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()

for i in range(8):
    plt.bar(i - i/2,
                    normal,
                    normal.loc[normal['data'].str.contains(str(i))],
                    width
                    )

plt.show()