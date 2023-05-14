import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import os

data_path = sys.argv[1]
model_name = sys.argv[2]
response_name = sys.argv[3]
test_size = sys.argv[4]
export_path = sys.argv[5]

print("Loading Dataset")
df = pd.read_csv(f"{data_path}{model_name}.csv")

print("Dropping response column")
response = df[response_name]
df = df.drop(columns=['index', 'enzyme_code', 'organism', 'sequence_id', 'sequence_aa', 'response'], errors='ignore')

print("Splitting Train and Test")
X_train, X_test, y_train, y_test = train_test_split(df, response, test_size=int(test_size)/100, random_state=42)
#X_train = X_train.to_dataframe()
#X_test = X_test.to_dataframe()
#y_train = y_train.to_dataframe()
#y_test = y_test.to_dataframe()

print("Saving data")
if not os.path.exists(f"{export_path}/train"):
    os.makedirs(f"{export_path}/train")
if not os.path.exists(f"{export_path}/test"):
    os.makedirs(f"{export_path}/test")

X_train.to_csv(f'{export_path}/train/X_train_{model_name}.csv', index=False)
X_test.to_csv(f'{export_path}/test/X_test_{model_name}.csv', index=False)
y_train.to_csv(f'{export_path}/train/y_train_{model_name}.csv', index=False)
y_test.to_csv(f'{export_path}/test/y_test_{model_name}.csv', index=False)

# X_train <class 'pandas.core.frame.DataFrame'>
# X_test <class 'pandas.core.frame.DataFrame'>
# y_train <class 'pandas.core.series.Series'>
# y_test <class 'pandas.core.series.Series'>