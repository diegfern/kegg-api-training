import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

data_path = sys.argv[1]
model_name = sys.argv[2]
algorythm = sys.argv[3]
export_path = sys.argv[4]

print("Loading Data")
X_train = pd.read_csv(f"{data_path}/train/X_train_{model_name}.csv")
X_test = pd.read_csv(f"{data_path}/test/X_test_{model_name}.csv")
y_train = pd.read_csv(f"{data_path}/train/y_train_{model_name}.csv")
y_test = pd.read_csv(f"{data_path}/test/y_test_{model_name}.csv")

y_train = y_train.squeeze()
y_test = y_test.squeeze()

train_size = len(y_train)/(len(y_test)+len(y_train))
test_size = len(y_test)/(len(y_test)+len(y_train))

print("Loading model")
model = joblib.load(f"{data_path}/models/{model_name}_{algorythm}.joblib")
p_from_model = model.predict(X_test)
print(p_from_model)
response_cm = confusion_matrix(y_test, p_from_model)

print("Cross validation")
scores = cross_validate(model, X_train, y_train, cv=10, scoring=['accuracy'])
print(scores)
print("Average: ", np.mean(scores['test_accuracy']))

print("Getting overfitting")
accuracy = accuracy_score(y_test, p_from_model)
overfitting_rate = (accuracy - np.mean(scores["test_accuracy"])/np.mean(scores['test_accuracy']))
print(overfitting_rate)

print("Saving to file")
#Save to JSON
'''
export_dict = {}
details_list = []
tmp_dict = {}
tmp_dict['encoding_model'] = model_name
tmp_dict['ml_algorythm'] = algorythm
tmp_dict['accuracy'] = accuracy
tmp_dict['train:test'] = f"{int(train_size * 100)}:{int(test_size * 100)}"
tmp_dict['avg'] = np.mean(scores['test_accuracy'])
tmp_dict['overfitting'] = overfitting_rate
tmp_dict['response_cm'] = response_cm.tolist()

details_list.append(tmp_dict)
export_dict['training_details'] = []
export_dict['testing_details'] = details_list

if not os.path.exists(f"{export_path}results.json"):
    with open(f"{export_path}results.json", "w") as outfile:
        json.dump(export_dict, outfile, indent=4)
else:
    with open(f"{export_path}results.json", "r+") as openfile:
        json_object = json.load(openfile)
        json_object['testing_details'].append(tmp_dict)
        openfile.seek(0)
        json.dump(json_object, openfile, indent=4)
'''

#Save to CSV
'''
try:
    df_results = pd.read_csv(f"{export_path}results.csv")
except:
    df_results = pd.DataFrame(,columns=['encoding','algorythm','test-train','avg','overfitting'])

df_results.loc[len(df_results.index)] = [model_name,algorythm,f"{train_size*100}:{test_size*100}",np.mean(scores['test_accuracy']), overfitting_rate]
df_results.to_csv(f"{export_path}results.csv",index=False)
'''