# argv 1 : dataset path
# argv 2 : response column name
# argv 3 : size (0-100)
import pandas as pd
import numpy as np
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate

# ./kegg-data/data_bepler.csv
data_path = sys.argv[1]
# ../kegg-api-data-tools/data-sampled/summary.csv
summary_path = sys.argv[2]
# ./kegg-data/
results_path = sys.argv[3]
# response
response_name = sys.argv[4]
# 30
test_size = sys.argv[5]
# 1
ec_start = sys.argv[6]
# 2
ec_end = sys.argv[7]

print("Loading Dataset")
df = pd.read_csv(data_path)

# This patch solution requires the summary data of the sample.
if summary_path != 'None':
    print("Truncating data")
    df_summary = pd.read_csv(summary_path)

    min_idx = -1
    max_idx = -1

    df_summary = df_summary[df_summary['terciary'].notna()]
    for idx, row in df_summary.iterrows():
        if min_idx == -1 and row['enzyme_code'][0] == ec_start:
            min_idx = idx
        if row['enzyme_code'][0] == ec_end:
            break
        max_idx = idx


    df = df.loc[ (df['response'] <= max_idx) & (df['response'] >= min_idx) ]

print("Dropping response column")
response = df[response_name]
df = df.drop(columns=['response','index', 'enzyme_code', 'organism', 'sequence_id', 'sequence_aa'])

X_train, X_test, y_train, y_test = train_test_split(df, response, test_size=int(test_size)/100, random_state=42)

print("Initializing Random Forest")
rf = RandomForestClassifier(n_jobs=-1, random_state=0)
rf.fit(X_train, y_train)

print("Making predictions")
p = rf.predict(X_test)

print("Getting metrics")
response_cm = confusion_matrix(y_test, p)
accuracy = accuracy_score(y_test, p)
print("Random Forest accuracy: ", accuracy)

print("Exporting model")
joblib.dump(rf, "model_random_forest.joblib")

print("Using model")
rf_load = joblib.load("model_random_forest.joblib")
p_from_model = rf_load.predict(X_test)
#print(p_from_model)

scores = cross_validate(rf, X_train, y_train, cv=5, scoring=['accuracy'])
#print(scores)
#print("Average: ", np.mean(scores['test_accuracy']))

print("Getting overfitting")
overfitting_rate = (accuracy - np.mean(scores["test_accuracy"])/np.mean(scores['test_accuracy']))
#print(overfitting_rate)

print("Saving to file")
try:
    df_results = pd.read_csv(results_path)
except:
    df_results = pd.DataFrame([],columns=['data','min_ec','max_ec','test_size','avg','overfitting'])

df_results.loc[len(df_results.index)] = [data_path,ec_start,ec_end,test_size,np.mean(scores['test_accuracy']), overfitting_rate]
df_results.to_csv(f"{results_path}results.csv",index=False)
