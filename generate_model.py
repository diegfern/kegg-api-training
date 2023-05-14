import sys
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
import json

data_path = sys.argv[1]
model_name = sys.argv[2]
algorythm = sys.argv[3]
export_path = sys.argv[4]

def Evaluate(p, y_test: pd.Series, ):
    print("Getting metrics")
    response_cm = confusion_matrix(y_test, p)
    accuracy = accuracy_score(y_test, p)
    print("Random Forest accuracy: ", accuracy)
    print("Confusion Matrix\n", response_cm)

    print("Saving results")
    export_dict = {}
    details_list = []
    tmp_dict = {}
    tmp_dict['encoding_model'] = model_name
    tmp_dict['ml_algorythm'] = algorythm
    tmp_dict['accuracy'] = accuracy
    tmp_dict['train:test'] = f"{int(train_size*100)}:{int(test_size*100)}"
    tmp_dict['response_cm'] = response_cm.tolist()

    details_list.append(tmp_dict)
    export_dict['training_details'] = details_list
    export_dict['testing_details'] = []

    if not os.path.exists(f"{export_path}results.json"):
        with open(f"{export_path}results.json", "w") as outfile:
            json.dump(export_dict, outfile, indent=4)
    else:
        with open(f"{export_path}results.json", "r+") as openfile:
            json_object = json.load(openfile)
            json_object['training_details'].append(tmp_dict)
            openfile.seek(0)
            json.dump(json_object, openfile, indent=4)



def ExportModel(model):
    print("Exporting model")
    if not os.path.exists(f"{export_path}/models/"):
        os.makedirs(f"{export_path}/models/")
    joblib.dump(model, f"{export_path}/models/{model_name}_{algorythm}.joblib")

def RandomForest(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    print("Initializing Random Forest")
    rf = RandomForestClassifier(n_jobs=-1, random_state=0)
    rf.fit(X_train, y_train)

    print("Making predictions")
    p = rf.predict(X_test)

    Evaluate(p, y_test)
    ExportModel(rf)

def SupportVectorMachine(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    print("Initializing Support Vector Machine")
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)

    print("Making predictions")
    p = svm.predict(X_test)

    Evaluate(p, y_test)
    ExportModel(svm)

def K_NearestNeighbor(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    print("Initializing K Nearest Neighbor")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    print("Making predictions")
    p = knn.predict(X_test)

    Evaluate(p, y_test)
    ExportModel(knn)

def NaiveBayes(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    print("Initializing K Nearest Neighbor")
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    print("Making predictions")
    p = nb.predict(X_test)

    Evaluate(p, y_test)
    ExportModel(nb)

print("Loading Data")
X_train = pd.read_csv(f"{data_path}/train/X_train_{model_name}.csv")
X_test = pd.read_csv(f"{data_path}/test/X_test_{model_name}.csv")
y_train = pd.read_csv(f"{data_path}/train/y_train_{model_name}.csv")
y_test = pd.read_csv(f"{data_path}/test/y_test_{model_name}.csv")

y_train = y_train.squeeze()
y_test = y_test.squeeze()

train_size = len(y_train)/(len(y_test)+len(y_train))
test_size = len(y_test)/(len(y_test)+len(y_train))

match algorythm:
    case "rf":
        RandomForest(X_train, X_test, y_train, y_test)
    case "svm":
        SupportVectorMachine(X_train, X_test, y_train, y_test)
    case "knn":
        K_NearestNeighbor(X_train, X_test, y_train, y_test)
    case "nb":
        NaiveBayes(X_train, X_test, y_train, y_test)

'''
COn esto grafijo el mejor k de knn
k_range = range(1, 20)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(k_range, scores)
    plt.xticks([0, 5, 10, 15, 20])
    plt.show()
'''