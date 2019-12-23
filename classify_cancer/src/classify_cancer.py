"""
Classifies cancer type using three classification algorithms, finding
the settings (hyperparameter values) that produce the most accurate results.

Authors: Drew Hager and Madelyn Gatchel

Time Spent: 20 hours
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn import svm, tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


FEATURE_SEL_THRESHOLD = 300

RANDOM_STATE = 150
MAX_K = 11


def get_data(path):
    """
    Reads patient data from .txt files, formats data and generates target
    labels.

    Parameters:
        path - string representing path to data

    Returns:
        df - Pandas dataframe containing features and targets
    """
    class_breakdown = {} # keeps track of num examples per class/cancer type
    df = pd.DataFrame()
    last_column = [] # target labels
    for cancer_type in os.listdir(path):
        cancer_folder_path = path + '/' + cancer_type
        for patient_folder in os.listdir(cancer_folder_path):
            if 'MANIFEST' in patient_folder:
                continue
            patient_folder_path = cancer_folder_path + '/' + patient_folder
            for patient_file in os.listdir(patient_folder_path):
                if patient_file != "annotations.txt":
                    # get miRNA data for current patient and append to df
                    file_path = patient_folder_path + '/' + patient_file
                    frame = pd.read_csv(file_path, sep='\t')
                    df = df.append(frame['reads_per_million_miRNA_mapped'])

                    # Update class breakdown
                    if cancer_type not in class_breakdown:
                        class_breakdown[cancer_type] = 1
                    else:
                        class_breakdown[cancer_type] += 1
                    # Append target label
                    last_column.append(cancer_type)
    df['cancer_type'] = last_column # Add target data
    df = df.reset_index(drop=True) # drop row labels "read"
    features = df.iloc[:, 0:(df.shape[1] - 1)]
    targets = pd.DataFrame(df['cancer_type'])
    print(class_breakdown)
    # features.to_csv("features.csv")
    # targets.to_csv("targets.csv")
    # df.to_csv("cancer_data.csv")
    return df

def prepare_data(cancer_data):
    """
    Prepares the data: splits the data into train, validate, and test sets;
    performs feature selection; normalizes feature data.

    Parameters:
        cancer_data - string representing path to data

    Returns:
        x_train - feature training data
        x_validate - feature validation data
        x_test - feature test data
        y_train - classifications for test data
        y_validate - classifications for validation data
        y_test - classifications for test data
    """
    features = np.array(cancer_data.iloc[:, 0:(cancer_data.shape[1] - 1)])
    targets = np.array(cancer_data['cancer_type'])
    x_train, x_test, y_train, y_test = train_test_split(features, targets, \
                                                        stratify=targets, \
                                                        test_size = 0.2, \
                                                      random_state=RANDOM_STATE)
    x_validate, x_test, y_validate, y_test = train_test_split(x_test, y_test, \
                                                              stratify=y_test, \
                                                              test_size = 0.5, \
                                                      random_state=RANDOM_STATE)
    print(x_train.shape)
    feature_selection = VarianceThreshold(threshold=FEATURE_SEL_THRESHOLD)
    x_train = feature_selection.fit_transform(x_train)
    feature_mask = feature_selection.get_support()
    del_features = np.array([i if feature_mask[i] == False else -1 \
                            for i in range(feature_mask.shape[0])])
    del_features = del_features[del_features != -1]
    x_validate = np.delete(x_validate, del_features, axis=1)
    x_test = np.delete(x_test, del_features, axis=1)
    x_train = normalize(x_train, axis=1)
    x_validate = normalize(x_validate, axis=1)
    x_test = normalize(x_test, axis=1)
    print(x_train.shape)
    return x_train, x_validate, x_test, y_train, y_validate, y_test

def cancer_class_DT(path, test):
    """
    Variations on the Decision Trees algorithm. If test == False,
    the function performs DT algorithm on the validation data to find the best
    hyperparameter values. If test == True, the function uses the best model
    (found in the validation stage) on the test data. In either case, the
    function prints the F1 score for the given model(s).

    Uses various sci-kit learn functions.

    Parameters:
        path - string representing path to data
        test - boolean that tells which stage we are in; if True, then in test
               stage; else in validation stage

    Returns:
        NONE
    """
    cancer_data = get_data(path)
    x_train, x_validate, x_test, \
    y_train,y_validate,y_test = prepare_data(cancer_data)

    if test:
        treeCLF = tree.DecisionTreeClassifier(random_state=RANDOM_STATE, \
                                              criterion="gini", \
                                              splitter="random", \
                                              max_features=None)
        treeCLF = treeCLF.fit(x_train, y_train)
        tree_y_pred = treeCLF.predict(x_test)
        print(classification_report(tree_y_pred, y_test))

    else:
        for criterion in ["gini", "entropy"]:
            for splitter in ["best", "random"]:
                for max_features in ["sqrt", "log2", None]:
                    treeCLF = tree.DecisionTreeClassifier(random_state=RANDOM_STATE, \
                    criterion=criterion, splitter=splitter, \
                    max_features = max_features)
                    treeCLF = treeCLF.fit(x_train, y_train)
                    tree_y_pred = treeCLF.predict(x_validate)
                    print("Testing DT with ", criterion, "criterion, ", \
                    splitter, " splitter, and ", max_features, " max_features")
                    #print(confusion_matrix(tree_y_pred, y_validate))
                    print(classification_report(tree_y_pred, y_validate))


def cancer_class_SVM(path, test):
    """
    Variations on the linear Support Vector Machine algorithm. If test == False,
    the function performs SVM on the validation data to find the best
    hyperparameter values. If test == True, the function uses the best model
    (found in the validation stage) on the test data. In either case, the
    function prints the F1 score for the given model(s).

    Uses various sci-kit learn functions.

    Parameters:
        path - string representing path to data
        test - boolean that tells which stage we are in; if True, then in test
               stage; else in validation stage

    Returns:
        NONE
    """
    cancer_data = get_data(path)
    x_train, x_validate, x_test, \
    y_train,y_validate,y_test = prepare_data(cancer_data)

    if test:
        svmCLF = svm.LinearSVC(random_state=RANDOM_STATE)
        svmCLF.fit(x_train, y_train)
        svm_y_pred = svmCLF.predict(x_test)
        print(confusion_matrix(svm_y_pred, y_test))
        print(classification_report(svm_y_pred, y_test))
    else:
        for penalty in ['l2']:
            for loss in ['squared_hinge', 'hinge']:
                for fit_intercept in [True, False]:
                    print("Testing SVM with ", penalty, " penalty", loss, \
                    "loss and fit_intercept equal to", fit_intercept)
                    svmCLF = svm.LinearSVC(penalty=penalty, loss=loss, \
                    fit_intercept=fit_intercept, random_state=RANDOM_STATE,\
                    max_iter=5000)
                    svmCLF.fit(x_train, y_train)
                    svm_y_pred = svmCLF.predict(x_validate)
                    #print(confusion_matrix(svm_y_pred, y_validate))
                    print(classification_report(svm_y_pred, y_validate))

def cancer_class_kNN(path, test):
    """
    Variations on the k-Nearest Neighbors algorithm. If test == False,
    the function performs kNN on the validation data to find the best
    hyperparameter values. If test == True, the function uses the best model
    (found in the validation stage) on the test data. In either case, the
    function prints the F1 score for the given model(s).

    Uses various sci-kit learn functions.

    Parameters:
        path - string representing path to data
        test - boolean that tells which stage we are in; if True, then in test
               stage; else in validation stage

    Returns:
        NONE
    """
    cancer_data = get_data(path)
    x_train, x_validate, x_test, \
    y_train,y_validate,y_test = prepare_data(cancer_data)

    if test:
        distance_metric = "l1"
        k = 4
        weights = "uniform"
        neigh = KNeighborsClassifier(n_neighbors=k, \
                                     weights=weights, p=1)
        neigh.fit(x_train, y_train)
        y_pred = neigh.predict(x_test)
        print("Results for k = ", k, ", distance metric = ", \
              distance_metric, ", weights preference = ", weights)
        print(classification_report(y_pred, y_test))
    else:
        for weights in ['uniform', 'distance']: # varying weights hyperparameter
            for p in range(1, 3): # varying distance metric - 1 = L
                for k in range(1,MAX_K):
                    neigh = KNeighborsClassifier(n_neighbors=k, \
                                                 weights=weights, p=p)
                    neigh.fit(x_train, y_train)
                    y_pred = neigh.predict(x_validate)
                    if p == 1:
                        distance_metric = "l1"
                    else:
                        distance_metric = "l2"
                    print("Results for k = ", k, ", distance metric = ", \
                          distance_metric, ", weights preference = ", weights)
                    print(classification_report(y_pred, y_validate))


def main():
    path = "../data"
    # cancer_class_DT(path, False)
    # cancer_class_SVM(path, False)
    # cancer_class_kNN(path, False)
    cancer_class_SVM(path, True)



if __name__ == "__main__":
    main()
