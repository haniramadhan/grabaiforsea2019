import numpy as np
import pandas as pd

import time

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def KfoldEvaluation(features, label, model_name, k):
    
    print("Performing K-Fold Evaluation...")
    #just in case someone forgets
    if "bookingID" in features.columns:
        features = features.drop(columns=['bookingID'])
    
    sum_acc = 0
    sum_auc = 0
    sum_avg_time = 0
    kf = KFold(n_splits=k)

    i=0
    
    for train, val in kf.split(features):
        time_start = time.time()
        train_data = np.array(features)[train]
        train_label = np.array(label)[train]
        val_data = np.array(features)[val]
        val_label = np.array(label)[val]
        
        if i==0:
            print("You may take a sip of coffee or doing some light workout... (1/10)")
        elif i == 4:
            print("halfway done... (5/10)")
        elif i == 7:
            print("just wait a little more, a game of Sudoku may entertain you... (8/10)")
        elif i == 9:
            print("Last batch! What? You've done the Sudoku? Cool! .... (10/10)")

        i = i + 1 
        if model_name == 'RandomForest':
            clf = RandomForestClassifier(n_jobs=8,n_estimators=600,max_depth=9)
        else:
            clf = XGBClassifier(n_jobs=8,colsample_bytree = 0.5426283255709721,
                gamma = 0.4151918304767302, max_depth = 2, n_estimators = 700)

        clf.fit(train_data, train_label)
        val_pred  = clf.predict_proba(val_data)[:,1]

        sum_acc = sum_acc + accuracy_score(val_label, 1 * (val_pred >0.5))
        sum_auc = sum_auc + roc_auc_score(val_label, val_pred)

        elapsed_time = time.time() - time_start
        sum_avg_time = sum_avg_time + (elapsed_time/(train_data.shape[0]))

	#we also measure the average time of single instance prediction in ms
    return sum_acc/k, sum_auc/k, (sum_avg_time/k)*1000


def TrainModel(train_data,train_label,model_name):
	print("Building the model, please wait. You may order a nice snack using GrabFood and it's done")
	if model_name == 'RandomForest':
		clf = RandomForestClassifier(n_jobs=8,n_estimators=600,max_depth=9)
	else:
		clf = XGBClassifier(n_jobs=8,colsample_bytree = 0.5426283255709721,
			gamma = 0.4151918304767302, max_depth = 2, n_estimators = 700)
	clf.fit(train_data,train_label)
	return clf