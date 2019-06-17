import pandas as pd
import numpy as np
import preprocess
import generate_features as gf
import evaluation as ev

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def CleanAndPreprocess(features, label=None):

	print("Still preprocessing...")
	
	start_trip = preprocess.Findstart_trip(features)
	non_consecutive_rows = preprocess.FindNonconsecutiveTrips(features,start_trip)

	sdf = preprocess.SmoothenFeatures(features,3,start_trip,non_consecutive_rows) #smoothed data
	sdf = gf.GenerateNewFeatures(sdf,start_trip,non_consecutive_rows)
	trip_length = gf.FindTripLength(sdf)
	agg_features = gf.AggregateAll(sdf,trip_length)

	if label is None:
		return agg_features
	else:
		label = preprocess.DeduplicateLabel(label)

	return agg_features, label


#setting the train folders
print("Loading Train Data...")
features, label = preprocess.LoadData("../features/","../labels/")
train_data, label = CleanAndPreprocess(features,label)

# #Random Forest Evaluation using Full Feature
# acc,auc,rtime = KfoldEvaluation(train_data,label['label'],"RandomForest",10)
# print("RandomForest: accuracy: "+str(acc)+" AUC: "+str(auc)+
# 	" Running time (ms): " +str(rtime))

#XGBoost Evaluation using Full Feature
acc,auc,rtime = ev.KfoldEvaluation(train_data,label['label'],"XGBoost",10)
print("XGBoost: accuracy: "+str(acc)+" AUC: "+str(auc)+
	" Running time (ms): " +str(rtime))

#training the model, we use the XGBoost
xgb_clf = ev.TrainModel(train_data,label['label'],"XGBoost")


#======================================================================
#setting the test folders
print("Loading Test Data...")

#if test label is not available, use this. If available, use commented codes below these codes after ------ line
#(!)change these paths to the desired test folders
test_features = preprocess.LoadData("../features/")
test_data = CleanAndPreprocess(test_features)

predicted_label = pd.DataFrame([])
predicted_label['bookingID'] = test_data['bookingID']

#prediction part
predicted_label['Unsafe'] = xgb_clf.predict(test_data) #use this one for getting rigid labels
#predicted_label['Unsafe'] = xgb_clf.predict_proba(test_data).loc[:,1] #use this one for getting probability labels

predicted_label.to_csv("output.csv",header=True,index=False)
# #------------------------------------------------------------------------

# ##if test label is available, use these codes. If not use, the codes above.
# #(!)change these paths to the desired test folders
# test_features,test_label = LoadData("../features/","../labels/")
# print(test_features.shape)
# test_data,test_label = CleanAndPreprocess(test_features,test_label)

#prediction part
# predicted_label['bookingID'] = test_data['bookingID']
# predicted_label['Unsafe'] = xgb_clf.predict(test_data) #use this one for getting rigid labels
# #predicted_label['Unsafe'] = xgb_clf.predict_proba(test_data).loc[:,1] #use this one for getting probability labels
# predicted_label.to_csv("output.csv",header=True,index=False)

# print("Test Accuracy: " + accuracy_score(test_label, 1 * (predicted_label['Unsafe'] >0.5) ))
# print("Test AUC: " +  roc_auc_score(test_label, predicted_label['Unsafe']))
