import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

def LoadData(feature_folderpath, label_folderpath=None):
	frames = []
	for root, dirs, files in os.walk(feature_folderpath):
		for file in files:
			if file.endswith(".csv"):
			    df = pd.read_csv(feature_folderpath+file)
			    frames.append(df)
	df = pd.concat(frames)
	df = df.sort_values(by=['bookingID', 'second']) #we sort it by bookingID and seconds by convenience
	df = df.reset_index(drop=True)

	if label_folderpath is not None:
		frames = []
		for root, dirs, files in os.walk(label_folderpath):
			for file in files:
				if file.endswith(".csv"):
				    label = pd.read_csv(label_folderpath+file,index_col=False)
				    frames.append(label)
		label = pd.concat(frames)
		label.sort_values(by=['bookingID'])
		label = label.reset_index(drop=True)
	else:
		label = None

	return df,label

def DeduplicateLabel(label):

	bookingCount = label.groupby(['bookingID']).count()

	#Showing IDs with duplicate labels
	dupIDs = bookingCount[bookingCount>1].dropna().reset_index()['bookingID']
	#using latest pick
	lastPick = label[label['bookingID'].isin(dupIDs)].reset_index(drop=True).iloc[1::2]
	#We only change the duplicate ID whose final label is 1, why? We'll use MIN aggregate groupby later.
	lastPickIDs1 = lastPick.loc[lastPick['label']==1,'bookingID'].reset_index(drop=True)

	labelc = label.groupby(['bookingID'],as_index=False).min() #cleaned label
	labelc.loc[labelc['bookingID'].isin(lastPickIDs1),'label']=1
	return labelc

def Findstart_trip(df):
	series_book1 = df['bookingID'].append(pd.Series([-1]), ignore_index=True)
	series_book2 = pd.Series([-1]).append(df['bookingID'], ignore_index=True)
	start_trip = ((series_book1 - series_book2)[:-1]>0)
	return start_trip


def FindNonconsecutiveTrips(df,start_trip):
	seconds = df.loc[:,['bookingID','second']]

	#check for difference
	series_out1 = pd.Series([0]).append(seconds['second'],ignore_index=True)
	series_out2 = seconds['second'].append(pd.Series([0]),ignore_index=True)
	seconds['second_diff'] = (series_out2 - series_out1)[:-1]

	#setting the second_diff of the beginning of the trip is always 0
	seconds.loc[start_trip, 'second_diff'] = 0

	second_diff = seconds['second_diff']
	non_consecutive_rows = (seconds['second_diff']>1)

	return non_consecutive_rows

def SmoothMWA(rows, isBearing, l,start_trip,non_consecutive_rows):
    #To make sure we did not use any values that are not are the start of the trip nor the non-consecutive seconds.
    a= 1*~ (start_trip | non_consecutive_rows )
    shift = rows
    rowsShift = rows
    for i in range(l-1):
        shift = pd.Series([0]).append(shift, ignore_index=True)[:-1] 
        dupRowsA = (a*shift)+((1-a) * rows)
        rowsShift = pd.concat([rowsShift,dupRowsA],axis=1)
        a = a * pd.Series([0]).append(a, ignore_index=True)[:-1] 

    if isBearing:
        rowsShift = rowsShift - 180
        rowsShift = np.radians(rowsShift)
        sina = np.sin(rowsShift).mean(axis=1)
        cosa = np.cos(rowsShift).mean(axis=1)
        rowsShift = np.degrees(np.arctan2(sina,cosa))
        return rowsShift
    else:
        return (rowsShift/l).sum(axis=1)

def SmoothenFeatures(df,l,start_trip,non_consecutive_rows):
	smoothingColumns = ['Bearing', 'acceleration_x', 'acceleration_y',
	    'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z','Speed']
	sdf = df.loc[:,['bookingID','Accuracy','second']]

	for column in smoothingColumns:
		if column=='Bearing':
			sdf[column] = SmoothMWA(df[column],True,l,start_trip,non_consecutive_rows)
		else :
			sdf[column] = SmoothMWA(df[column],False,l,start_trip,non_consecutive_rows)

	return sdf


def ComputeResultant(data_xyz,isAcc=False):
    data_r = np.sqrt(np.power(data_xyz.iloc[:,0],2)
                          +np.power(data_xyz.iloc[:,1],2)
                          +np.power(data_xyz.iloc[:,2],2))

    if isAcc:
        data_r = np.abs(data_r-9.8)
    
    return data_r

def ComputeChange(data, isBearing, start_trip, non_consecutive_rows):
    #calculating the change of speed between time step, for bearing: angularSpeed
    series_data1 = data.append(pd.Series([0]), ignore_index=True)
    series_data2 = pd.Series([0]).append(data, ignore_index=True)
    d_data = (series_data1 - series_data2)[:-1]

    #marking the beginning of the trip
    d_data[start_trip|non_consecutive_rows] = 0
    
    if isBearing:
        #Negative change is equal to the positive change in degree
        d_data = np.abs(d_data)

        #The bearing change >180 must be re-computed as the angle change > 180 is equal to angle change <=180.
        ## This is caused by [0...360] range
        d_data.loc[d_data>180] = 360 - d_data[d_data>180]
    return d_data

def IntegrateAcc(acc_r,start_trip, non_consecutive_rows):
    #integrating the acceleration for each time step, duplicates and appends a 0 value
    series_acc1 = acc_r.append(pd.Series([0]), ignore_index=True)
    series_acc2 = pd.Series([0]).append(acc_r, ignore_index=True)

    #Setting the beginning of the trip is always 0
    start_tripS = start_trip.append(pd.Series([True]), ignore_index=True)
    series_acc2[start_tripS|non_consecutive_rows] = 0

    #Computing the integration
    i_acc = ((series_acc1 + series_acc2)/2)[:-1]

    return i_acc

def GenerateNewFeatures(df,start_trip, non_consecutive_rows):
	df.loc[:,'acc_r'] = ComputeResultant(df[['acceleration_x','acceleration_y','acceleration_z']],True)
	df.loc[:,'d_Speed'] = ComputeChange(df['Speed'],False, start_trip, non_consecutive_rows)
	df.loc[:,'i_acc'] = IntegrateAcc(df['acc_r'], start_trip, non_consecutive_rows)
	df.loc[:,'gyro_r'] = ComputeResultant(df[['gyro_x','gyro_y','gyro_z']],False)
	df.loc[:,'d_Bearing'] = ComputeChange(df['Bearing'],True, start_trip, non_consecutive_rows)
	return df

def FindTripLength(df):
	trip_max_length = df[['bookingID','second']].groupby(['bookingID']).max()
	trip_min_length = df[['bookingID','second']].groupby(['bookingID']).min() 

	trip_length = trip_max_length - trip_min_length

	return trip_length

def AggregateAll(features,trip_length):
    f_nosec = features.drop(columns=['second'])
    f_agg_mean= f_nosec.groupby(['bookingID'], as_index=False).mean()
    f_agg_max = f_nosec.groupby(['bookingID'], as_index=False).max()
    f_agg_min = f_nosec.groupby(['bookingID'], as_index=False).min()
    f_agg_var = f_nosec.groupby(['bookingID'], as_index=False).var()

    #renaming the columns
    col_mean = ['bookingID']
    col_max = ['bookingID']
    col_min = ['bookingID']
    col_var = ['bookingID']

    colnames_default = f_nosec.columns
    for colname in colnames_default:
        if colname=='bookingID':
            continue
        col_mean.append(colname+"_mean")
        col_max.append(colname+"_max")
        col_min.append(colname+"_min")
        col_var.append(colname+"_var")

    f_agg_mean.columns= col_mean
    f_agg_max.columns = col_max
    f_agg_min.columns = col_min
    f_agg_var.columns = col_var

    #joining all the aggregations
    f_agg = pd.merge(f_agg_mean,f_agg_max,on="bookingID")
    f_agg = pd.merge(f_agg,f_agg_min,on="bookingID")
    f_agg = pd.merge(f_agg,f_agg_var,on="bookingID")
    f_agg = pd.merge(f_agg,trip_length,on="bookingID")
    
    return f_agg

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
       	print(i) 
       	i=i+1
        train_data = np.array(features)[train]
        train_label = np.array(label)[train]
        val_data = np.array(features)[val]
        val_label = np.array(label)[val]
        
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
 	if model_name == 'RandomForest':
 		clf = RandomForestClassifier(n_jobs=8,n_estimators=600,max_depth=9)
 	else:
 		clf = XGBClassifier(n_jobs=8,colsample_bytree = 0.5426283255709721,
 			gamma = 0.4151918304767302, max_depth = 2, n_estimators = 700)
 	clf.fit(train_data,train_label)
 	return clf

def CleanAndPreprocess(features, label=None):
	if label is not None:
		label = DeduplicateLabel(label)
	else:
		label = None

	print("Still preprocessing...")
	
	start_trip = Findstart_trip(features)
	non_consecutive_rows = FindNonconsecutiveTrips(features,start_trip)

	sdf = SmoothenFeatures(features,3,start_trip,non_consecutive_rows) #smoothed data
	sdf = GenerateNewFeatures(sdf,start_trip,non_consecutive_rows)
	trip_length = FindTripLength(sdf)
	agg_features = AggregateAll(sdf,trip_length)
	return agg_features, label


#setting the train folders
print("Loading Train Data...")
features, label = LoadData("../features/","../labels/")
train_data, label = CleanAndPreprocess(features,label)

#Random Forest Evaluation using Full Feature
acc,auc,rtime = KfoldEvaluation(train_data,label['label'],"RandomForest",10)
print("RandomForest: accuracy: "+str(acc)+" AUC: "+str(auc)+
	" Running time (ms): " +str(rtime))

#XGBoost Evaluation using Full Feature
acc,auc,rtime = KfoldEvaluation(train_data,label['label'],"XGBoost",10)
print("XGBoost: accuracy: "+str(acc)+" AUC: "+str(auc)+
	" Running time (ms): " +str(rtime))

#training the model, we use the XGBoost
xgb_clf = TrainModel(train_data,label['label'],"XGBoost")


#======================================================================
#setting the test folders
print("Loading Test Data...")

#if test label is not available, use this. If available, use commented codes below these codes
#(!)change these paths to the desired test folders
test_features = LoadData("../features/")
test_data, _ = CleanAndPreprocess(test_features)

predicted_label['bookingID'] = test_data['bookingID']

predicted_label['Unsafe'] = xgb_clf.predict(test_data) #use this one for getting rigid labels
#predicted_label['Unsafe'] = xgb_clf.predict_proba(test_data).loc[:,1] #use this one for getting probability labels

predicted_label.to_csv("output.csv",header=True,index=False)
#------------------------------------------------------------------------

##if test label is available, use these codes. If not use, the codes above.
#(!)change these paths to the desired test folders
test_features,test_label = LoadData("../features/","../labels/")
test_data,test_label = CleanAndPreprocess(test_features,test_label)

predicted_label['bookingID'] = test_data['bookingID']
predicted_label['Unsafe'] = xgb_clf.predict(test_data) #use this one for getting rigid labels
#predicted_label['Unsafe'] = xgb_clf.predict_proba(test_data).loc[:,1] #use this one for getting probability labels
predicted_label.to_csv("output.csv",header=True,index=False)

print("Test Accuracy: " + accuracy_score(test_label, 1 * (predicted_label['Unsafe'] >0.5) ))
print("Test AUC: " +  roc_auc_score(test_label, predicted_label['Unsafe']))
