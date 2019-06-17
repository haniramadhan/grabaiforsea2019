import pandas as pd
import numpy as np
import os
import math


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
		return df

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
	series_book1 = df.loc[:,'bookingID'].append(pd.Series([-1]), ignore_index=True)
	series_book2 = pd.Series([-1]).append(df.loc[:,'bookingID'], ignore_index=True)
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