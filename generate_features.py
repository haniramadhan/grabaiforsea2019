import pandas as pd
import numpy as np

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