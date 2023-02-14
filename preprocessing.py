# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:50:45 2019
@author: s-moh
"""
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta

generation = '65_80'
# Path to Original Training Dataset "Clicks" File
dataBefore = './data/raw_data/rakuten-data/'+generation+'/'+generation+'.csv'
# Path to Original Testing Dataset "Clicks" File
dataTestBefore = './data/raw_data/rakuten-data/'+generation+'/'+generation+'.csv'
# Path to Processed Dataset Folder
dataAfter = './data/preprocessed_data/rakuten-data/'+generation+'/'
dayTime = 86400  # Validation Only one day = 86400 seconds


def removeShortSessions(data):
    # delete sessions of length < 1
    # group by sessionID and get size of each session
    sessionLen = data.groupby('SessionID').size()
    data = data[np.in1d(data.SessionID, sessionLen[sessionLen > 1].index)]
    return data


# Read Dataset in pandas Dataframe (Ignore Category Column)
train = pd.read_csv(dataBefore, sep=',', header=0, usecols=[
                    0, 1, 6], dtype={0: np.int32, 1: np.int64, 2: np.int64})
test = pd.read_csv(dataTestBefore, sep=',', header=0, usecols=[
                   0, 1, 6], dtype={0: np.int32, 1: np.int64, 2: np.int64})
train.columns = ['SessionID', 'Time', 'ItemID']  # Headers of dataframe
test.columns = ['SessionID', 'Time', 'ItemID']  # Headers of dataframe

# 2019/1/1を第一週として、Timeを算出する。
d = datetime.date(2019, 1, 1)

# Convert time objects to timestamp
train['Time'] = train.Time.apply(lambda x: datetime.datetime.strptime(
    str((d+timedelta(days=7*int(x)))), '%Y-%m-%d').timestamp())
test['Time'] = test.Time.apply(lambda x: datetime.datetime.strptime(
    str((d+timedelta(days=7*int(x)))), '%Y-%m-%d').timestamp())
# train['Time'] = train.Time.apply(lambda x: datetime.datetime.strptime(
# x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())  # Convert time objects to timestamp


# remove sessions of less than 2 interactions
train = removeShortSessions(train)
# delete records of items which appeared less than 5 times
# groupby itemID and get size of each item
itemLen = train.groupby('ItemID').size()
train = train[np.in1d(train.ItemID, itemLen[itemLen > 4].index)]
# remove sessions of less than 2 interactions again
train = removeShortSessions(train)

# 3
'''
#Separate Data into Train and Test Splits
timeMax = data.Time.max() #maximum time in all records
sessionMaxTime = data.groupby('SessionID').Time.max() #group by sessionID and get the maximum time of each session
sessionTrain = sessionMaxTime[sessionMaxTime < (timeMax - dayTime)].index #training split is all sessions that ended before the last day
sessionTest  = sessionMaxTime[sessionMaxTime >= (timeMax - dayTime)].index #testing split is all sessions has records in the last day
train = data[np.in1d(data.SessionID, sessionTrain)]
test = data[np.in1d(data.SessionID, sessionTest)]
'''
# Delete records in testing split where items are not in training split
test = test[np.in1d(test.ItemID, train.ItemID)]
# Delete Sessions in testing split which are less than 2
test = removeShortSessions(test)

# Convert To CSV
# print('Full Training Set has', len(train), 'Events, ', train.SessionID.nunique(), 'Sessions, and', train.ItemID.nunique(), 'Items\n\n')
# train.to_csv(dataAfter + 'recSys15TrainFull.txt', sep='\t', index=False)
print('Testing Set has', len(test), 'Events, ', test.SessionID.nunique(),
      'Sessions, and', test.ItemID.nunique(), 'Items\n\n')
test.to_csv(dataAfter + 'recSys15Test.txt', sep=',', index=False)

# 3
# Separate Training set into Train and Validation Splits
timeMax = train.Time.max()
sessionMaxTime = train.groupby('SessionID').Time.max()
# training split is all sessions that ended before the last 2nd day
sessionTrain = sessionMaxTime[sessionMaxTime < (timeMax - dayTime)].index
# validation split is all sessions that ended during the last 2nd day
sessionValid = sessionMaxTime[sessionMaxTime >= (timeMax - dayTime)].index
trainTR = train[np.in1d(train.SessionID, sessionTrain)]
trainVD = train[np.in1d(train.SessionID, sessionValid)]
# Delete records in validation split where items are not in training split
trainVD = trainVD[np.in1d(trainVD.ItemID, trainTR.ItemID)]
# Delete Sessions in testing split which are less than 2
trainVD = removeShortSessions(trainVD)
# Convert To CSV
print('Training Set has', len(trainTR), 'Events, ', trainTR.SessionID.nunique(
), 'Sessions, and', trainTR.ItemID.nunique(), 'Items\n\n')
trainTR.to_csv(dataAfter + 'recSys15TrainOnly.txt', sep=',', index=False)
print('Validation Set has', len(trainVD), 'Events, ', trainVD.SessionID.nunique(
), 'Sessions, and', trainVD.ItemID.nunique(), 'Items\n\n')
trainVD.to_csv(dataAfter + 'recSys15Valid.txt', sep=',', index=False)
