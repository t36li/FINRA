# /User/bin/python
# coding: utf-8

__author__ = 'Bob Li'
__date__ = '2015.10.24'

import numpy as np
import pandas as pd
import pdb
import csv
import useful_functions as udf

""" 
This is for FINRA data challenge
First clean data.....
1. Remove columns that are >95% missing
2. Impute missing values with median (replace NaNs with column median ignoring NaNs)
3. Remove columns that are near zero variance

Next...
1. Compute the univariate gini scores (absolute values - as feature selection tool)
2. Discard features with zero gini
3. Fit GBM into the data (train, cv, test)

NEXT STEPS:
1. Model Blending (GBM, RF, LR-L1, LR-L2, KNN)
2. Feature engineering

"""
print 'Reading training data...'
train_data = pd.read_csv('cs-training.csv')
print 'Finished reading training data...'

print 'Reading test data...'
test_data = pd.read_csv('cs-test.csv')
print 'Finished reading test data...'

### drop first column (some just row counts. For some reason, Excel csv appends this column on)
train_data=train_data.iloc[:,1:]
test_data=test_data.iloc[:,2:]

print 'Original train and test data size:'
print train_data.shape, test_data.shape

### Some data exploration ###
print train_data.describe()
### NumberOfOpenCreditLinesAndLoans: 19.8% missing, 
### NumberOfDependents: 2.5% missing

###############################################################################################################
### Impute Numeric columns with median ###
y_train = train_data.values[:,0].astype(int)
x_train = train_data.values[:,1:]
x_test = test_data.values

print 'Imputing missing values with median...'
x_train=udf.imputeMedian(x_train) ### Self implemented, Imputer has a bug that sometimes removes first column
x_test=udf.imputeMedian(x_test)
print 'Finished imputing missing values with median...'

print 'Train and test data feature space:'
print x_train.shape, x_test.shape

###############################################################################################################
### Feature Engineering ###
### Create two new columns ###
### ONE: TotalNumLateDays = All late days sum
### TWO: Income-Expenses

TotalNumLateDays_train = x_train[:,2] + x_train[:,6] + x_train[:,8]
MonthlyLeftOver_train = np.multiply(x_train[:,4],(1-x_train[:,3]))
TotalNumLateDays_train=np.reshape(TotalNumLateDays_train,(len(x_train),1))
MonthlyLeftOver_train=np.reshape(MonthlyLeftOver_train,(len(x_train),1))

TotalNumLateDays_test = x_test[:,2] + x_test[:,6] + x_test[:,8]
MonthlyLeftOver_test = np.multiply(x_test[:,4],(1-x_test[:,3]))
TotalNumLateDays_test=np.reshape(TotalNumLateDays_test,(len(x_test),1))
MonthlyLeftOver_test=np.reshape(MonthlyLeftOver_test,(len(x_test),1))

### Append feature onto original training set ###
x_train=np.concatenate((x_train,TotalNumLateDays_train,MonthlyLeftOver_train),axis=1)
x_test=np.concatenate((x_test,TotalNumLateDays_test,MonthlyLeftOver_test),axis=1)

###############################################################################################################
## Obtain column names for training set (test and train should have same column names now) ##
## This is used for outputing the significant GINI scores list ##
## MAKE SURE THESE ARE MATCHED CORRECTLY! ##
feature_names = list(train_data.columns.values)
feature_names.remove('SeriousDlqin2yrs')
feature_names.append('TotalNumLateDays')
feature_names.append('MonthlyLeftOver')
feature_names = np.array(feature_names)
print len(feature_names), x_train.shape[1], x_test.shape[1] #All three should match

### POSSIBLE OPTION: ###
### Remove binary columns with near zero variance ###
### near zero variance defined as 1 value takes more than 99.5% of the column ###
removeZeroVar=False
if removeZeroVar:
	nearZero_idx=udf.nearZeroVar(x_train, 0.95) ## get this to return a list of features instead...
	nearZeroCols=feature_names[nearZero_idx]
	print 'Zero variance volumns: ' 
	print nearZeroCols

	x_train=np.delete(x_train, nearZero_idx, axis=1)
	x_test=np.delete(x_test,nearZero_idx,axis=1)
	print 'Train and test data feature space after dropping nearZeroCols:'
	print x_train.shape, x_test.shape
	feature_names=np.delete(feature_names,nearZero_idx)

### Compute the univariate gini scores on the training dataset###
### Add a third arguement if wants to calculate GINI weighted ###
### For example udf.computeGinis(x,y,weights)
gini_list = udf.computeGinis(x_train, y_train)

write_to_file=True
if write_to_file:
	print 'Writing GINI results file....'
	with open('Train_GINI.csv','wb') as testfile:
		w=csv.writer(testfile)
		for i in range(len(gini_list)):
			w.writerow((feature_names[i],gini_list[i]))
	testfile.close()
	print 'File written to disk...'

### Final Check ###
print 'Final Train and test data:'
print x_train.shape, x_test.shape
print 'NAN values remaining (SHOULD BE 0)!'
print np.isnan(x_train).sum() #0
print np.isnan(x_test).sum() #0

#### Save cleaned train and test into folder ###
final_train=np.concatenate((np.reshape(y_train,(len(y_train),1)),x_train),axis=1)

np.savetxt("bob_cleaned_train.csv", final_train, fmt='%.4f', delimiter=',', newline='\n')
np.savetxt("bob_cleaned_test.csv", x_test, fmt='%.4f', delimiter=',', newline='\n')

##########################################################################################################
