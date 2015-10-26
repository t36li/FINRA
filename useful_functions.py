# /User/bin/python
# coding: utf-8

__author__ = 'Bob Li'
__date__ = '2015.10.01'

import os
import csv
import numpy as np
import pandas as pd
import random
import time
import pdb

def cross_val_score_proba(X,y,nfold,classifier):
	"""
	This function calculates cross_val_score for auc with predicted probabilities
	returns a list of scores
	"""
	from sklearn.cross_validation import KFold
	from sklearn.metrics import roc_auc_score
	
	print 'Running K-fold CV....'
	x_train=X
	y_train=y
	kf = KFold(n=len(x_train),n_folds=nfold,shuffle=True)
	cv_score=list()
	for train_index, test_index in kf:
		X_train_cv, X_test_cv = x_train[train_index], x_train[test_index]
		y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

		## Fit model to this fold ###
		classifier.fit(X_train_cv,y_train_cv)
		y_pred_cv=classifier.predict_proba(X_test_cv)[:,1]
		cv_score.append(roc_auc_score(y_test_cv,y_pred_cv))
	
	return cv_score

def imputeMedian(X_matrix):
	"""
	given a ndarray, loop through columns and impute missing values with median
	median is computed ignoring the nan values in the column
	"""
	for i in range(X_matrix.shape[1]):
		y_pred = X_matrix[:,i]
		median_val = np.median(y_pred[~np.isnan(y_pred)])		
		inds = np.where(np.isnan(y_pred))		
		y_pred[inds]=median_val
		X_matrix[:,i]=y_pred

	return X_matrix

def nearZeroVar(x_array, var_threshold):
	"""
	Given a ndarray, loop through columns and keep only those that have variance above var_threshold
	var_threshold is the percent of constant values
	return: a list of indices that needs to be removed
	"""
	#x_train = x_array
	results = list()
	for i in range(x_array.shape[1]):
		temp_col = x_array[:,i]
		temp_var = np.var(temp_col)
		if temp_var < (var_threshold*(1-var_threshold)):
			results.append(i)
			#x_train=np.delete(x_train, i, axis=1)
	return results
	#return x_train

def computeNan(df):
	"""
	This functions returns a list computing the %NaN for each column given a dataframe
	"""
	results=list()
	for col in df:
		results.append(float(df[col].isnull().sum())/len(df))
	return results

def computeGinis(X, y, weights=None):
	"""
	This function takes in two vectors
	y: binary 0 and 1 labels
	X: a nd-array, feature values, prediction_probabilities, or 0/1 predictions
	weight: numpy array that contains weight for computing weighted GINI. Weighted GINI is implemented in model_validation2.py

	return: a list of gini scores computed using y vs Columns(X)
	Note: absolute value of gini should be examined for selection
	"""
	gini_list = list()

	if weights!=None:
		import model_validation2 as mv

		for i in range(X.shape[1]):
			y_pred = X[:,i]
			
			dataset = np.array([y,y_pred,weights]).T
			dataset = pd.DataFrame(dataset)
			dataset.columns=list(['Y','score','weights'])

			print 'Calculating Gini for feature %i' % i
			gini_list.append(mv.gini(dataset))
	else:
		from sklearn.metrics import roc_auc_score

		for i in range(X.shape[1]):
			y_pred = X[:,i]
			gini_list.append(2*roc_auc_score(y, y_pred) - 1)

	return gini_list

def obtain_dummy_cols(df):
	"""
	given a dataframe, obtain the dummy columns (response is not considered a dummy column)
	dummy column defined as having only 2 different values
	"""
	
	dummy_cols=list()
	for col in df:
		if len(np.unique(np.array(df[col]))) == 2:
			dummy_cols.append(col)
	return dummy_cols

def ridge_dummy_regression(X,y,testData,lambda_val=None):
	"""
	Train ridge L2 Logistic Regression on X,y. Then predict on x_test
	If lambda_val is provided, will just use this parameter for the L2 LR
	otherwise, will run 5-fold CV on C = log(-1.5, 1.5,5)

	This function returns a list of predicted probabilities as a list
	"""
	from sklearn.linear_model import LogisticRegression
	from sklearn.cross_validation import cross_val_score
	from sklearn.metrics import roc_auc_score
	
	Cs=np.logspace(-1.5, 1.5, 5)
	lr = LogisticRegression(penalty='l2')
	cv_list=list()

	if not lambda_val:
		# Fit ridge to various choices of regularization parameter C to select best C
		for c in Cs:
			lr.C = c

			### randomly divide data into 80/20 split ###
			### because response is very sparse ###
			from sklearn.cross_validation import train_test_split
			X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=42)

			lr.fit(X_train,y_train)
			y_pred=lr.predict_proba(X_test)[:,1]
			cv_list.append(roc_auc_score(y_test,y_pred))

		print 'Best lambda based on Ridge Cross-Validation...'
		max_score=np.max(cv_list)
		lambda_val=Cs[cv_list.index(max_score)]
		print 1.0/lambda_val, max_score

	# Train LR with the optimized regularization parameter ###
	lr.C = lambda_val
	lr.fit(X,y)
	proba_lst = lr.predict_proba(testData)[:,1]

	return proba_lst

def get_dummies_removeBase(df_train, df_test, level=10):
	"""
	### This function encodes categorial columns into dummy variables ###
	### df_train and df_test are dataframe of all categorical features
	### Determine number of levels. Take n-1 levels (to treat collinearity) ###
	### Use pandas get_dummies for each column and remove the first ###
	### AKA: One-Hot-Encoding ###
	### Returns: a dataframe of all dummy columns
	"""
	print 'Running dummy label encoding....'
	dummy_train=list()
	dummy_test=list()
	column_names=list()
	for col in df_train:
		
		df_col=df_train[col]
		levels=df_col.unique()

		### DROP all levels > 50 (too much sparsity otherwise) ###
		if (len(levels) <= 50):
			print col
			print df_col.unique()

			levels=levels[:-1] #remove the last one (to treat collinearity)
			for lvl in levels:
				column_names.append(col+'_'+str(lvl))
				train_lvl=np.array(df_train[col]==lvl).astype(int)
				test_lvl=np.array(df_test[col]==lvl).astype(int)
				dummy_train.append(train_lvl)
				dummy_test.append(test_lvl)

	dummy_train=np.array(dummy_train).T
	dummy_test=np.array(dummy_test).T

	dummy_train=pd.DataFrame(dummy_train)
	dummy_train.columns=column_names

	dummy_test=pd.DataFrame(dummy_test)
	dummy_test.columns=column_names

	return dummy_train, dummy_test
