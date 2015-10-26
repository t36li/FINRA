# /User/bin/python
# coding: utf-8

__author__ = 'Bob Li'
__date__ = '2015.10.24'

import csv
import numpy as np
import pandas as pd
from time import time
import pdb
import useful_functions as udf
import matplotlib.pyplot as plt

print 'Reading training data...'
train_data = pd.read_csv('bob_cleaned_train.csv',header=None)
print 'Finished reading training data...'

print 'Reading test data...'
test_data = pd.read_csv('bob_cleaned_test.csv', header=None)
print 'Finished reading test data...'

y_train = train_data.values[:,0].astype(int)
x_train = train_data.values[:,1:]
x_test = test_data.values

"""
Model 1: Fit Gradient Boosting Classifier (To see variable interactions)
1. Run 5-fold Cross-Validation (or GridSearchCV - lock in depth, child weight, first)
2. Plotting the train, cv, and test error
3. Select the optimal parameters

Model 2+3: Fit Logistic Regression with L2, L1 (To see linear relationships)

Model 4: Fit random forest (To complement GBM overfit possibility)

Model 5: Fit KNN (To see non-linear relationships)

Final: Blend models 1-5 using a) average, b) LR-Ridge (fancier weighted average)
"""
##########################################################################################################
def plot_feature_importance(est, names):
	# sort importances
	indices = np.argsort(est.feature_importances_)
	# plot as bar chart
	plt.barh(np.arange(len(names)), est.feature_importances_[indices])
	plt.yticks(np.arange(len(names)) + 0.25, np.array(names)[indices])
	plt.xlabel('Relative importance')
	plt.show()

def logistic_regression(x_train,y_train,x_test,penalty='L2', regularization=1.0, do_CV=False):
	from sklearn.linear_model import LogisticRegression
	from sklearn.cross_validation import KFold

	### Mean Normalize variables before regression ###
	from sklearn.preprocessing import StandardScaler
	ss=StandardScaler()
	x_train=ss.fit_transform(x_train)
	x_test=ss.fit_transform(x_test)

	lr=LogisticRegression()	
	
	if penalty=='L1':
		lr = LogisticRegression(penalty='l1')
		filename="Lasso_submission.csv"
	else:
		lr = LogisticRegression(penalty='l2')
		filename="Ridge_submission.csv"
	
	if do_CV:
		Cs=np.logspace(-1.5, 1.5, 10)
		cv_list=list()

		### Fit lasso to various choices of regularization parameter C to select optimal C
		for c in Cs:
			lr.C = c
			print 'Running K-fold CV with C = %.5f' % (1.0/c)
			cv_scores=udf.cross_val_score_proba(x_train,y_train,5,lr)
			cv_list.append(np.mean(cv_scores))

		print 'Best lambda based on Cross-Validation...'
		max_score=np.max(cv_list)
		max_lambda=Cs[cv_list.index(max_score)]
		print 1.0/max_lambda, max_score
	else:
		print 'Making prediction with optimal lambda....'
		lr.C=1.0/regularization
		lr.fit(x_train,y_train)
		y_pred=lr.predict_proba(x_test)[:,1]

		print 'Coefficients of the regression:'
		print lr.coef_

		print 'Writing submission file....'
		with open(filename,'wb') as testfile:
			w=csv.writer(testfile)
			w.writerow(('Id','Probability'))
			for i in range(len(y_pred)):
				w.writerow(((i+1),y_pred[i]))
		testfile.close()
		print 'File written to disk...'

def GBM(x_train,y_train,x_test,udf_trees=100,udf_lr=0.01,udf_max_depth=5,udf_minsam=50,do_CV=False,names=None):
	### GridSearchCV for GradientBoostingClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.metrics import roc_auc_score

	if do_CV:
		param_grid = {'max_depth': [2,3,4,5],
						'min_samples_leaf':[50,250,1000,2500]}

		est=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1, verbose=1)
		cv_scores=list()
		params_list=list()

		start = time()
		for mdep in param_grid['max_depth']:
			for minSamples in param_grid['min_samples_leaf']:
				print 'Trying parameter combination: (Max_Depth=%i, minSamples=%i)' % (mdep,minSamples)
				est.min_samples_leaf=minSamples
				est.max_depth=mdep

				cv_score=udf.cross_val_score_proba(x_train,y_train,5,est)
				cv_scores.append(np.mean(cv_score))

				### Create the labels for display purposes ###
				params_list.append((mdep,minSamples))

		print 'Took %.2f seconds for parameter tuning.' %(time()-start)
		print 'writing CV results to file...'
		results = np.array([params_list,cv_scores]).T ## should have 48 results...

		print 'GBM Parameter tuning results........'
		print 'Parameters (max_depth, min_samples_in_leaf), CV_Scores'
		for i in range(len(results)):
			print results[i]
	else:
		### Train the GBM Classifier with the optimal parameters found above ###
		print 'Fitting GBM with optimal user-defined parameters....'
		est=GradientBoostingClassifier(n_estimators=udf_trees,learning_rate=udf_lr,max_depth=udf_max_depth,min_samples_leaf=7500,verbose=1)
		est.fit(x_train,y_train)

		y_pred=est.predict_proba(x_test)[:,1] ## Must predict probability!! ##

		### Plot feature importances ###
		plot_feature_importance(est, names)

		print 'Writing submission file....'
		with open('GBM_Submission.csv','wb') as testfile:
			w=csv.writer(testfile)
			w.writerow(('Id','Probability'))
			for i in range(len(y_pred)):
				w.writerow(((i+1),y_pred[i]))
		testfile.close()
		print 'File written to disk...'

def RFC(x_train,y_train,x_test,udf_trees=100,udf_max_features='auto', udf_min_samples=50, do_CV=False,names=None):

	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import roc_auc_score

	if do_CV:
		param_grid = {'max_features': [2,3,4],
						'min_samples_leaf':[50,250,1000,2500]}

		est=RandomForestClassifier(n_estimators=100,verbose=1)
		cv_scores=list()
		params_list=list()

		start = time()
		for mfeatures in param_grid['max_features']:
			for minSamples in param_grid['min_samples_leaf']:
				print 'Trying parameter combination: (MaxFeatures=%i, minSamples=%i)' % (mfeatures,minSamples)
				est.min_samples_leaf=minSamples
				est.max_features=mfeatures

				cv_score=udf.cross_val_score_proba(x_train,y_train,5,est)
				cv_scores.append(np.mean(cv_score))

				### Create the labels for display purposes ###
				params_list.append((mfeatures,minSamples))

		print 'Took %.2f seconds for parameter tuning.' %(time()-start)
		print 'writing CV results to file...'
		results = np.array([params_list,cv_scores]).T ## should have 48 results...

		print 'Parameter tuning results........'
		print 'Parameters (max_features, min_samples_leaf), CV_Scores'
		for i in range(len(results)):
			print results[i]
	else:
		### Train the RFC Classifier with the optimal parameters found above ###
		print 'Fitting Random Forest with optimal user-defined parameters....'
		est=RandomForestClassifier(n_estimators=udf_trees, max_features=udf_max_features,min_samples_leaf=udf_min_samples,verbose=1)
		est.fit(x_train,y_train)
		y_pred=est.predict_proba(x_test)[:,1] ## Must predict probability!! ##

		### Plot feature importances ###
		plot_feature_importance(est, names)

		print 'Writing submission file....'
		with open('RFC_Submission.csv','wb') as testfile:
			w=csv.writer(testfile)
			w.writerow(('Id','Probability'))
			for i in range(len(y_pred)):
				w.writerow(((i+1),y_pred[i]))
		testfile.close()
		print 'File written to disk...' 

def KNN(x_train,y_train,x_test, udf_kneighbors=100, do_CV=False):
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.cross_validation import train_test_split
	from sklearn.metrics import roc_auc_score

	### variables may be in different scales, so mean standardize the variables ###
	### Mean Normalize variables before regression ###
	from sklearn.preprocessing import StandardScaler
	ss=StandardScaler()
	x_train=ss.fit_transform(x_train)
	x_test=ss.fit_transform(x_test)

	neigh=KNeighborsClassifier(weights='distance')	
	if do_CV:
		k_list=[25,125,255,387] #important to have odd numbers

		### Try different parameters of K for optimal value ###
		### Randomly divide training set into 80/20 split ###
		cv_score=list()		
		for k in k_list:
			neigh.n_neighbors=k
			x_train_cv, x_test_cv, y_train_cv, y_test_cv = train_test_split(x_train,y_train,test_size=0.20, random_state=42)

			neigh.fit(x_train_cv,y_train_cv)
			y_pred=neigh.predict_proba(x_test_cv)[:,1]
			cv_score.append(roc_auc_score(y_test_cv,y_pred))			

		neigh.fit(x_train,y_train)
		y_pred=neigh.predict_proba(x_test)[:,1]

		print 'Cross Validation KNN Results........'
		print 'Parameters, CV_Scores'
		for i in range(len(cv_score)):
			print k_list[i], cv_score[i]
	else:
		print 'Making Prediction with optimal K neighbors...'
		neigh.n_neighbors=udf_kneighbors
		neigh.fit(x_train,y_train)
		y_pred=neigh.predict_proba(x_test)[:,1]
		print 'Writing submission file....'
		with open('KNN_Submission.csv','wb') as testfile:
			w=csv.writer(testfile)
			w.writerow(('Id','Probability'))
			for i in range(len(y_pred)):
				w.writerow(((i+1),y_pred[i]))
		testfile.close()
		print 'File written to disk...'

def model_blend(x_train,y_train, option='average'):
	"""
	UNFISNIHED...TO BE DONE
	"""
	
	if option=='average':
		y_pred = np.mean(x_train, axis=1) ## average along rows
	else:
		### blend models using L2 LR ###
		logistic_regression(x_train,y_train,x_train,penalty='L2', regularization=14.6779926762, do_CV=False)


feature_names = ['RevolvingUtilizationOfUnsecuredLines',
'age',
'NumberOfTime30-59DaysPastDueNotWorse',
'DebtRatio',
'MonthlyIncome',
'NumberOfOpenCreditLinesAndLoans',
'NumberOfTimes90DaysLate',
'NumberRealEstateLoansOrLines',
'NumberOfTime60-89DaysPastDueNotWorse',
'NumberOfDependents']

feature_names_new = ['TotalNumLateDays', 'MonthlyLeftOver']

cross_val=False
if cross_val:
	#### Cross Validation ###
	GBM(x_train,y_train,x_test, do_CV=True)
	RFC(x_train,y_train,x_test, do_CV=True)
	logistic_regression(x_train, do_CV=True)
	logistic_regression(x_train,y_train,x_test, do_CV=True)
	KNN(x_train,y_train,x_test, do_CV=True)

### Use these optimal parameters to create submission ###
new_features=True
if new_features:
	feature_names += feature_names_new

GBM(x_train,y_train,x_test, udf_trees=5000, udf_lr=0.05, udf_max_depth=5, udf_minsam=250,names=feature_names)
RFC(x_train,y_train,x_test, udf_trees=1000, udf_max_features=3, udf_min_samples=50,names=feature_names)
#logistic_regression(x_train,y_train,x_test,penalty='L2', regularization=31.6227766017)
#logistic_regression(x_train,y_train,x_test,penalty='L1', regularization=31.6227766017)
#KNN(x_train,y_train,x_test, udf_kneighbors=255)

