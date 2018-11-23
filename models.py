# -*- coding: utf-8 -*-
import sys
import scipy
import numpy as np
import matplotlib
import pandas 
import sklearn
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.externals import joblib

woman_info = 'diabetes.csv'
woman_set = pd.read_csv(woman_info)

#---------------------------Woman with no preg------------------------------------------
preg0 = woman_set[woman_set.Pregnancies==0]
array = preg0.values
preg0_X = preg0[['Pregnancies',	'Glucose',	'BloodPressure',	'SkinThickness',	'Insulin',	'BMI',	'Age']]
preg0_Y = array[:,8]
test_size=0.20
seed=7
preg0_X_train, preg0_X_test, preg0_Y_train, preg0_Y_test = model_selection.train_test_split(preg0_X, preg0_Y, test_size= test_size, random_state=seed)

preg0_m = GaussianNB()
preg0_m.fit(preg0_X_train, preg0_Y_train)
predictions = preg0_m.predict(preg0_X_test)
joblib.dump(preg0_m, 'noPregnant.joblib')

#------------------------Woman had been pregnant---------------------------------------

pregnants = woman_set[woman_set.Pregnancies>=1] 
array = pregnants.values
pregnants_X = pregnants[['Pregnancies',	'Glucose',	'BloodPressure',	'SkinThickness','Insulin',	'BMI',	'Age']]
pregnants_Y = array[:,8]
test_size=0.20
seed=7
pregnants_X_train, pregnants_X_test, pregnants_Y_train, pregnants_Y_test = model_selection.train_test_split(pregnants_X, pregnants_Y, test_size= test_size, random_state=seed)

preg1_m = GaussianNB()
preg1_m.fit(pregnants_X_train, pregnants_Y_train)
predictions = preg1_m.predict(pregnants_X_test)
joblib.dump(preg1_m, 'pregnant.joblib')

#----------------------------Man---------------------------------

men_info = 'diabetesWM.txt'
men_set = pd.read_csv(men_info,sep='\t')
men = men_set[men_set.SEX==2]
men['Outcome'] =np.where(men['Y'] >=150, 1, 0)
men_arr = men.values
men_x = men[['AGE','BMI','BP']]
men_y = men_arr[:,11]
test_size = 0.2
seed = 7
men_x_train, men_x_test, men_y_train, men_y_test = model_selection.train_test_split(men_x, men_y, test_size= test_size, random_state=seed)
men_m = GaussianNB()
men_m.fit(men_x_train, men_y_train)
predictions = men_m.predict([[59,32,101]])
print(predictions)
joblib.dump(men_m, 'men.joblib') 

#----------------------------------------------------------------------------

'''
model_men = joblib.load('men.joblib')
Xnew = [[59,32,101]]
ynew = model_men.predict(Xnew)
print('xxxxxxxxxxxxxxxxxxx = ',ynew)

model_np = joblib.load('noPregnant.joblib')
Xnew1 = [[180,	66,	39,	0,	42.0,	1.893,	25		]]
#Xnew1 = [[118,	84,	47,	230,	45.8,	0.551,	31		]]
ynew1 = model_np.predict(Xnew1)
#print('yyyyyyyyyyyyyyyyyyyyyyy = ',ynew1)

model_wp = joblib.load('pregnant.joblib')
Xnew2 = [[5,	116	,74,	0	,0,	25.6,	0.201,	30		]]
ynew2 = model_wp.predict(Xnew2)
print('zzzzzzzzzzzz = ',ynew2)
'''