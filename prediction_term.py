import sys
import scipy
import numpy as np
import matplotlib
import pandas as pd 
import sklearn
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from statistics import mode,StatisticsError

model_men = joblib.load('men.joblib')
model_np = joblib.load('noPregnant.joblib')
model_wp = joblib.load('pregnant.joblib')

def calBMI(w,h):
    w = float(w)
    h = float(h)/100
    return w/(h**h)

def question():
    global age,bmi,bp,glucose,skin,insulin,weight,height
    age = input('How old are you? : ')
    weight = input('How much of your weight? (kg.): ')
    height = input('How tall are you? (cm.): ')
    bp = input('How much of your blood pressure? : ')
    glucose = input('How much of the level of glucose do you have? : ')
    skin = input('How much of skin thickness do you have? : ')
    insulin = input('How much of the level of insulin do you have? :')
    bmi = calBMI(weight,height)
    return age,bmi,bp,glucose,skin,insulin
    

def preProcess(details) :
    age,bmi,bp,glucose,skin,insulin = details
    if (age == '?'):
        age = 38

    if(bmi == '?'):
        bmi = 30

    if(bp == '?'):
        bp = 78

    if(glucose == '?'):
        glucose = 120

    if(skin == '?'):
        skin = 21

    if(insulin == '?'):
        insulin=80

    return age,bmi,bp,glucose,skin,insulin

def calculate(details,sex,preg =0) :
    age,bmi,bp,glucose,skin,insulin = details
    if (sex=='m') : 
        info = [[int(age),int(bmi),int(bp)]]
        result = model_men.predict(info)
        print('outcome ',result)
        final_result = result

    elif (sex=='w') :
        if(preg=='?'):
            preg = 0
            info = [[preg,int(glucose),int(bp),int(skin),int(insulin),int(bmi),int(age)]]
            pred_1 = model_np.predict(info)
            pred_2 = model_wp.predict(info)
            print(pred_1)
            print(pred_2)
            try:
                outcome = mode([pred_1[0],pred_2[0]])
                print('Final ==== ',outcome)
                final_result = int(pred_2)
            except StatisticsError as e:
                #final_result = int(pred_2) 
                print('This is the case exception : ',e)
        
        elif (int(preg) == 0): 
            info = [[int(preg),int(glucose),int(bp),int(skin),int(insulin),int(bmi),int(age)]]
            result = model_np.predict(info)
            print('outcome ',result)
            final_result = result
        
        elif(int(preg)>0):
            info = [[int(preg),int(glucose),int(bp),int(skin),int(insulin),int(bmi),int(age)]]
            result = model_wp.predict(info)
            print('outcome ',result)
            final_result = result
    else :
        if (preg == '?'): preg = 0

        info_m = [[int(age),int(bmi),int(bp)]]
        info_w = [[int(preg),int(glucose),int(bp),int(skin),int(insulin),int(bmi),int(age)]]

        pred_1 = model_men.predict(info_m)
        pred_2 = model_np.predict(info_w)
        pred_3 = model_wp.predict(info_w)
        print('result_m',pred_1)
        print('result_np',pred_2)
        print('result_wp',pred_3)

        result = mode([pred_1[0],pred_2[0],pred_3[0]])
        print('result ==== ',result)
        final_result = result

    final_result = int(final_result)
    if(final_result == 1) :
        print('You tend to have Diabetes problem, Please contact your doctors')
       
    else: 
        print('Congratulation, you still have low chance for diabetes issue')
     

def main(): 
    global sex
    print('----------------------- Welcome to TU-Diagnosis system -----------------------')
    print("Description : Answer following questions, if you don't know, please input '?'")
    sex = input('What is your sex? (m / w / ?) : ')
    if (sex == 'm'):
       calculate(preProcess(question()),sex)

    elif(sex == 'w' or sex =='?'):
        global preg
        preg = input('How many times have you ever been pregnant before? : ')
        calculate(preProcess(question()),sex,preg)
    else : 
        print('Please re-insert correct information')

#-------------------------------------------------------------------------------------

if __name__== "__main__":
  main()








