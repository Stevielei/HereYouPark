# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 05:39:12 2023

@author: steve
"""
from collections import Counter
import scipy.stats
import numpy as np
import pywt
import pandas as pd
import time
from sklearn.ensemble import GradientBoostingClassifier

import time
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from statistics import mode

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy
 
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]
 
def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]
 
def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics
    #return [entropy] + statistics

def xoverlap_windows (df, window_size):
    windows = [df[i:i+window_size] for i in range(0, len(df), window_size)]
    return windows

def DWT_analysis (windows, column, wavelet, level):
    coeffslist = []
    labels = []
    for window in windows:
        signal = window[column].values
        label = mode(window['activity'])
        waveletname = wavelet
        coeffs = pywt.wavedec(signal, waveletname, level=level)      
        coeffslist.append (coeffs)
        labels.append (label)
    return coeffslist, labels # [0]return input list [1]return output list


def inputgen (windows, columnlist, wavelet, level):
    # directly use the code below, somehow function doesn't work
    inputall = pd.DataFrame()
    for c in columnlist:
        DWT_result = list(DWT_analysis(windows, c , wavelet, level))
        allfeature = []
        for n in DWT_result[0]:
            featuren = []
            for l in n:
                featuren += get_features(l)
            allfeature.append(featuren)
            allfeature_df = pd.DataFrame(allfeature)
        inputall = pd.concat([inputall, allfeature_df],axis =1)
    inputall[len(inputall.columns)] = DWT_result[1]
    inputname = [n for n in range(len(inputall.columns))] # rename the dataframe
    inputall = inputall.set_axis(inputname,axis ='columns')
        
    trainx = inputall.loc[:,:len(inputall.columns)-2]
    trainy = inputall.loc[:,len(inputall.columns)-1:]
        
    return inputall, trainx, trainy

def Gradientboosting (X_train, Y_train, X_test, Y_test, n):
    
    cls = GradientBoostingClassifier(n_estimators=n)
    cls.fit(X_train, Y_train)
    #train_score = cls.score(X_train, Y_train)
    #test_score = cls.score(X_test, Y_test)
    resulty = cls.predict (X_test)
    
    compare = pd.DataFrame (zip (Y_test, resulty), columns = ['RealClass','ModelClass'])

    compare['correctness'] = np.where(compare['RealClass']==compare['ModelClass'],'True','False')
    print("\n")
    count = compare['correctness'].value_counts()

    print(count)
    print('\n')
    percen = round(count['True']/(count['True']+count['False']),2)*100
    print('Testset correctness:'+str(percen)+'%')

    return percen, resulty

def get_train_test(df, ratio):

    #df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(inputall, len(inputall.columns)-1, inputname[:-1], 0.7)
    #
    inputname = [n for n in range(len(df.columns))]
    y_col = len(df.columns)-1
    x_cols = inputname[:-1]
    #
    mask = np.random.rand(len(df)) < ratio
    df_train = df[mask]
    df_test = df[~mask]
       
    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    return df_train, df_test, X_train, Y_train, X_test, Y_test

# Model Experiment
def sliding_windows (df, gap, window_size):
    
    windows = [df[i:i+window_size] for i in range(0, len(df) - window_size + 1, gap)]
    num_windows = (len(df) - window_size) // gap + 1
    
    return windows, num_windows


# classifier list for comparison
dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB(),

def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers, verbose = True):
    
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.time()
        classifier.fit(X_train, Y_train)
        t_end = time.time()
        
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        
        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models

def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    
    display(df_.sort_values(by=sort_by, ascending=False))
    return df_.sort_values(by=sort_by, ascending=False)

#Sensor filter
from sklearn.cluster import KMeans
from keras.utils import to_categorical
import math as m

def filter2 (numcluster, df,):
    model = KMeans(n_clusters= numcluster)
    X= np.array(df[['X (m/s^2)','Y (m/s^2)','Z (m/s^2)']])
    model.fit(X)
    classresults = model.predict (X)
    Orientlist = ['Orient'+str(i+1) for i in range (numcluster)]
    return df[Orientlist] = pd.DataFrame(to_categorical(classresults))

def filter1 (df):
    for index, line in df.iterrows():
        Gx = float(line['X (m/s^2)']+0.00001)
        Gy = float(line['Y (m/s^2)']+0.00001)
        Gz = float(line['Z (m/s^2)']+0.00001)
        Bx = float(line['X (µT)']+0.00001)
        By = float(line['Y (µT)']+0.00001)
        Bz = float(line['Z (µT)']+0.00001)
        
        # calculate row, pitch and yaw
        row = m.degrees(m.atan(-Gx/Gz))
        #if row < 0:row += 360  
        row2= m.radians(row)
        
        pitch = m.degrees(m.atan2(Gy,((-1*Gx*m.sin(row))+(Gz*m.cos(row)))))
        #if pitch < 0:pitch += 360
        pitch2= m.radians(pitch)
        
        GxSE = -1*Gy
        GySE = -1*Gx
        GzSE = Gz
        
        BxSE = By
        BySE = Bx
        BzSE = -1*Bz
        
        rowSE = m.degrees(m.atan2(GySE,GzSE))
        pitchSE = m.degrees(m.atan(-GxSE/(GySE*m.sin(rowSE)+GzSE*m.cos(rowSE))))
        
        yawSE= m.degrees (m.atan2(BzSE * m.sin(rowSE) - BySE * m.cos(rowSE) ,
                                  BxSE * m.cos(pitchSE) + BySE * m.sin(pitchSE) * m.sin(rowSE) + BzSE * m.sin(pitchSE) * m.cos(rowSE)))
        rowlist2.append(rowSE)
        pitchlist2.append(pitchSE)
        yawlist2.append(yawSE)
        
        df['RowSE'] = rowlist2
        df['PitchSE'] = pitchlist2
        df['YawSE'] = yawlist2
        
    return df