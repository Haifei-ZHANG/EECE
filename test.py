# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:39:53 2024

@author: zhanghai
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from CFGenerators import CFGenerators
from FeatureTweaking import feature_tweaking


def cost_func(a, b):
    return np.linalg.norm(a-b)

if __name__ == "__main__":
    data = pd.read_csv('data/pima.csv')
    feature_names = list(data.columns)[:-1]
    feature_cons = {'immutable':None,
                    'increasing':None,
                    'decreasing':None,
                    'data types':['float']*len(feature_names)}
    dist_type = 'L2'
    X = np.array(data.iloc[:,:-1])
    y = np.array(data.iloc[:,-1]) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = RandomForestClassifier(n_estimators=100, max_depth=20)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    
    explainer = CFGenerators(clf, X_train, feature_names, feature_cons, dist_type)
    explainer.fit()
    
    generator_list = ['mo', 'discern', 'lire', 'eece']
    
    epsilon = 0.05
    class_labels = [0, 1]
    tc_eece = 0
    tc_ft = 0
    for i in tqdm(range(len(y_test))):
        
        prediction = predictions[i]
        target = 1 - prediction
        x = X_test[i]
        tc_info = explainer.generate_cf(x, target, 'eece', 'L2')
        print(tc_info)
        
        
        cf, tc = feature_tweaking(clf, x, class_labels , target, epsilon, cost_func)
        print(cf)
        
        if i==5: break