# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:39:53 2024

@author: zhanghai
"""

import numpy as np
import pandas as pd
# import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from CFGenerators import CFGenerators


if __name__ == "__main__":
    data = pd.read_csv('pima.csv')
    feature_names = list(data.columns)[:-1]
    feature_cons = {'immutable':None,
                    'increasing':None,
                    # 'increasing':['Pregnancies', 'Age'],
                    'decreasing':None,
                    'data types':['int', 'int', 'int', 'int', 'int', 'float', 'float', 'int']}
    dist_type = 'L2'
    X = np.array(data.iloc[:,:-1])
    y = np.array(data.iloc[:,-1]) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    
    explainer = CFGenerators(clf, X_train, feature_names, feature_cons, dist_type)
    explainer.fit()
    
    generator_list = ['mo', 'discern', 'lire', 'eece']
    
    for i in range(len(y_test)):
        target = 1 - predictions[i]
        for generator in generator_list:
            print(explainer.generate_cf(X_test[i], target, generator, 'L2'))
            
        if i==3:
            break
