# -*- coding: utf-8 -*-
"""
Created on Tue May  6 09:24:58 2025

@author: zh106121
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from CFGenerators import CFGenerators


def cost_func(a, b):
    return np.linalg.norm(a-b)

if __name__ == "__main__":
    data_names = ['adult', 'compas', 'heloc',  'phishing', 'spam']
    
    
    n_trees = [100, 200, 300, 400, 500]
    efficience_wrt_ntrees = np.zeros((len(data_names), len(n_trees)))
    for d in range(len(data_names)):
        data_name = data_names[d]
        print(data_name)
        data = pd.read_csv('data/{}.csv'.format(data_name))
        data = data.astype(float, errors='ignore')
        feature_names = list(data.columns)[:-1]
        feature_cons = {'immutable':None,
                        'increasing':None,
                        'decreasing':None,
                        'data types':['float']*len(feature_names)}
        dist_type = 'L2'
        X = np.array(data.iloc[:,:-1])
        y = np.array(data.iloc[:,-1]).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
        
        for t in range(len(n_trees)):
            print('n_trees=',n_trees[t])
            clf = RandomForestClassifier(n_estimators=n_trees[t], max_depth=None)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)

    
            explainer = CFGenerators(clf, X_train, feature_names, feature_cons, dist_type)
            explainer.fit()
    
    
            epsilon = 0.001
            class_labels = [0, 1]
            tc_eece = 0
            tc_ft = 0
            n_test = len(y_test)
            for i in tqdm(range(end_index)):
                target = 1 - predictions[i]
                x = X_test[i]
                tc = explainer.generate_cf(x, target, 'eece', 'L1')['time cost']
                if tc is not None:
                    tc_eece += tc
                else:
                    n_test -= 1
                
            efficience_wrt_ntrees[d,t] = round(tc_eece/n_test, 4)
            print(efficience_wrt_ntrees[d,t])
    	np.save('./results/efficience_wrt_ntrees_large_dataset.npy', efficience_wrt_ntrees)
    
    np.save('./results/efficience_wrt_ntrees_large_dataset.npy', efficience_wrt_ntrees)
    print(efficience_wrt_ntrees)
    
