# -*- coding: utf-8 -*-
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
    data_names = ['adult', 'banknote', 'biodeg', 'breast-cancer', 'compas', 'german', 'heart', 
                  'heloc', 'liver', 'magic', 'mammographic', 'phishing', 'pima', 'spam', 'wine']
    
    
    n_trees = [20, 40, 60, 80, 100]
    efficience_wrt_ntrees = np.zeros((len(data_names), 2, len(n_trees)))
    # efficience_wrt_ntrees = np.load('./results/efficience_wrt_ntrees.npy')
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
            clf = RandomForestClassifier(n_estimators=n_trees[t], max_depth=20)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)

    
            explainer = CFGenerators(clf, X_train, feature_names, feature_cons, dist_type)
            explainer.fit()
    
    
            epsilon = 0.001
            class_labels = [0, 1]
            tc_eece = 0
            tc_ft = 0
            n_test = len(y_test)
            for i in tqdm(range(n_test)):
                target = 1 - predictions[i]
                x = X_test[i]
                tc1 = explainer.generate_cf(x, target, 'eece', 'L2')['time cost']
                tc_eece += tc1
                cf, tc2 = feature_tweaking(clf, x, class_labels , target, epsilon, cost_func)
                tc_ft += tc2
                
            efficience_wrt_ntrees[d,0,t] = round(tc_eece/n_test, 4)
            efficience_wrt_ntrees[d,1,t] = round(tc_ft/n_test, 4)
            print(efficience_wrt_ntrees[d,:,t])
            
        np.save('./results/efficience_wrt_ntrees.npy', efficience_wrt_ntrees)
    
    np.save('./results/efficience_wrt_ntrees.npy', efficience_wrt_ntrees)
    
    
    
    n_depths = [5, 10, 15, 20]
    efficience_wrt_depths = np.zeros((len(data_names), 2, len(n_depths)))
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
        
        for t in range(len(n_depths)):
            print('max depth=',n_depths[t])
            clf = RandomForestClassifier(n_estimators=100, max_depth=n_depths[t])
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)

    
            explainer = CFGenerators(clf, X_train, feature_names, feature_cons, dist_type)
            explainer.fit()
    
    
            epsilon = 0.001
            class_labels = [0, 1]
            tc_eece = 0
            tc_ft = 0
            n_test = len(y_test)
            for i in tqdm(range(n_test)):
                target = 1 - predictions[i]
                x = X_test[i]
                tc1 = explainer.generate_cf(x, target, 'eece', 'L2')['time cost']
                tc_eece += tc1
                cf, tc2 = feature_tweaking(clf, x, class_labels , target, epsilon, cost_func)
                tc_ft += tc2
                
            efficience_wrt_depths[d,0,t] = round(tc_eece/n_test, 4)
            efficience_wrt_depths[d,1,t] = round(tc_ft/n_test, 4)
            print(efficience_wrt_depths[d,:,t])
            
        np.save('./results/efficience_wrt_depths.npy', efficience_wrt_depths)
    
    np.save('./results/efficience_wrt_depths.npy', efficience_wrt_depths)
    
    