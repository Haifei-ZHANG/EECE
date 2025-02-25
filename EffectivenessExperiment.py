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
    data_names = ['adult', 'banknote', 'biodeg', 'breast-cancer', 'compas', 'german', 'heart', 
                  'heloc', 'liver', 'magic', 'mammographic', 'phishing', 'pima', 'spam', 'wine']
    
    
    methods = ['mo', 'discern', 'lire', 'eece']
    metrics = ['l2', 'l1', 'l0', 'plausible', 'time cost']
    
    effectiveness = np.zeros((2, len(metrics),len(data_names), len(methods)))
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
        X = np.array(data.iloc[:,:-1])
        y = np.array(data.iloc[:,-1]) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=20)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

    
        explainer = CFGenerators(clf, X_train, feature_names, feature_cons)
        explainer.fit()
        
        for dist_type in ['L1', 'L2']:
            print('distance type: ', dist_type)
            dt_index = 0 if dist_type=='L1' else 1
            n_test = len(y_test)
            for i in tqdm(range(n_test)):
                target = 1 - predictions[i]
                x = X_test[i]
                for m in range(len(methods)):
                    effectiveness[dt_index,:,d,m] += list(explainer.generate_cf(x, target, methods[m], dist_type).values())[6:11]
                
            effectiveness[dt_index,:,d,:] = (effectiveness[dt_index,:,d,:]/n_test).round(4)
            print(effectiveness[dt_index,:,d,:])
    
    np.save('./results/effectiveness_opt_L1.npy', effectiveness[0])
    np.save('./results/effectiveness_opt_L2.npy', effectiveness[1])

