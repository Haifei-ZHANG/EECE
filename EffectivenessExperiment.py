# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from CFGenerators import CFGenerators


def cost_func(a, b):
    return np.linalg.norm(a-b)

if __name__ == "__main__":
    data_names = ['adult', 'banknote', 'biodeg', 'breast-cancer', 'compas', 'german', 'heart', 
                  'heloc', 'liver', 'magic', 'mammographic', 'phishing', 'pima', 'spam', 'wine']
    
    # data_names = ['breast-cancer','liver','mammographic']
    methods = ['mo', 'discern', 'lire', 'eece']
    metrics = ['l1', 'l2', 'l0', 'plausible', 'time cost']
    
    K = 10
    effectiveness = np.zeros((2, K, len(metrics),len(data_names), len(methods)))
    avg = np.zeros((2, len(metrics),len(data_names), len(methods)))
    std = np.zeros((2, len(metrics),len(data_names), len(methods)))
    
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
        y = np.array(data.iloc[:,-1]).astype(int)
        
        kf = KFold(n_splits=K, shuffle=True)
        
        k = -1
        for train_index, test_index in kf.split(X):
            k += 1
            print(f"Fold {k+1}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
            clf = RandomForestClassifier(n_estimators=100, max_depth=20)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
    
        
            explainer = CFGenerators(clf, X_train, feature_names, feature_cons)
            explainer.fit()
            
            for dist_type in ['L1', 'L2']:
                print('distance type: ', dist_type)
                dt_index = 0 if dist_type=='L1' else 1
                if data_name in ['adult', 'heloc']:
                    n_test = 300
                else:
                    n_test = len(y_test)
                for i in tqdm(range(n_test)):
                    target = 1 - predictions[i]
                    x = X_test[i]
                    for m in range(len(methods)):
                        effectiveness[dt_index, k, :,d,m] += list(explainer.generate_cf(x, target, methods[m], dist_type).values())[6:11]
                    
                effectiveness[dt_index,k,:,d,:] = (effectiveness[dt_index,k,:,d,:]/n_test).round(4)
        avg[0,:,d,:] = effectiveness[0,:,:,d,:].mean(axis=0)
        avg[1,:,d,:] = effectiveness[1,:,:,d,:].mean(axis=0)
        std[0,:,d,:] = effectiveness[0,:,:,d,:].std(axis=0)
        std[1,:,d,:] = effectiveness[1,:,:,d,:].std(axis=0)
        print('Optimizing L1:')
        print(avg[0,:,d,:])
        print(std[0,:,d,:])
        print('Optimizing L2:')
        print(avg[1,:,d,:])
        print(std[1,:,d,:])
       
    np.save('./results/effectiveness_opt_L1.npy', effectiveness[0])
    np.save('./results/effectiveness_opt_L2.npy', effectiveness[1])
    np.save('./results/effectiveness_opt_L1_avg.npy', avg[0])
    np.save('./results/effectiveness_opt_L2_avg.npy', avg[1])
    np.save('./results/effectiveness_opt_L1_std.npy', std[0])
    np.save('./results/effectiveness_opt_L2_std.npy', std[1])

