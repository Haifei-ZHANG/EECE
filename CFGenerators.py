# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:25:32 2024

@author: zhanghai
"""

import numpy as np
import time
import copy
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor

class CFGenerators:
    def __init__(self, rf, train_set, feature_names, feature_cons=None, dist_type='L1'):
        self.rf = rf
        self.n_trees = rf.n_estimators
        self.classes = rf.classes_
        self.n_classes = len(self.classes)
        self.train_set = train_set
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.immutable_features = feature_cons['immutable']
        self.increasing_features = feature_cons['increasing']
        self.decreasing_features = feature_cons['decreasing']
        self.feature_types = feature_cons['data types']
        self.dist_type = dist_type
        self.epsilon = 0.001
        self.lof = LocalOutlierFactor(n_neighbors=5, novelty=True)
        
    
    def fit(self):
        # get feature importance to determin feature order
        feature_importance = self.rf.feature_importances_
        self.feature_order = np.argsort(feature_importance)
        
        # calculate the distance adjustment terms
        self.dist_std = np.std(self.train_set, axis=0)
        medians_abs_diff = abs(self.train_set - np.median(self.train_set, axis=0))
        self.dist_mad = np.mean(medians_abs_diff, axis=0)
        self.dist_mad[self.dist_mad==0] = 1
        
        # train the local outlier factor
        self.lof.fit(self.train_set)
        
        # extract decision paths
        decision_paths = np.zeros(2 * self.n_features)
        decision_paths_dict = {}
        t = -1
        print('Building candidate regions')
        for tree in tqdm(self.rf.estimators_):
            t += 1
            decision_paths_dict[t] = {}
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
            
            dp_stack = [[[-1e8, 1e8] for i in range(self.n_features)]]
    
            for i in range(n_nodes):
                is_internal_node = (children_left[i] != children_right[i])
                if is_internal_node:
                    parent_dp = dp_stack.pop()
                    left_dp = copy.deepcopy(parent_dp)
                    right_dp = copy.deepcopy(parent_dp)
                    dp_stack.append(right_dp)
                    dp_stack.append(left_dp)
                    dp_stack[-1][feature[i]][1] = threshold[i]
                    dp_stack[-2][feature[i]][0] = threshold[i]
                else:
                    dp_to_add = np.array(dp_stack.pop()).flatten()
                    decision_paths_dict[t][i] = dp_to_add
                    decision_paths = np.vstack((decision_paths, dp_to_add))
            
        self.decision_paths = decision_paths[1:,:]
        
        leaves_index = self.rf.apply(self.train_set)
        leaves_index, remain_index = np.unique(leaves_index, axis=0, return_index=True)
        self.live_regions_predictions = (self.rf.predict(self.train_set))[remain_index]
        self.leaves_index = leaves_index
        self.live_regions = np.zeros((len(leaves_index), 2 * self.n_features))
        # construct live regions
        for i in range(len(leaves_index)):
            current_live_region = np.zeros((self.n_trees, 2 * self.n_features))
            for t in range(self.n_trees):
                current_live_region[t] = decision_paths_dict[t][leaves_index[i,t]]
    
            self.live_regions[i,0:-1:2] = current_live_region[:,0:-1:2].max(axis=0)
            self.live_regions[i,1::2] = current_live_region[:,1::2].min(axis=0)
    
        # self.epsilon = ((self.live_regions[:,1::2]-self.live_regions[:,0:-1:2])/2).min()
        # print(self.epsilon)
     
        
    def __instance_dist(self, x, X, dist_type):
        if dist_type == 'L1':
            dists = (abs(X - x)/self.dist_mad).sum(axis=1)
        elif dist_type == 'L0':
            dists = (X != x).sum(axis=1)
        else:
            dists = np.sqrt((((X - x)/self.dist_std)**2).sum(axis=1))
            
        return dists
    
    
    def __interval_dist(self, d, x_d, intervals, dist_type):
        in_index = (intervals[:,0] < x_d) * (x_d <= intervals[:,1])
        left_index = intervals[:,1] < x_d
        right_index = intervals[:,0] >= x_d
        dists = np.zeros(len(intervals))
        dists[in_index] = 0
        if self.feature_types[d] == 'int':
            dists[left_index] = x_d - np.floor(intervals[left_index,1])
            dists[right_index] = np.ceil(intervals[right_index,1] + self.epsilon) - x_d
        else:
            dists[left_index] = x_d - intervals[left_index,1]
            dists[right_index] = intervals[right_index,1] - x_d + self.epsilon
        if dist_type == 'L1':
            dists = dists/self.dist_mad[d]
        elif dist_type == 'L0':
            dists = (dists!=0)*1
        else:
            dists = (dists/self.dist_std[d])**2
        
        return dists
    
    
    def __generate_cf_in_regions(self, x, regions):
        candidates = x.reshape((1,-1)).repeat(len(regions), axis=0)
        for d in range(self.n_features):
            take_inf = regions[:,2*d] >= x[d]
            take_sup = regions[:,2*d+1] < x[d]
            if self.feature_types[d] == 'int':
                inf_values = np.ceil(regions[take_inf,2*d] + self.epsilon)
                sup_values = np.floor(regions[take_sup,2*d+1])
            else:
                inf_values = regions[take_inf,2*d] + self.epsilon
                sup_values = regions[take_sup,2*d+1]
            if len(inf_values) > 0:
                candidates[take_inf,d] = inf_values
            if len(sup_values) > 0:
                candidates[take_sup,d] = sup_values
                
        # check_in_region = (candidates> regions[:,0:-1:2]) * (candidates <= regions[:,0::2])
        # if check_in_region.sum() > 0 :
        #     print('ALERT: not all in regions!')
        return candidates
    
    
    def mo(self, x, target, dist_type=None):
        predictions = self.rf.predict(self.train_set)
        remain_instances = self.train_set[predictions==target].copy()
        
        for d in range(self.n_features):
            feature = self.feature_names[d]
            
            if self.immutable_features is not None:
                if feature in self.immutable_features:
                    index = (x[d] == remain_instances[:,d])
                    remain_instances = remain_instances[index]
                    continue
            elif self.increasing_features is not None:
                if feature in self.increasing_features:
                    index = (x[d] <= remain_instances[:,d])
                    remain_instances = remain_instances[index]
                    continue
            elif self.decreasing_features is not None:
                if feature in self.decreasing_features:
                    index = (x[d] >= remain_instances[:,d])
                    remain_instances = remain_instances[index]
                    continue
            else:
                continue
            
        if len(remain_instances) == 0:
            print("Your feature constrains are too strict for this instance! Can't generate satisfied counterfactual example!")
            return None, None
        dists = self.__instance_dist(x, remain_instances, dist_type)
        cf_index = np.argmin(dists)
        min_dist = dists[cf_index]
        cf = remain_instances[cf_index]
        
        return cf, round(min_dist,4)
    
    
    def discern(self, x, target, dist_type=None):
        init_cf, init_min_dist = self.mo(x, target, dist_type)
        cf = x.copy()
        if init_cf is None:
            return init_cf, init_min_dist
        else:
            for d in self.feature_order:
                cf[d] = init_cf[d]
                prediction = self.rf.predict(cf.reshape((1,-1)))[0]
                if prediction ==  target:
                    min_dist = self.__instance_dist(x, cf.reshape((1,-1)), dist_type)[0]
                    return cf, round(min_dist, 4)
            
    
    def lire(self, x, target, dist_type=None):
        live_regions = self.live_regions[self.live_regions_predictions==target].copy()
        for d in range(self.n_features):
            feature = self.feature_names[d]
            
            if self.immutable_features is not None:
                if feature in self.immutable_features:
                    index = (live_regions[:,2*d] < x[d]) * (x[d] <= live_regions[:,2*d+1])
                    live_regions = live_regions[index]
                    continue
            elif self.increasing_features is not None:
                if feature in self.increasing_features:
                    index = live_regions[:,2*d+1] > x[d]
                    live_regions = live_regions[index]
                    continue
            elif self.decreasing_features is not None:
                if feature in self.decreasing_features:
                    index = live_regions[:,2*d] < x[d]
                    live_regions = live_regions[index]
                    continue
            else:
                continue
        
        if len(live_regions)==0:
            print("Your feature constrains are too strict for this instance! Can't generate satisfied counterfactual example!")
            return None, None
        else:
            candidates = self.__generate_cf_in_regions(x, live_regions)
            predictions = self.rf.predict(candidates)
            candidates = candidates[predictions==target]
            dists = self.__instance_dist(x, candidates, dist_type)
            cf_index = np.argmin(dists)
            min_dist = dists[cf_index]
            cf = candidates[cf_index]
        
            return cf, round(min_dist,4)
        
    
    
    def eece(self, x, target, dist_type=None):     
        regions = np.concatenate((self.decision_paths.copy(), self.live_regions[self.live_regions_predictions==target].copy()), axis=0)
        for d in range(self.n_features):
            feature = self.feature_names[d]
            
            if self.immutable_features is not None:
                if feature in self.immutable_features:
                    index = (regions[:,2*d] < x[d]) * (x[d] <= regions[:,2*d+1])
                    regions = regions[index]
                    continue
            elif self.increasing_features is not None:
                if feature in self.increasing_features:
                    index = regions[:,2*d+1] > x[d]
                    regions = regions[index]
                    continue
            elif self.decreasing_features is not None:
                if feature in self.decreasing_features:
                    index = regions[:,2*d] < x[d]
                    regions = regions[index]
                    continue
            else:
                continue
        
        if len(regions)==0:
            print("1 Your feature constrains are too strict for this instance! Can't generate satisfied counterfactual example!")
            return None, None
        else:
            candidates = self.__generate_cf_in_regions(x, regions)
            predictions = self.rf.predict(candidates)
            candidates = candidates[predictions==target]
            if len(candidates)==0:
                print("2 Can not generate counterfactual example!")
                return None, None
            else:
                plausibility = self.lof.predict(candidates)
                if sum(plausibility)!=0:
                    candidates = candidates[plausibility==1]
                dists = self.__instance_dist(x, candidates, dist_type)
                cf_index = np.argmin(dists)
                min_dist = dists[cf_index]
                cf = candidates[cf_index]
                
                return cf, round(min_dist,4)
            
            
    def generate_cf(self, x, target, generator='eece', dist_type=None):
        y_hat = self.rf.predict(x.reshape((1,-1)))[0]
        if target not in self.classes or target==y_hat:
            print("Your input target dose not existe!")
            cf = None
            min_dist = None
        else:
            start_time = time.time()
            if dist_type is None and self.dist_type is None:
                dist_type = 'L1'
                
            if generator=='mo':
                cf, min_dist = self.mo(x, target, dist_type)
            elif generator=='discern':
                cf, min_dist = self.discern(x, target, dist_type)
            elif generator=='lire':
                cf, min_dist = self.lire(x, target, dist_type)
            else:
                cf, min_dist = self.eece(x, target, dist_type)
                
        end_time = time.time()  
        result = {'x': x, 'y_hat': y_hat, 'cf': cf, 'target': target, 'valid': None,
                  'dist_type': dist_type, 'L1': None, 'L2': None, 'L0': None,
                  'plausible': None, 'time cost':None}
        if cf is None:
            return result
        else:
            cf_y_hat = self.rf.predict(cf.reshape((1,-1)))[0]
            if target == cf_y_hat:
                result['valid'] = True
            else:
                result['valid'] = False
            
            result['L1'] = round(self.__instance_dist(x, cf.reshape((1,-1)), 'L1')[0], 5)
            result['L2'] = round(self.__instance_dist(x, cf.reshape((1,-1)), 'L2')[0], 5)
            result['L0'] = int(self.__instance_dist(x, cf.reshape((1,-1)), 'L0')[0])
            
            if self.lof.predict(cf.reshape((1,-1)))[0] == 1:
                result['plausible'] = True
            else:
                result['plausible'] = False
                
            result['time cost'] = round(end_time-start_time, 5)
            
            return result
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        