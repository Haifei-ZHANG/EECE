# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:25:32 2024

@author: zhanghai
"""

import numpy as np
import time
import copy
from sklearn.neighbors import LocalOutlierFactor
import multiprocessing
from multiprocessing import Pool
from joblib import Parallel, delayed


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
        self.epsilon = 0.001
        self.lof = LocalOutlierFactor(novelty=True)
        # print(self.feature_names, self.immutable_features , self.increasing_features,self.decreasing_features)
        

    @staticmethod
    def _process_tree(tree, n_features):
        """Parallel processing of individual trees"""
        decision_paths_dict_part = {}
        decision_paths_part = []
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        
        dp_stack = [[[-1e8, 1e8] for _ in range(n_features)]]

        for i in range(n_nodes):
            if children_left[i] != children_right[i]:  # Internal node
                parent_dp = dp_stack.pop()
                left_dp = copy.deepcopy(parent_dp)
                right_dp = copy.deepcopy(parent_dp)
                dp_stack.append(right_dp)
                dp_stack.append(left_dp)
                dp_stack[-1][feature[i]][1] = threshold[i]
                dp_stack[-2][feature[i]][0] = threshold[i]
            else:  # Leaf node
                dp_to_add = np.array(dp_stack.pop()).flatten()
                decision_paths_dict_part[i] = dp_to_add
                decision_paths_part.append(dp_to_add)
        
        return decision_paths_dict_part, decision_paths_part

    @staticmethod
    def _process_leaf(args):
        """Parallel processing of leaf regions"""
        i, leaves_index, decision_paths_dict, n_trees, n_features = args
        current_live_region = np.zeros((n_trees, 2 * n_features))
        for t in range(n_trees):
            current_live_region[t] = decision_paths_dict[t][leaves_index[i, t]]
        
        live_region = np.zeros(2 * n_features)
        live_region[0::2] = current_live_region[:, 0::2].max(axis=0)
        live_region[1::2] = current_live_region[:, 1::2].min(axis=0)
        return live_region

    def fit(self):
        # Get feature importance
        feature_importance = self.rf.feature_importances_
        self.feature_order = np.argsort(feature_importance)
        
        # Distance adjustments
        self.dist_std = np.std(self.train_set, axis=0)
        medians_abs_diff = abs(self.train_set - np.median(self.train_set, axis=0))
        self.dist_mad = np.mean(medians_abs_diff, axis=0)
        self.dist_mad[self.dist_mad == 0] = 1
        
        # Train LOF
        self.lof.fit(self.train_set)
        
        # Parallel tree processing
        print('Building candidate regions with parallel processing')
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(self._process_tree, 
                                 [(tree, self.n_features) for tree in self.rf.estimators_])
        
        # Aggregate tree results
        self.decision_paths_dict = {}
        decision_paths = np.zeros(2 * self.n_features)
        for t, (dict_part, paths_part) in enumerate(results):
            self.decision_paths_dict[t] = dict_part
            if paths_part:
                decision_paths = np.vstack((decision_paths, np.array(paths_part)))
        self.decision_paths = decision_paths[1:,:]

        # Process leaves in parallel
        leaves_index = self.rf.apply(self.train_set)
        leaves_index, remain_index = np.unique(leaves_index, axis=0, return_index=True)
        self.live_regions_predictions = self.rf.predict(self.train_set)[remain_index]
        self.leaves_index = leaves_index
        
        print('Building live regions with parallel processing')
        with Pool(processes=min(4, multiprocessing.cpu_count())) as pool:
            args = [(i, leaves_index, self.decision_paths_dict, 
                    self.n_trees, self.n_features) for i in range(len(leaves_index))]
            self.live_regions = np.array(pool.map(self._process_leaf, args))
            
    
        
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
            if self.increasing_features is not None:
                if feature in self.increasing_features:
                    index = (x[d] <= remain_instances[:,d])
                    remain_instances = remain_instances[index]
                    continue
            if self.decreasing_features is not None:
                if feature in self.decreasing_features:
                    index = (x[d] >= remain_instances[:,d])
                    remain_instances = remain_instances[index]
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
            if self.increasing_features is not None:
                if feature in self.increasing_features:
                    index = live_regions[:,2*d+1] > x[d]
                    live_regions = live_regions[index]
                    continue
            if self.decreasing_features is not None:
                if feature in self.decreasing_features:
                    index = live_regions[:,2*d] < x[d]
                    live_regions = live_regions[index]
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
            
            if len(candidates) == 0:
                return cf, min_dist
            cf_index = np.argmin(dists)
            min_dist = dists[cf_index]
            cf = candidates[cf_index]
            
            return cf, round(min_dist, 4)
        
    
    def eece(self, x, target, dist_type=None):
        cf, min_dist = self.lire(x, target, dist_type)
        
        candidates = self.__generate_cf_in_regions(x, self.decision_paths)
        dists = self.__instance_dist(x, candidates, dist_type)

        
        index = dists < min_dist
        candidates = candidates[index]
        dists = dists[index]
        
        if len(candidates) == 0:
            return cf, min_dist
        
        # Optimize random forest predictions - this is the computationally intensive part
        # Many random forest implementations already support internal parallelism
        n_jobs = multiprocessing.cpu_count()
        
        # If random forests support parallelism, use directly
        if hasattr(self.rf, 'n_jobs'):
            old_n_jobs = self.rf.n_jobs
            self.rf.n_jobs = n_jobs
            predictions = self.rf.predict(candidates)
            self.rf.n_jobs = old_n_jobs
        else:
            # Otherwise manually parallelized high-volume forecasts
            batch_size = max(1, len(candidates) // n_jobs)
            
            def predict_batch(batch):
                return self.rf.predict(batch)
                
            # Parallel processing only when sample size is large enough
            if len(candidates) > 1000:  # can be adjusted
                batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]
                batch_results = Parallel(n_jobs=n_jobs)(
                    delayed(predict_batch)(batch) for batch in batches
                )
                predictions = np.concatenate(batch_results)
            else:
                predictions = self.rf.predict(candidates)
        
        index = predictions == target
        candidates = candidates[index]
        dists = dists[index]
        
        if len(candidates) == 0:
            return cf, min_dist
        
        # Optimize LOF calculations - often also computationally intensive
        if hasattr(self.lof, 'n_jobs'):
            # Use built-in parallelism
            old_n_jobs = self.lof.n_jobs
            self.lof.n_jobs = n_jobs
            plausibility = self.lof.predict(candidates)
            self.lof.n_jobs = old_n_jobs
        else:
            # Manual batch parallel processing
            # Parallelize only when sample size is large enough
            if len(candidates) > 500:  # can be adjusted
                batch_size = max(1, len(candidates) // n_jobs)
                batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]
                
                def lof_predict_batch(batch):
                    return self.lof.predict(batch)
                    
                batch_results = Parallel(n_jobs=n_jobs)(
                    delayed(lof_predict_batch)(batch) for batch in batches
                )
                plausibility = np.concatenate(batch_results)
            else:
                plausibility = self.lof.predict(candidates)
        
        index = (plausibility == 1)
        candidates = candidates[index]
        dists = dists[index]
        
        if len(candidates) == 0:
            return cf, min_dist
        cf_index = np.argmin(dists)
        min_dist = dists[cf_index]
        cf = candidates[cf_index]
        
        return cf, round(min_dist, 4)

        
    
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
            