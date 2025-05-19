
import numpy as np
import time
import copy
from multiprocessing import Pool
from sklearn.neighbors import LocalOutlierFactor
import multiprocessing
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
        self.dist_type = dist_type
        self.epsilon = 0.001
        self.lof = LocalOutlierFactor(novelty=True)

    def _get_top_3(self, candidates, distances):
        if len(distances) == 0:
            return None
        top_indices = np.argsort(distances)[:3]
        return candidates[top_indices]
    def _feature_change_mask(self, x, cf):
        return tuple((cf != x).astype(int))

    def compute_diversity(self, x, cf_list):
        if cf_list is None:
            return 0
        change_masks = [tuple((cf != x).astype(int)) for cf in cf_list]
        unique_masks = set(change_masks)
        return len(unique_masks)
        
    def compute_jaccard_diversity(self, x, cf_list):
        if cf_list is None or len(cf_list) < 2:
            return 0.0
        masks = [(cf != x).astype(int) for cf in cf_list]
        n = len(masks)
        total = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                a = masks[i]
                b = masks[j]
                intersection = np.sum(np.logical_and(a, b))
                union = np.sum(np.logical_or(a, b))
                if union > 0:
                    jaccard_dist = 1 - intersection / union
                    total += jaccard_dist
                    count += 1
        return total / count if count > 0 else 0.0

        if cf_list is None:
            return 0
        change_masks = [tuple((cf != x).astype(int)) for cf in cf_list]
        unique_masks = set(change_masks)
        return len(unique_masks)

    @staticmethod
    def _process_tree(tree, n_features):
        decision_paths_dict_part = {}
        decision_paths_part = []
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        dp_stack = [[[-1e8, 1e8] for _ in range(n_features)]]

        for i in range(n_nodes):
            if children_left[i] != children_right[i]:
                parent_dp = dp_stack.pop()
                left_dp = copy.deepcopy(parent_dp)
                right_dp = copy.deepcopy(parent_dp)
                dp_stack.append(right_dp)
                dp_stack.append(left_dp)
                dp_stack[-1][feature[i]][1] = threshold[i]
                dp_stack[-2][feature[i]][0] = threshold[i]
            else:
                dp_to_add = np.array(dp_stack.pop()).flatten()
                decision_paths_dict_part[i] = dp_to_add
                decision_paths_part.append(dp_to_add)

        return decision_paths_dict_part, decision_paths_part

    @staticmethod
    def _process_leaf(args):
        i, leaves_index, decision_paths_dict, n_trees, n_features = args
        current_live_region = np.zeros((n_trees, 2 * n_features))
        for t in range(n_trees):
            current_live_region[t] = decision_paths_dict[t][leaves_index[i, t]]

        live_region = np.zeros(2 * n_features)
        live_region[0::2] = current_live_region[:, 0::2].max(axis=0)
        live_region[1::2] = current_live_region[:, 1::2].min(axis=0)
        return live_region

    def fit(self):
        feature_importance = self.rf.feature_importances_
        self.feature_order = np.argsort(feature_importance)
        self.dist_std = np.std(self.train_set, axis=0)
        medians_abs_diff = abs(self.train_set - np.median(self.train_set, axis=0))
        self.dist_mad = np.mean(medians_abs_diff, axis=0)
        self.dist_mad[self.dist_mad == 0] = 1
        self.lof.fit(self.train_set)

        with Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(self._process_tree,
                                   [(tree, self.n_features) for tree in self.rf.estimators_])

        self.decision_paths_dict = {}
        decision_paths = np.zeros(2 * self.n_features)
        for t, (dict_part, paths_part) in enumerate(results):
            self.decision_paths_dict[t] = dict_part
            if paths_part:
                decision_paths = np.vstack((decision_paths, np.array(paths_part)))
        self.decision_paths = decision_paths[1:, :]

        leaves_index = self.rf.apply(self.train_set)
        leaves_index, remain_index = np.unique(leaves_index, axis=0, return_index=True)
        self.live_regions_predictions = self.rf.predict(self.train_set)[remain_index]
        self.leaves_index = leaves_index

        with Pool(processes=min(4, multiprocessing.cpu_count())) as pool:
            args = [(i, leaves_index, self.decision_paths_dict, self.n_trees, self.n_features)
                    for i in range(len(leaves_index))]
            self.live_regions = np.array(pool.map(self._process_leaf, args))

    def __instance_dist(self, x, X, dist_type):
        if dist_type == 'L1':
            dists = (abs(X - x)/self.dist_mad).sum(axis=1)
        elif dist_type == 'L0':
            dists = (X != x).sum(axis=1)
        else:
            dists = np.sqrt((((X - x)/self.dist_std)**2).sum(axis=1))
        return dists

    def mo(self, x, target, dist_type=None):
        predictions = self.rf.predict(self.train_set)
        remain_instances = self.train_set[predictions == target].copy()

        for d in range(self.n_features):
            feature = self.feature_names[d]
            if self.immutable_features is not None and feature in self.immutable_features:
                remain_instances = remain_instances[x[d] == remain_instances[:, d]]
            elif self.increasing_features is not None and feature in self.increasing_features:
                remain_instances = remain_instances[x[d] <= remain_instances[:, d]]
            elif self.decreasing_features is not None and feature in self.decreasing_features:
                remain_instances = remain_instances[x[d] >= remain_instances[:, d]]

        if len(remain_instances) == 0:
            return None

        dists = self.__instance_dist(x, remain_instances, dist_type or self.dist_type)
        return self._get_top_3(remain_instances, dists)

    def lire(self, x, target, dist_type=None):
        live_regions = self.live_regions[self.live_regions_predictions == target].copy()
        for d in range(self.n_features):
            feature = self.feature_names[d]
            if self.immutable_features is not None and feature in self.immutable_features:
                live_regions = live_regions[(live_regions[:,2*d] < x[d]) & (x[d] <= live_regions[:,2*d+1])]
            elif self.increasing_features is not None and feature in self.increasing_features:
                live_regions = live_regions[live_regions[:,2*d+1] > x[d]]
            elif self.decreasing_features is not None and feature in self.decreasing_features:
                live_regions = live_regions[live_regions[:,2*d] < x[d]]

        if len(live_regions) == 0:
            return None

        candidates = self.__generate_cf_in_regions(x, live_regions)
        predictions = self.rf.predict(candidates)
        candidates = candidates[predictions == target]

        if len(candidates) == 0:
            return None

        dists = self.__instance_dist(x, candidates, dist_type or self.dist_type)
        return self._get_top_3(candidates, dists)

    def discern(self, x, target, dist_type=None):
        init_cfs = self.mo(x, target, dist_type)
        if init_cfs is None:
            return None

        results = []
        for init_cf in init_cfs:
            cf = x.copy()
            for d in self.feature_order:
                cf[d] = init_cf[d]
                prediction = self.rf.predict(cf.reshape((1, -1)))[0]
                if prediction == target:
                    results.append(cf.copy())
                    break

        if len(results) == 0:
            return None

        results = np.array(results)
        dists = self.__instance_dist(x, results, dist_type or self.dist_type)
        return self._get_top_3(results, dists)

    def __generate_cf_in_regions(self, x, regions):
        candidates = x.reshape((1, -1)).repeat(len(regions), axis=0)
        for d in range(self.n_features):
            take_inf = regions[:, 2*d] >= x[d]
            take_sup = regions[:, 2*d+1] < x[d]
            if self.feature_types[d] == 'int':
                inf_values = np.ceil(regions[take_inf, 2*d] + self.epsilon)
                sup_values = np.floor(regions[take_sup, 2*d+1])
            else:
                inf_values = regions[take_inf, 2*d] + self.epsilon
                sup_values = regions[take_sup, 2*d+1]
            if len(inf_values) > 0:
                candidates[take_inf, d] = inf_values
            if len(sup_values) > 0:
                candidates[take_sup, d] = sup_values
        return candidates

    def eece(self, x, target, dist_type=None):
        regions = np.vstack([self.decision_paths,
                             self.live_regions[self.live_regions_predictions == target]])
        regions = self._vectorized_region_filter(regions, x)

        if len(regions) == 0:
            return None

        candidates = self.__generate_cf_in_regions(x, regions)
        predictions = self.rf.predict(candidates)
        candidates = candidates[predictions == target]

        if len(candidates) == 0:
            return None

        plausibility = self.lof.predict(candidates)
        candidates = candidates[plausibility == 1]

        if len(candidates) == 0:
            return None

        dists = self.__instance_dist(x, candidates, dist_type or self.dist_type)
        return self._get_top_3(candidates, dists)

    def _vectorized_region_filter(self, regions, x):
        for d, name in enumerate(self.feature_names):
            if name in self.immutable_features:
                regions = regions[(regions[:, 2*d] < x[d]) & (x[d] <= regions[:, 2*d+1])]
            if name in self.increasing_features:
                regions = regions[regions[:, 2*d+1] > x[d]]
            if name in self.decreasing_features:
                regions = regions[regions[:, 2*d] < x[d]]
        return regions


    def eece_new(self, x, target, dist_type=None):
        init_cfs = self.lire(x, target, dist_type)
        if init_cfs is None:
            return None

        regions = self.decision_paths.copy()
        candidates = self.__generate_cf_in_regions(x, regions)

        dists = self.__instance_dist(x, candidates, dist_type or self.dist_type)
        index = dists < self.__instance_dist(x, init_cfs[0:1], dist_type or self.dist_type)[0]
        candidates = candidates[index]
        dists = dists[index]

        if len(candidates) == 0:
            return init_cfs

        predictions = self.rf.predict(candidates)
        candidates = candidates[predictions == target]
        dists = dists[predictions == target]

        if len(candidates) == 0:
            return init_cfs

        plausibility = self.lof.predict(candidates)
        candidates = candidates[plausibility == 1]
        dists = dists[plausibility == 1]

        if len(candidates) == 0:
            return init_cfs

        return self._get_top_3(candidates, dists)

    def eece_pparallelized(self, x, target, dist_type=None):
        init_cfs = self.lire(x, target, dist_type)
        if init_cfs is None:
            return None

        candidates = self.__generate_cf_in_regions(x, self.decision_paths)
        dists = self.__instance_dist(x, candidates, dist_type or self.dist_type)

        dists_lire = self.__instance_dist(x, init_cfs[0:1], dist_type or self.dist_type)[0]
        index = dists < dists_lire
        candidates = candidates[index]
        dists = dists[index]

        if len(candidates) == 0:
            return init_cfs

        n_jobs = multiprocessing.cpu_count()

        if hasattr(self.rf, 'n_jobs'):
            old_n_jobs = self.rf.n_jobs
            self.rf.n_jobs = n_jobs
            predictions = self.rf.predict(candidates)
            self.rf.n_jobs = old_n_jobs
        else:
            if len(candidates) > 1000:
                batch_size = max(1, len(candidates) // n_jobs)
                batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]
                batch_results = Parallel(n_jobs=n_jobs)(delayed(self.rf.predict)(b) for b in batches)
                predictions = np.concatenate(batch_results)
            else:
                predictions = self.rf.predict(candidates)

        index = predictions == target
        candidates = candidates[index]
        dists = dists[index]

        if len(candidates) == 0:
            return init_cfs

        if hasattr(self.lof, 'n_jobs'):
            old_n_jobs = self.lof.n_jobs
            self.lof.n_jobs = n_jobs
            plausibility = self.lof.predict(candidates)
            self.lof.n_jobs = old_n_jobs
        else:
            if len(candidates) > 500:
                batch_size = max(1, len(candidates) // n_jobs)
                batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]
                batch_results = Parallel(n_jobs=n_jobs)(delayed(self.lof.predict)(b) for b in batches)
                plausibility = np.concatenate(batch_results)
            else:
                plausibility = self.lof.predict(candidates)

        index = plausibility == 1
        candidates = candidates[index]
        dists = dists[index]

        if len(candidates) == 0:
            return init_cfs

        sorted_indices = np.argsort(dists)
        diverse_cfs = []
        used_masks = set()

        fallback_pool = []
        for idx in sorted_indices:
            cf = candidates[idx]
            mask = self._feature_change_mask(x, cf)
            if mask not in used_masks:
                diverse_cfs.append(cf)
                used_masks.add(mask)
            else:
                fallback_pool.append(cf)
            if len(diverse_cfs) == 3:
                break

        while len(diverse_cfs) < 3 and fallback_pool:
            diverse_cfs.append(fallback_pool.pop(0))

        return np.array(diverse_cfs[:3]) if len(diverse_cfs) > 0 else init_cfs

    def generate_cf(self, x, target, generator='eece', dist_type=None):
        y_hat = self.rf.predict(x.reshape((1, -1)))[0]
        if target not in self.classes or target == y_hat:
            return None, 0.0

        if generator == 'mo':
            cfs = self.mo(x, target, dist_type)
        elif generator == 'discern':
            cfs = self.discern(x, target, dist_type)
        elif generator == 'lire':
            cfs = self.lire(x, target, dist_type)
        elif generator == 'eece':
            cfs = self.eece(x, target, dist_type)
        elif generator == 'eece_new':
            cfs = self.eece_new(x, target, dist_type)
        elif generator == 'eece_pparallelized':
            cfs = self.eece_pparallelized(x, target, dist_type)
        else:
            raise NotImplementedError(f"Generator {generator} not supported for top-3 mode.")

        if cfs is not None:
            diversity = self.compute_diversity(x, cfs)
            jaccard_div = self.compute_jaccard_diversity(x, cfs)
            print(f"Diversity of explanations: {diversity}")
            print(f"Jaccard diversity: {round(jaccard_div, 4)}")
        else:
            jaccard_div = 0.0
        return cfs, jaccard_div

