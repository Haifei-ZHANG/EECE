# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import copy
import scipy.stats
import time


# def search_path(estimator, class_labels, aim_label):
#     """
#     return path index list containing [{leaf node id, inequality symbol, threshold, feature index}].
#     estimator: decision tree
#     maxj: the number of selected leaf nodes
#     """
#     """ select leaf nodes whose outcome is aim_label """
#     children_left = estimator.tree_.children_left  # information of left child node
#     children_right = estimator.tree_.children_right
#     feature = estimator.tree_.feature
#     threshold = estimator.tree_.threshold
#     # leaf nodes ID
#     leaf_nodes = np.where(children_left == -1)[0]
#     # outcomes of leaf nodes
#     leaf_values = estimator.tree_.value[leaf_nodes].reshape(len(leaf_nodes), len(class_labels))
#     # select the leaf nodes whose outcome is aim_label
#     leaf_nodes = np.where(leaf_values[:, aim_label] != 0)[0]
#     """ search the path to the selected leaf node """
#     paths = {}
#     for leaf_node in leaf_nodes:
#         """ correspond leaf node to left and right parents """
#         child_node = leaf_node
#         parent_node = -100  # initialize
#         parents_left = [] 
#         parents_right = [] 
#         while (parent_node != 0):
#             if (np.where(children_left == child_node)[0].shape == (0, )):
#                 parent_left = -1
#                 parent_right = np.where(
#                     children_right == child_node)[0][0]
#                 parent_node = parent_right
#             elif (np.where(children_right == child_node)[0].shape == (0, )):
#                 parent_right = -1
#                 parent_left = np.where(children_left == child_node)[0][0]
#                 parent_node = parent_left
#             parents_left.append(parent_left)
#             parents_right.append(parent_right)
#             """ for next step """
#             child_node = parent_node
#         # nodes dictionary containing left parents and right parents
#         paths[leaf_node] = (parents_left, parents_right)
        
#     path_info = {}
#     for i in paths:
#         node_ids = []  # node ids used in the current node
#         # inequality symbols used in the current node
#         inequality_symbols = []
#         thresholds = []  # thretholds used in the current node
#         features = []  # features used in the current node
#         parents_left, parents_right = paths[i]
#         for idx in range(len(parents_left)):
#             if (parents_left[idx] != -1):
#                 """ the child node is the left child of the parent """
#                 node_id = parents_left[idx]  # node id
#                 node_ids.append(node_id)
#                 inequality_symbols.append(0)
#                 thresholds.append(threshold[node_id])
#                 features.append(feature[node_id])
#             elif (parents_right[idx] != -1):
#                 """ the child node is the right child of the parent """
#                 node_id = parents_right[idx]
#                 node_ids.append(node_id)
#                 inequality_symbols.append(1)
#                 thresholds.append(threshold[node_id])
#                 features.append(feature[node_id])
#             path_info[i] = {'node_id': node_ids,
#                             'inequality_symbol': inequality_symbols,
#                             'threshold': thresholds,
#                             'feature': features}
#     return path_info


def search_path(estimator, class_labels, aim_label):
    """
    Find paths to leaf nodes in a decision tree where the predicted label is aim_label.

    Parameters:
    - estimator: A decision tree classifier.
    - class_labels: List of all class labels.
    - aim_label: The target class label.

    Returns:
    - path_info: Dictionary with paths for selected leaf nodes.
    """

    # 获取决策树的子节点信息
    children_left = estimator.tree_.children_left  # 左子节点索引
    children_right = estimator.tree_.children_right  # 右子节点索引
    feature = estimator.tree_.feature  # 进行划分的特征索引
    threshold = estimator.tree_.threshold  # 阈值

    # 确保 aim_label 是有效的索引
    if aim_label >= len(class_labels) or aim_label < 0:
        print(f"Error: aim_label {aim_label} is out of range.")
        return {}

    # 识别叶子节点（没有子节点的节点）
    leaf_nodes = np.where(children_left == -1)[0]

    # 获取叶子节点的分类结果
    leaf_values = estimator.tree_.value[leaf_nodes].reshape(len(leaf_nodes), len(class_labels))

    # 选取与 aim_label 相关的叶子节点
    leaf_nodes = leaf_nodes[leaf_values[:, aim_label] != 0]

    # 如果没有符合的叶子节点，返回空
    if len(leaf_nodes) == 0:
        return {}

    paths = {}  # 存储所有叶子节点的路径信息

    # 遍历每个目标叶子节点，寻找其路径
    for leaf_node in leaf_nodes:
        parents_left, parents_right = [], []
        child_node = leaf_node
        parent_node = None

        # 反向寻找父节点，直到达到根节点（index 0）
        while parent_node != 0:
            left_parents = np.where(children_left == child_node)[0]
            right_parents = np.where(children_right == child_node)[0]

            if left_parents.size > 0:
                parent_node = left_parents[0]
                parents_left.append(parent_node)
                parents_right.append(-1)  # 不是右孩子
            elif right_parents.size > 0:
                parent_node = right_parents[0]
                parents_right.append(parent_node)
                parents_left.append(-1)  # 不是左孩子
            else:
                break  # 无法找到父节点，跳出循环

            child_node = parent_node  # 更新子节点，继续向上遍历

        paths[leaf_node] = (parents_left, parents_right)

    # 构建最终路径信息
    path_info = {}
    for leaf_id, (parents_left, parents_right) in paths.items():
        node_ids, inequality_symbols, thresholds_list, features_list = [], [], [], []

        for idx in range(len(parents_left)):
            if parents_left[idx] != -1:
                node_id = parents_left[idx]
                inequality_symbols.append(0)  # 0 代表左孩子，条件是 <= threshold
            elif parents_right[idx] != -1:
                node_id = parents_right[idx]
                inequality_symbols.append(1)  # 1 代表右孩子，条件是 > threshold
            else:
                continue  # 如果 parent=-1，则跳过

            node_ids.append(node_id)
            thresholds_list.append(threshold[node_id])
            features_list.append(feature[node_id])

        if node_ids:  # 只有在非空时才存入
            path_info[leaf_id] = {
                'node_id': node_ids,
                'inequality_symbol': inequality_symbols,
                'threshold': thresholds_list,
                'feature': features_list
            }

    return path_info


def esatisfactory_instance(x, epsilon, path_info):
    """
    return the epsilon satisfactory instance of x.
    """
    esatisfactory = copy.deepcopy(x)
    for i in range(len(path_info['feature'])):
        # feature index
        feature_idx = path_info['feature'][i]
        # threshold used in the current node
        threshold_value = path_info['threshold'][i]
        # inequality symbol
        inequality_symbol = path_info['inequality_symbol'][i]
        if inequality_symbol == 0:
            esatisfactory[feature_idx] = threshold_value - epsilon
        elif inequality_symbol == 1:
            esatisfactory[feature_idx] = threshold_value + epsilon
        else:
            print('something wrong')
    return esatisfactory
 
def feature_tweaking(ensemble_classifier, x, class_labels, aim_label, epsilon, cost_func):
    """
    This function return the active feature tweaking vector.
    x: feature vector
    class_labels: list containing the all class labels
    aim_label: the label which we want to transform the label of x to
    """
    """ initialize """
    start_time = time.time()
    x_out = copy.deepcopy(x)  # initialize output
    delta_mini = 10**3  # initialize cost
    # for estimator in ensemble_classifier:
    #     if (ensemble_classifier.predict(x.reshape(1, -1)) == estimator.predict(x.reshape(1, -1))
    #         and estimator.predict(x.reshape(1, -1) != aim_label)):
    #         paths_info = search_path(estimator, class_labels, aim_label)
    for estimator in ensemble_classifier.estimators_:  # 遍历每棵树
        ensemble_pred = ensemble_classifier.predict(x.reshape(1, -1))[0]
        estimator_pred = estimator.predict(x.reshape(1, -1))[0]
        # print(ensemble_pred, estimator_pred, ensemble_classifier.n_estimators)
        if (ensemble_pred == estimator_pred and estimator_pred != aim_label):
            paths_info = search_path(estimator, class_labels, aim_label)

            for key in paths_info:
                """ generate epsilon-satisfactory instance """
                path_info = paths_info[key]
                es_instance = esatisfactory_instance(x, epsilon, path_info)
                if estimator.predict(es_instance.reshape(1, -1)) == aim_label:
                    if cost_func(x, es_instance) < delta_mini:
                        x_out = es_instance
                        delta_mini = cost_func(x, es_instance)
            else:
                continue
    end_time = time.time()
    return x_out, round(end_time-start_time, 4)


