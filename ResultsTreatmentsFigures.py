# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:46:19 2025

@author: zh106121
"""
import numpy as np
import matplotlib.pyplot as plt
# figures

data_names = ['adult', 'banknote', 'biodeg', 'breast-cancer', 'compas', 'german', 'heart', 
              'heloc', 'liver', 'magic', 'mammographic', 'phishing', 'pima', 'spam', 'wine']

efficience_wrt_ntrees = np.load('./results/efficience_wrt_ntrees.npy')
print(efficience_wrt_ntrees.round(4))
n_trees = [20, 40, 60, 80, 100]
line_width = 3.0
marker_size = 8
for d in range(len(data_names)):
    data_name = data_names[d]
    
    eval_eece = efficience_wrt_ntrees[d][0].round(3)
    eval_ft = efficience_wrt_ntrees[d][1].round(3)
    
    plt.figure(figsize=(8, 8))
    plt.plot(n_trees, eval_eece, marker='o', linestyle='-', label='EECE', color='b', linewidth=line_width, markersize=marker_size)
    plt.plot(n_trees, eval_ft, marker='s', linestyle='--', label='Feature Tweaking', color='r', linewidth=line_width, markersize=marker_size)
    
    offset_y = 0.01 * max(eval_ft)
    for i, (x, y) in enumerate(zip(n_trees, eval_eece)):
        if data_name == 'breast-cancer' and i == 0:
            offset_y = -0.02 * max(eval_ft)
            va='top'
        else:
            offset_y = 0.01 * max(eval_ft)
            va='bottom'
        plt.text(x, y + offset_y, f"{y:.3f}", ha='center', va=va, fontsize=14, color='b')

    for i, (x, y) in enumerate(zip(n_trees, eval_ft)):
        ha = 'left' if i < 2 else 'right'
        va = 'top' if i < 2 else 'bottom'
        offset_x = 1 if i < 2 else -1
        plt.text(x + offset_x, y, f"{y:.3f}", ha=ha, va=va, fontsize=14, color='r')
    
    plt.xticks(n_trees)
    plt.xlabel('Number of trees in random forest', fontsize=14)
    plt.ylabel('Time per example (seconds)', fontsize=14)
    # plt.title('Comparison of Explanation Generation Time')
    plt.legend(fontsize=14)
    plt.grid(True)
    
    plt.savefig("./results/figures/{}-efficience-wrt-ntrees.png".format(data_name), dpi=400, bbox_inches='tight')
    plt.show()
    
    
efficience_wrt_depths = np.load('./results/efficience_wrt_depths.npy')
print(efficience_wrt_depths.round(4))
n_depths = [5, 10, 15, 20]
for d in range(len(data_names)):
    data_name = data_names[d]
    
    eval_eece = efficience_wrt_depths[d][0].round(3)
    eval_ft = efficience_wrt_depths[d][1].round(3)
    
    plt.figure(figsize=(8, 8))
    plt.plot(n_depths, eval_eece, marker='o', linestyle='-', label='EECE', color='b', linewidth=line_width, markersize=marker_size)
    plt.plot(n_depths, eval_ft, marker='s', linestyle='--', label='Feature Tweaking', color='r', linewidth=line_width, markersize=marker_size)
    
    offset_y = 0.02 * max(eval_ft)
    for i, (x, y) in enumerate(zip(n_depths, eval_eece)):
        plt.text(x, y - offset_y, f"{y:.3f}", ha='center', va='top', fontsize=14, color='b')

    for i, (x, y) in enumerate(zip(n_depths, eval_ft)):
        ha = 'center'
        va = 'top'
        offset_y = - 0.03 * max(eval_ft)
        if i==1:
            offset_x = 0.3
        else:
            offset_x = 0
        if data_name in ['adult', 'heloc', 'phishing', 'compas', 'spam']:
            if i == 0:
                ha = 'center'
                va = 'bottom'
                offset_y = 0.02 * max(eval_ft)
            else:
                ha = 'right'
                va = 'bottom'
                offset_y = 0
                offset_x = -0.25
        plt.text(x + offset_x, y + offset_y, f"{y:.3f}", ha=ha, va=va, fontsize=14, color='r')
    
    plt.xticks(n_depths)
    plt.xlabel('Depth of trees in random forest', fontsize=14)
    plt.ylabel('Time per example (seconds)', fontsize=14)
    # plt.title('Comparison of Explanation Generation Time')
    plt.legend(fontsize=14)
    plt.grid(True)
    
    plt.savefig("./results/figures/{}-efficience-wrt-depths.png".format(data_name), dpi=400, bbox_inches='tight')
    plt.show()