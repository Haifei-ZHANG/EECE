# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:46:19 2025

@author: zh106121
"""
import numpy as np

data_names = ['adult', 'banknote', 'biodeg', 'breast-cancer', 'compas', 'german', 'heart', 
              'heloc', 'liver', 'magic', 'mammographic', 'phishing', 'pima', 'spam', 'wine']
methods = ['MO', 'DisCERN', 'LIRE', 'EECE']


effectiveness_opt_L1 = np.load('./results/effectiveness_opt_L1.npy')
effectiveness_opt_L2 = np.load('./results/effectiveness_opt_L2.npy')


# time comparison

print('Time comparison')
avg_time = (effectiveness_opt_L1[-1] + effectiveness_opt_L2[-1])/2
avg_time = avg_time.round(3)

latex_string = ''
for d in range(len(data_names)):
    
    latex_string += str.capitalize(data_names[d]) + ' & '
    
    for m in range(len(methods)):
        latex_string += "{:.3f}".format(avg_time[d,m]) + ' & '
        
    latex_string = latex_string[:-2]
    latex_string += '\\\\ \n'
    
print(latex_string)



# plausibility comparison

print('Plausibility comparison')
plausibility = (effectiveness_opt_L1[-2] + effectiveness_opt_L2[-2])/2
plausibility = plausibility.round(3) * 100

latex_string = ''
for d in range(len(data_names)):
    
    latex_string += str.capitalize(data_names[d]) + ' & '
    
    for m in range(len(methods)):
        if plausibility[d,m] == 100:
            latex_string += "{}".format(int(plausibility[d,m])) + ' & '
        else:
            latex_string += "{:.1f}".format(plausibility[d,m]) + ' & '
        
    latex_string = latex_string[:-2]
    latex_string += '\\\\ \n'
    
print(latex_string)



# L2 comparison

print('L2 distance comparison')

l2_distance = effectiveness_opt_L2[1].round(3)
ratios = (l2_distance/l2_distance[:,-1].reshape((-1,1))).round(3)

latex_string = ''
for d in range(len(data_names)):
    
    latex_string += str.capitalize(data_names[d]) + ' & '
    
    for m in range(len(methods)):
        latex_string += "{:.3f}({:.3f}) & ".format(l2_distance[d,m], ratios[d,m])
        
    latex_string = latex_string[:-2]
    latex_string += '\\\\ \n'
    
print(latex_string)



# L1 comparison

print('L1 distance comparison')

l1_distance = effectiveness_opt_L1[0].round(3)
ratios = (l1_distance/l1_distance[:,-1].reshape((-1,1))).round(3)

latex_string = ''
for d in range(len(data_names)):
    
    latex_string += str.capitalize(data_names[d]) + ' & '
    
    for m in range(len(methods)):
        latex_string += "{:.3f}({:.3f}) & ".format(l1_distance[d,m], ratios[d,m])
        
    latex_string = latex_string[:-2]
    latex_string += '\\\\ \n'
    
print(latex_string)



# L0 comparison

print('L0 distance comparison')

l0_opt_l2 = effectiveness_opt_L2[2].round(3)
l0_opt_l1 = effectiveness_opt_L1[2].round(3)

latex_string = ''
for d in range(len(data_names)):
    
    latex_string += str.capitalize(data_names[d]) + ' & '
    
    for m in range(len(methods)):
        latex_string += "{:.3f} & ".format(l0_opt_l2[d,m])
        
    for m in range(len(methods)):
        latex_string += "{:.3f} & ".format(l0_opt_l1[d,m])
        
    latex_string = latex_string[:-2]
    
    latex_string += '\\\\ \n'
print(latex_string)
