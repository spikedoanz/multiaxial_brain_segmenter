#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:46:44 2024

@author: deeperthought
"""



import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize


df = pd.read_csv('/home/deeperthought/Projects/Multiaxial/Data/dice_scores.csv')

df = pd.read_csv('/home/deeperthought/Projects/Multiaxial/Data/anonymous_dice_scores.csv')

'/home/deeperthought/Projects/Multiaxial/Data/dice_scores_avg.csv'

models = df['segmentation'].unique()


df['avg_dice'] = np.mean(df[['dice_score0','dice_score1','dice_score2','dice_score3','dice_score4','dice_score5','dice_score6']], axis=1)

# df['avg_brain_dice'] = np.mean(df[['dice_score2','dice_score3']], axis=1)
df['avg_brain_dice'] = np.mean(df[['dice_score3']], axis=1)

# df.to_csv('/home/deeperthought/Projects/Multiaxial/Data/dice_scores_avg.csv', index=False)


# df = df.loc[df['subject'] != 'GU036']

#%% Average dice


# import seaborn as sns

# for model in models:
    
#     print(model)
#     print(np.median(df.loc[df['segmentation'] == model, 'avg_dice']))

    
# sag = df.loc[df['segmentation'] == 'sagittal',['subject','avg_dice']]
# sag.columns = ['subject', 'sagittal']
# ax = df.loc[df['segmentation'] == 'axial',['subject','avg_dice']]
# ax.columns = ['subject', 'axial']
# cor = df.loc[df['segmentation'] == 'coronal',['subject','avg_dice']]
# cor.columns = ['subject', 'coronal']
# mx = df.loc[df['segmentation'] == 'layer',['subject','avg_dice']]
# mx.columns = ['subject', 'layer']
# mp = df.loc[df['segmentation'] == 'multipriors',['subject','avg_dice']]
# mp.columns = ['subject', 'multipriors']

# df_all = pd.merge(sag,cor, on='subject')
# df_all = pd.merge(df_all,ax, on='subject')
# df_all = pd.merge(df_all,mx, on='subject')
# df_all = pd.merge(df_all,mp, on='subject')





# plt.plot([np.random.normal(i,0.01,len(df_all)) for i in range(5)], [df_all['sagittal'], df_all['coronal'], df_all['axial'], df_all['layer'], df_all['multipriors']],
#          color='gray', alpha=0.35)
# sns.violinplot(data=df_all)#, x='subject', y='sagittal', hue=True, hue_order=[False, True], split=True)

# plt.xticks([0,1,2,3,4],['Sagittal','Coronal','Axial','MultiAxial','MultiPriors'])
# plt.ylabel('Dice score')

# from scipy.stats import ranksums, wilcoxon

# wilcoxon(mx['layer'].values-mp['multipriors'].values)

# wilcoxon(mx, sag)
# wilcoxon(mx, cor)
# wilcoxon(mx, ax)

#%% Average brain dice


import seaborn as sns

for model in models:
    
    print(model)
    print(np.median(df.loc[df['segmentation'] == model, 'avg_brain_dice']))

    
layer = df.loc[df['segmentation'] == 'layer',['subject','avg_brain_dice']]
layer.columns = ['subject', 'layer']

deface = df.loc[df['segmentation'] == 'deface',['subject','avg_brain_dice']]
deface.columns = ['subject', 'deface']

reface = df.loc[df['segmentation'] == 'reface',['subject','avg_brain_dice']]
reface.columns = ['subject', 'reface']

reface_plus = df.loc[df['segmentation'] == 'reface_plus',['subject','avg_brain_dice']]
reface_plus.columns = ['subject', 'reface_plus']



df_all = pd.merge(layer,deface, on='subject')
df_all = pd.merge(df_all,reface, on='subject')
df_all = pd.merge(df_all,reface_plus, on='subject')


df_all = df_all.sort_values('reface_plus')

df_all = df_all[3:]




plt.plot([np.random.normal(i,0.01,len(df_all)) for i in range(4)], [df_all['layer'], df_all['deface'], df_all['reface'], df_all['reface_plus']],
         color='gray', alpha=0.35)
sns.violinplot(data=df_all)#, x='subject', y='sagittal', hue=True, hue_order=[False, True], split=True)

plt.ylabel('Dice score')

from scipy.stats import ranksums, wilcoxon, friedmanchisquare

friedmanchisquare(df_all['layer'], df_all['deface'], df_all['reface'], df_all['reface_plus'])

multiplecomparisons = 6

print(wilcoxon(df_all['layer'].values , df_all['deface'].values)[1]/multiplecomparisons)
print(wilcoxon(df_all['layer'].values , df_all['reface'].values)[1]/multiplecomparisons)
print(wilcoxon(df_all['layer'].values , df_all['reface_plus'].values)[1]/multiplecomparisons)
print(wilcoxon(df_all['deface'].values , df_all['reface'].values)[1]/multiplecomparisons)
print(wilcoxon(df_all['deface'].values , df_all['reface_plus'].values)[1]/multiplecomparisons)
print(wilcoxon(df_all['reface'].values , df_all['reface_plus'].values)[1]/multiplecomparisons)





#%% dice per tissue

CLASSES = [f'dice_score{i}' for i in range(7)]

TISSUES = ['Background','Air','WhiteMatter','GrayMatter','CSF','Bone','Skin']

plt.figure(figsize=(10,20))
i = 1
for tissue in CLASSES:
        
    sag = df.loc[df['segmentation'] == 'sagittal', tissue].values
    ax = df.loc[df['segmentation'] == 'axial', tissue].values
    cor = df.loc[df['segmentation'] == 'coronal', tissue].values
    mx = df.loc[df['segmentation'] == 'layer', tissue].values
    mp = df.loc[df['segmentation'] == 'multipriors', tissue].values
    
    plt.subplot(4,2,i); plt.title(f'{TISSUES[i-1]}')
    plt.plot([np.random.normal(i,0.01,len(mx)) for i in range(5)], [sag, cor, ax, mx, mp], color='lightblue', alpha=0.8)
    plt.plot([np.random.normal(i,0.01,len(mx)) for i in range(5)], [sag, cor, ax, mx, mp],'.', color='dodgerblue', alpha=0.8)
    
    plt.xticks([0,1,2,3,4],['Sagittal','Coronal','Axial','MultiAxial','MultiPriors'])
    
    print(f'{TISSUES[i-1]}')
    print(np.median(mx), np.median(mp))
    if np.median(mx) > np.median(mp): print('MultiAxial is better')
    else: print('MultiPriors is better')
    print(ranksums(mx, mp))
    
    i += 1
    
# ranksums(mx, sag)
# ranksums(mx, cor)
# ranksums(mx, ax)



#%% Post hoc evaluation


index = np.argmin(mx - mp)

df.loc[df['segmentation'] == 'layer', 'avg_dice'].iloc[index]
df.loc[df['segmentation'] == 'multipriors', 'avg_dice'].iloc[index]

df.loc[df['segmentation'] == 'multipriors'].iloc[index]




