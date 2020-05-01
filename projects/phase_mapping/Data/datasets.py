# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:20:14 2020

@author: Zhi Li
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification

# create 1D synthetic dataset for regression problem
X_syn1 = np.random.choice(np.linspace(0, 20, 10000), size=200, replace=False).reshape(-1, 1)
y_syn1 = (np.cos(X_syn1))**2 + np.random.normal(scale=0.3, size=X_syn1.shape)

# create 3D synthetic dataset to simulate single perovskite dataset
X_syn2, y_syn2 = make_classification(n_samples=5000, n_features=3, n_redundant=0, n_repeated=0,\
                                   n_informative=3, n_classes=3, n_clusters_per_class=1,\
                                   weights=[0.2,0.60,0.20], class_sep = 0.5, random_state=1, flip_y = 0.1)



# import experiment data
df = pd.read_csv("../Active-learning phase-mapping project/Data/perovskitedata.csv")
Inchi = pd.read_csv("../Active-learning phase-mapping project/Data/OrganicInchikey.csv")
Inchidict = dict(zip(Inchi['Chemical Name'], Inchi['InChI Key (ID)']))

# generate the list of ammonium name
Amine_done = []
for i in df['_raw_organic_0_inchikey']:
    if i not in Amine_done: 
        Amine_done.append(i)
Amine_number = len(Amine_done)
for i in range(Amine_number):
    Amine_done[i] = dict(zip(Inchi['InChI Key (ID)'],Inchi['Chemical Name']))[Amine_done[i]]


df_dict = {}
for i in Amine_done:
    inchikey = Inchidict[i]
    ammonium_index = df.index[df['_raw_organic_0_inchikey'] == inchikey].tolist()
    df_dict[i] = df.filter(ammonium_index, axis = 0)
    








