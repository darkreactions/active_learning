# This module includes ML models, active learning models, and 
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:23:15 2020

@author: Zhi Li
"""

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
import xgboost as xgb 
from modAL.batch import uncertainty_batch_sampling
from modAL.utils.combination import make_linear_combination, make_product
from modAL.uncertainty import classifier_uncertainty, classifier_margin, classifier_entropy, entropy_sampling
from modAL.utils.selection import multi_argmax
import time
from tqdm import tqdm

def PearsonVII_kernel(X1,X2, sigma=1.0, omega=1.0):
    if X1 is X2 :
        kernel = squareform(pdist(X1, 'euclidean'))
    else:
        kernel = cdist(X1, X2, 'euclidean')

    kernel = (1 + (kernel * 4 * np.sqrt(2**(1.0/omega)-1)) / sigma**2) ** omega
    kernel = 1/kernel

    return kernel

def custom_query_strategy(classifier, X, n_instances=1):
    query_idx = multi_argmax(classifier_uncertainty(classifier, X), n_instances=n_instances)
    return query_idx, X[query_idx]

def random_sampling(classifier, X, n_instances=1):
    n_samples = len(X)
    query_idx = np.random.choice(range(n_samples), size = n_instances)
    return query_idx, X[query_idx]

def actlearn_perf (learner, X, y, X_training, y_training, X_test, y_test, n_queries, n_instances):
    train_size = [X_training.shape[0]] 
    train_pool_size = train_size[0]
    X_training_pool = X_training.copy()
    y_training_pool = y_training.copy()
    X_test_pool = X_test.copy()
    y_test_pool = y_test.copy()
    scores = [learner.score(X,y)]
    for idx in tqdm(range(n_queries)):
        query_idx, query_instances = learner.query(X_test_pool, n_instances=n_instances)
        learner.teach(X_test_pool[query_idx], y_test_pool[query_idx])
        X_training_pool = np.vstack((X_training_pool, X_test_pool[query_idx]))
        y_training_pool = np.hstack((y_training_pool, y_test_pool[query_idx]))
        X_test_pool = np.delete(X_test_pool, query_idx, axis=0)
        y_test_pool = np.delete(y_test_pool, query_idx)
        scores.append(learner.score(X,y))
        train_pool_size += n_instances
        train_size.append(train_pool_size)
        time.sleep(0)
    return train_size, scores

estimator = {'SVC_rbf': SVC(C=1,gamma=1,cache_size=6000,max_iter=-1,kernel='rbf', \
                        decision_function_shape='ovr', probability=True, \
                        class_weight='balanced', random_state=42),\
             'SVC_Pearson': SVC(C=1,cache_size=6000,max_iter=-1,kernel=PearsonVII_kernel, \
                        decision_function_shape='ovr', probability=True, \
                        class_weight='balanced', random_state=42),\
             'RF': RandomForestClassifier(criterion='entropy', n_estimators = 100, max_depth = 9,\
                                         min_samples_leaf = 1),\
             'xgboost': xgb.XGBClassifier(base_score=0.5, booster='gbtree', n_estimators=100),\
             'kNN': KNeighborsClassifier(n_neighbors = 1, weights = "uniform", p = 2),\
             'GPC': GaussianProcessClassifier(1.0*RBF(1.0))}