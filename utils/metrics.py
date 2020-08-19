from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold

from utils.delong import auc_ci

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def cutoff_youdens_j(y_true, y_pred_proba):
    '''
    Find probability cutoff threshold using Youden's J statistic
    '''

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba, pos_label=1)
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores, thresholds))
    return j_ordered[-1][1]

def get_performance(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = (tp + tn) / (tn + fp + fn + tp)
    spec = tn / (fp + tn)
    sens = tp / (tp + fn)

    return acc, sens, spec

def get_train_cv_performance_youdens(model, train_df, kf, opt):
    cv = kf.n_splits

    # X,y split
    train_X = train_df.drop(opt.label_col, axis=1).values
    train_y = train_df[opt.label_col].values
    
    # Result container
    auc_result = 0
    acc_result = 0
    sens_result = 0
    spec_result = 0

    # Calculate performance
    auc_result_list = []
    for train_index, valid_index in kf.split(train_X, train_y):
        # Split k-fold dataset
        train_X_kf, X_valid_kf = train_X[train_index], train_X[valid_index]
        train_y_kf, y_valid_kf = train_y[train_index], train_y[valid_index]

        # Train on k-fold split dataset with random search hyper-parameters
        model_kf = model.estimator
        model_kf.probability = True
        model_kf.fit(train_X_kf, train_y_kf)

        # Get threshold cutoff using Youden's J index
        train_y_kf_pred_proba = model_kf.predict_proba(train_X_kf)[:, 1]
        cutoff = cutoff_youdens_j(train_y_kf, train_y_kf_pred_proba)
        
        # Predict k-fold validation data
        y_valid_kf_pred_proba = model_kf.predict_proba(X_valid_kf)[:, 1]
        y_pred_kf = (y_valid_kf_pred_proba > cutoff).astype(int)

        # Calculate performance k-fold validation data
        auc = auc_ci(y_valid_kf, y_valid_kf_pred_proba, return_ci=False)
        acc, sens, spec = get_performance(y_valid_kf, y_pred_kf)
        
        # Update k-fold validation result
        auc_result += auc
        acc_result += acc
        sens_result += sens
        spec_result += spec
        auc_result_list.append(auc)
    
    # Train CV Performance Dictionary
    train_cv_result = dict()

    auc_mean = np.mean(np.array(auc_result_list))
    auc_std = np.std(np.array(auc_result_list))
    margin = 1.96 * auc_std / cv ** (1/2)
        
    train_cv_result['AUC'] = (auc_result / cv)
    train_cv_result['Accuracy'] = (acc_result / cv)
    train_cv_result['Sensitivity'] = (sens_result / cv)
    train_cv_result['Specificity'] = (spec_result / cv)
    train_cv_result['AUC_95CI_LOWER'] = (auc_mean - margin)
    train_cv_result['AUC_95CI_UPPER'] = (auc_mean + margin)
    train_cv_result['RSD'] = (auc_std / auc_mean * 100.)

    return train_cv_result