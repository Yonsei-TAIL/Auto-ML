import os
import random
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from options import parse_option
from options.search_grid import params_dict

from utils import load_data, train_valid_split
from utils.core import model_initialization, model_fit
from utils.preprocess import zscore_normalization, fill_NaN_w_mean
from utils.feature_selection import LassoSelection, FscoreSelection, MISelection
from utils.oversampling import oversample_ROSE, oversample_SMOTE

# Seed
np.random.seed(0)
random.seed(0)

# Parse argument
opt = parse_option(print_option=True)

# Load Data
print("Loading Data...")
data_df = load_data(opt)
train_df_org, valid_df_org = train_valid_split(data_df, opt)

# Pre-processing : Z-score Normalization & Remove NaN values 
print("Pre-processing...")
train_df_z, valid_df_z = zscore_normalization(train_df_org, valid_df_org, opt)
train_df_filled, valid_df_filled = fill_NaN_w_mean(train_df_z, valid_df_z)

# Feature Selection (Utilize 'Feature selection method' : 'Dataset with selected features' dictionary)
print("Feature Selection...")
dataset_dict_fs = dict()
dataset_dict_fs['LASSO'] = LassoSelection(train_df_filled, valid_df_filled, opt)
dataset_dict_fs['Fscore'] = FscoreSelection(train_df_filled, valid_df_filled, opt)
dataset_dict_fs['MI'] = MISelection(train_df_filled, valid_df_filled, opt)

# Training & Evaluation
for feature_selection_name, (train_df, valid_df) in dataset_dict_fs.items():
    # Oversampling : SMOTE & ROSE
    dataset_dict_os = {'NO' : train_df}
    dataset_dict_os['SMOTE'] = oversample_SMOTE(train_df, opt)
    dataset_dict_os['ROSE'] = oversample_ROSE(train_df, opt)

    # Result container
    result_dict = dict()

    # Main
    for sampling_name, train_df_sampled in dataset_dict_os.items():
        print("\n=====================================================")
        print(" Start : %s Oversampling [%s Feature Selection] " % (sampling_name, feature_selection_name))
        print("=====================================================\n")

        # Model
        models_dict = model_initialization(params_dict)
        for model_name, model in models_dict.items():
            params = params_dict[model_name]

            model, train_cv_result, valid_result = model_fit(model, params, train_df_sampled, valid_df, model_name, opt)
            result_dict[sampling_name] = [model_name, model, train_cv_result, valid_result]

    # plot_calibration_curve_result(no_trained_model_list, model_name_list, train_X_selected, train_y, test_X_selected, test_y, label_col, 'No', save_sub_dir)

    # # Save Results
    # plot_result(train_score_result, 'Train', save_dir=save_sub_dir, model_name_list=model_name_list, train_cv=train_cv)
    # plot_result(test_score_result, 'Test', save_dir=save_sub_dir, model_name_list=model_name_list)

    # train_auc_result = np.array([train_score_result[sampling] for sampling in sampling_list])[:, :, 0]
    # auc_ci_to_excel(train_auc_result, train_auc_ci_result, sampling_list, 'Train', save_dir=save_sub_dir, model_name_list=model_name_list)
    # plot_best_params(model_name_list, best_params_result, train_auc_result, train_rsd_result, train_auc_ci_result, sampling_list, save_dir=save_sub_dir)

    # test_auc_result = np.array([test_score_result[sampling] for sampling in sampling_list])[:, :, 0]
    # auc_ci_to_excel(test_auc_result, test_auc_ci_result, sampling_list, 'Test', save_dir=save_sub_dir, model_name_list=model_name_list)


print("Finished...")