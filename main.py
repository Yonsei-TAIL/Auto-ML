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
from utils.export_tools import export_result_to_excel

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
    # Make save directory
    save_dir_fs = os.path.join(opt.exp, '%s_feature_selection'%feature_selection_name)
    if not os.path.exists(save_dir_fs):
        os.makedirs(save_dir_fs)

    # Oversampling : SMOTE & ROSE
    dataset_dict_os = {'NO' : train_df}
    dataset_dict_os['SMOTE'] = oversample_SMOTE(train_df, opt)
    dataset_dict_os['ROSE'] = oversample_ROSE(train_df, opt)

    # Result container
    result_dict_fs = dict()

    # Main
    for sampling_name, train_df_sampled in dataset_dict_os.items():
        print("\n=====================================================")
        print(" Start : %s Oversampling [%s Feature Selection] " % (sampling_name, feature_selection_name))
        print("=====================================================\n")

        # Model
        result_dict_sp = dict()
        models_dict = model_initialization(params_dict)
        for model_name, model in models_dict.items():
            params = params_dict[model_name]

            model, train_cv_result, valid_result = model_fit(model, params, train_df_sampled, valid_df, model_name, opt)
            result_dict_sp[model_name] = [model, train_cv_result, valid_result]
        
        result_dict_fs[sampling_name] = result_dict_sp

    # Export results to excel file
    export_result_to_excel(result_dict_fs, save_dir_fs)
    # plot_calibration_curve_result(no_trained_model_list, model_name_list, train_X_selected, train_y, test_X_selected, test_y, label_col, 'No', save_sub_dir)


print("Finished...")