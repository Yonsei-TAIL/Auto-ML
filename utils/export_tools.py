import os
import numpy as np
import pandas as pd

def export_result_to_excel(result_dict_fs, save_dir):
    # Result DataFrame
    train_result_df = pd.DataFrame()
    valid_result_df = pd.DataFrame()
    best_params_df  = pd.DataFrame()

    # Append Result Rows to DataFrame
    for sampling_name, result in result_dict_fs.items():
        for model_name, (model, train_score, valid_score) in result.items():
            model_sampling_name = '%s+%s'%(model_name,sampling_name)
            
            train_result_row = pd.DataFrame.from_dict(train_score, orient='index', columns=[model_sampling_name]).T
            valid_result_row = pd.DataFrame.from_dict(valid_score, orient='index', columns=[model_sampling_name]).T
            best_params_row  = pd.DataFrame(['%s = %s'%(n, v) for n, v in model.best_params_.items()], columns=[model_sampling_name]).T

            train_result_df = train_result_df.append(train_result_row)
            valid_result_df = valid_result_df.append(valid_result_row)
            best_params_df = best_params_df.append(best_params_row)

    # Raname DataFrame's index
    train_result_df.index.name = 'Model+Sampling'
    valid_result_df.index.name = 'Model+Sampling'
    best_params_df.index.name = 'Model+Sampling'

    # Round to three decimal places
    train_result_df = train_result_df.apply(lambda x : np.round(x, 3))
    valid_result_df = valid_result_df.apply(lambda x : np.round(x, 3))

    # Concat AUC, AUC_95CI_LOWER, AUC_95CI_UPPER Columns to Single Columne
    if ('AUC' in train_result_df.columns) and ('AUC_95CI_LOWER' in train_result_df.columns) and ('AUC_95CI_UPPER' in train_result_df.columns):
        train_result_df['AUC (95% CI)'] = ['%s (%s-%s)' % (*row,) for row in train_result_df[['AUC', 'AUC_95CI_LOWER', 'AUC_95CI_UPPER']].values]
        valid_result_df['AUC (95% CI)'] = ['%s (%s-%s)' % (*row,) for row in valid_result_df[['AUC', 'AUC_95CI_LOWER', 'AUC_95CI_UPPER']].values]

        train_result_df.drop(['AUC', 'AUC_95CI_LOWER', 'AUC_95CI_UPPER'], 1, inplace=True)
        valid_result_df.drop(['AUC', 'AUC_95CI_LOWER', 'AUC_95CI_UPPER'], 1, inplace=True)

    # Export to excel
    train_result_df.to_excel(os.path.join(save_dir, 'train_dataset_performance.xlsx'))
    valid_result_df.to_excel(os.path.join(save_dir, 'valid_dataset_performance.xlsx'))
    best_params_df.to_excel(os.path.join(save_dir, 'hyper-parameters.xlsx'))


# def plot_calibration_curve(result_dict, train_data, test_data, opt, n_bins=10):
#     plot_calibration_curve(trained_model_list, model_name_list, X_train, y_train, label_name, sampling_name, 'Train', save_dir, n_bins)
#     plot_calibration_curve(trained_model_list, model_name_list, X_test,  y_test,  label_name, sampling_name, 'Test',  save_dir, n_bins)

# def plot_calibration_curve(result_dict, train_df, valid_df, save_dir, n_bins=10):
#     train_X = train_df.drop(opt.label_col, axis=1)
#     train_y = train_df[opt.label_col]
#     valid_X = valid_df.drop(opt.label_col, axis=1)
#     valid_y = valid_df[opt.label_col]
    
#     for X, y, set_name in zip([train_X, valid_X], [train_y, valid_y], ['Train', 'Validation']):
#         for sampling_name, (model_name, model, _, _) in result_dict.items():
#             plt.figure(figsize=(20, 10))
#             title = '%s %s Oversampling Calibration Curve' % (set_name, sampling_name)

#             y_pred = model.predict_proba(X)[:, 1]
#             fraction_of_positives, mean_predicted_value = calibration_curve(y, y_pred, n_bins=n_bins)

#             plt.plot(mean_predicted_value, fraction_of_positives, "s-",
#                     label="%s" % (model_name))
                
#         plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated",)
#         plt.xlabel("Nomogram Predicted Probability", fontsize=20)
#         plt.ylabel("Actual %s Rate" % (label_name), fontsize=20)

#         plt.ylim([-0.05, 1.05])
#         plt.legend(loc="lower right", prop={'size': 20})
#         plt.title(title, fontsize=20)
#         plt.savefig(os.path.join(save_dir, title+'.png'))