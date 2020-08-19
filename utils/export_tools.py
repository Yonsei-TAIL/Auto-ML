def save_train_test_split(train_data, test_data, save_dir, index_name, label_col):
    train_data = train_data.copy()
    test_data  = test_data.copy()

    if train_data.index.name != index_name:
        train_data.index.name = index_name
        test_data.index.name = index_name

    train_data.reset_index(level=0, inplace=True)
    test_data.reset_index(level=0, inplace=True)

    train_data['set'] = 'train'
    test_data['set'] = 'test'

    train_data = train_data[[index_name, label_col, 'set'] + train_data.columns.to_list()[2:-1]]
    test_data  = test_data [[index_name, label_col, 'set'] + test_data.columns.to_list()[2:-1]]

    train_test_data = pd.concat([train_data, test_data])
    train_test_data.to_excel(os.path.join(save_dir, 'Train_test_set.xlsx'), index=False)

            model, train_cv_result, valid_result = model_fit(model, params, train_df_sampled, valid_df, model_name, opt)

def plot_calibration_curve(result_dict, train_data, test_data, opt, n_bins=10):
    plot_calibration_curve(trained_model_list, model_name_list, X_train, y_train, label_name, sampling_name, 'Train', save_dir, n_bins)
    plot_calibration_curve(trained_model_list, model_name_list, X_test,  y_test,  label_name, sampling_name, 'Test',  save_dir, n_bins)

def plot_calibration_curve(result_dict, train_df, valid_df, save_dir, n_bins=10):
    train_X = train_df.drop(opt.label_col, axis=1)
    train_y = train_df[opt.label_col]
    valid_X = valid_df.drop(opt.label_col, axis=1)
    valid_y = valid_df[opt.label_col]
    
    for X, y, set_name in zip([train_X, valid_X], [train_y, valid_y], ['Train', 'Validation']):
        for sampling_name, (model_name, model, _, _) in result_dict.items():
            plt.figure(figsize=(20, 10))
            title = '%s %s Oversampling Calibration Curve' % (set_name, sampling_name)

            y_pred = model.predict_proba(X)[:, 1]
            fraction_of_positives, mean_predicted_value = calibration_curve(y, y_pred, n_bins=n_bins)

            plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label="%s" % (model_name))
                
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated",)
        plt.xlabel("Nomogram Predicted Probability", fontsize=20)
        plt.ylabel("Actual %s Rate" % (label_name), fontsize=20)

        plt.ylim([-0.05, 1.05])
        plt.legend(loc="lower right", prop={'size': 20})
        plt.title(title, fontsize=20)
        plt.savefig(os.path.join(save_dir, title+'.png'))


def plot_result(score_result, subset, save_fig=True, save_dir='plot_result',
                sampling_list=['No','SMOTE','ROSE'], metrics_name_list=['AUC','Accuracy','Sensitivity','Specificity'],
                model_name_list=['SVM', 'Random Forest', 'Extra Trees', 'AdaBoost', 'XGBoost', 'LightGBM'],
                train_cv=None):


    # Colormap
    cmap = sns.diverging_palette(150, 10, as_cmap=True)
    
    # Make Directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    score_values = np.array([score_result[sampling] for sampling in sampling_list])
            
    for i, metrics_name in enumerate(metrics_name_list):
        title = '{} Dataset {} Score'.format(subset, metrics_name_list[i])
        title = (title + ' (%d CV)' % train_cv) if subset == 'Train' else title
        
        metrics_values = score_values[:, :, i].T

        plt.figure(figsize=(13, 6))

        sns.set(font_scale = 1.4)
        ax = sns.heatmap(metrics_values, cmap=cmap, linewidths=.5, linecolor='black', annot=True, fmt='.3f', annot_kws={"size": 17},
                    xticklabels=sampling_list, yticklabels=model_name_list)
        ax.set_title(title)


        if save_fig:
            plt.savefig(os.path.join(save_dir, title + '.png'))
        
        else:
            plt.show()

def auc_to_str(auc_score):
    auc, (auc_ci_lower, auc_ci_upper) = auc_score
    
    return '%.3f [%.3f - %.3f]' % (auc, auc_ci_lower, auc_ci_upper)

def auc_ci_to_excel(auc_result, auc_ci_result, sampling_list, subset, save_dir='plot_result', model_name_list=['SVM', 'Random Forest', 'Extra Trees', 'AdaBoost', 'XGBoost', 'LightGBM']):
    auc_ci_values = [auc_ci_result[sampling] for sampling in sampling_list]
    auc_ci_result = dict(zip(sampling_list, [[auc_to_str([auc, s2]) for auc, s2 in zip(aucs, s1)] for aucs, s1 in zip(auc_result, auc_ci_values)]))
    auc_ci_df = pd.DataFrame(auc_ci_result, index=model_name_list)
    auc_ci_df = auc_ci_df[sampling_list]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    auc_ci_df.to_excel(os.path.join(save_dir, '{}_AUC_CI_Result.xlsx'.format(subset)))

def plot_best_params(model_name_list, best_params_result, train_auc_result, train_rsd_result, train_auc_ci_result, sampling_list, save_dir='plot_result'):
    best_params_values = [best_params_result[sampling] for sampling in sampling_list]
    best_params_array = np.array(best_params_values).T

    train_rsd_values = [train_rsd_result[sampling] for sampling in sampling_list]
    rsd_array = np.array(train_rsd_values).T
    auc_array = train_auc_result.T
    train_auc_ci_values = [train_auc_ci_result[sampling] for sampling in sampling_list]
    auc_ci_array = np.array(train_auc_ci_values).reshape(7, 3, 2)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for model_name, best_params, auc, auc_ci, rsd in zip(model_name_list, best_params_array, auc_array, auc_ci_array, rsd_array):
        model_df = pd.DataFrame()
        
        for i, sampling_name in enumerate(sampling_list):
            df_row = pd.DataFrame(best_params[i], index=[sampling_name])
            df_row['mean_AUC (95% CI)'] = '%.3f [%.3f - %.3f]' % (auc[i], *auc_ci[i])
            df_row['RSD (%)'] = rsd[i]
            
            model_df = model_df.append(df_row)
        
        model_df.to_excel(os.path.join(save_dir, '%s_hyperparameter_tuned_result.xlsx' % model_name))