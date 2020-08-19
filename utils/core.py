from utils.delong import auc_ci
from utils.metrics import get_train_cv_performance_youdens, cutoff_youdens_j, get_performance

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

import xgboost
import lightgbm
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.ensemble        import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier

def model_initialization(params_dict):
    models_dict = {
        'KNN' : KNeighborsClassifier(),
        'SVM' : SVC(probability=True, random_state=42),
        'RANDOM_FOREST' : RandomForestClassifier(random_state=42),
        'EXTRA_TREES' : ExtraTreesClassifier(random_state=42),
        'ADABOOST' : AdaBoostClassifier(random_state=42),
        'XGBOOST' : xgboost.XGBClassifier(random_state=42),
        'LIGHTGBM' : lightgbm.LGBMClassifier(random_state=42),
    }

    model_name_list = models_dict.keys()
    params_name_list = params_dict.keys()

    for model_name in model_name_list:
        if model_name not in params_name_list:
            raise ValueError("Please specify hyper-parameter search space on options/search_grid.py")
    assert len(model_name_list) == len(params_name_list), "Check model list and search grid list."

    return models_dict

def model_fit(model, params, train_df, valid_df, model_name, opt):
    print(model_name, "Hyper-parameter Tuning...")

    # X,y split
    train_X = train_df.drop(opt.label_col, axis=1)
    train_y = train_df[opt.label_col]
    valid_X = valid_df.drop(opt.label_col, axis=1)
    valid_y = valid_df[opt.label_col]
    
    # Hyper-parameter Randomized Search
    kf = StratifiedKFold(n_splits=opt.search_cv, shuffle=True, random_state=42)
    model = RandomizedSearchCV(model,
                               params,
                               cv=kf,
                               scoring=opt.search_scoring,
                               n_iter=opt.search_n_iter)
    model.fit(train_X, train_y)

    # Train Result
    train_cv_result = get_train_cv_performance_youdens(model, train_df, kf, opt)

    # Get threshold cutoff using Youden's J index
    train_y_pred_proba = model.predict_proba(train_X)[:, 1]
    cutoff = cutoff_youdens_j(train_y, train_y_pred_proba)

    # Validation Result
    valid_result = dict()

    valid_y_pred_prob = model.predict_proba(valid_X)[:, 1]
    valid_y_pred = (valid_y_pred_prob > cutoff).astype(int)
    valid_auc, (valid_auc_95ci_lower, valid_auc_95ci_upper) = auc_ci(valid_y, valid_y_pred_prob, return_ci=True)
    valid_acc, valid_sens, valid_spec = get_performance(valid_y, valid_y_pred)

    valid_result['AUC'] = valid_auc
    valid_result['Accuracy'] = valid_acc
    valid_result['Sensitivity'] = valid_sens
    valid_result['Specificity'] = valid_spec
    valid_result['AUC_95CI_LOWER'] = valid_auc_95ci_lower
    valid_result['AUC_95CI_UPPER'] = valid_auc_95ci_upper

    # Print Result
    print(">>> Best params :", model.best_params_)
    print(">>> Train Score : AUC %.4f (95%% CI %.4f ~ %.4f) - ACC %.4f - Sens %.4f - Spec %.4f"
        % (*[train_cv_result[metric] for metric in ['AUC', 'AUC_95CI_LOWER', 'AUC_95CI_UPPER', 'Accuracy', 'Sensitivity', 'Specificity']],))
    print(">>> Valid Score : AUC %.4f (95%% CI %.4f ~ %.4f) - ACC %.4f - Sens %.4f - Spec %.4f\n\n"
        % (*[valid_result[metric] for metric in ['AUC', 'AUC_95CI_LOWER', 'AUC_95CI_UPPER', 'Accuracy', 'Sensitivity', 'Specificity']],))
    
    return model, train_cv_result, valid_result