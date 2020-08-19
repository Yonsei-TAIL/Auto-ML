# Randomized Search Parameter Grid Setting
# You can control hyper-paramter search grind in this python script as our examples.
# Ex. 'MODEL_NAME' : {'param_name1' : range, 'param_name2' : range, ...}

params_dict = {
    'KNN' : {'n_neighbors' : range(2, 20, 5),
             'weights' : ['uniform', 'distance'],
             'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
             'leaf_size' : range(4, 61, 2)},
    
    'SVM' : {'C': [1.0, 3.0, 5.0, 7.0, 9.0],
             'kernel': ['linear', 'rbf'],
             'class_weight' : [None, 'balanced'],
             'gamma' : ['scale', 'auto', 0.1, 1.0, 10.0]},

    'RANDOM_FOREST' : {'n_estimators': range(50, 401, 30),
                       'criterion' : ['gini', 'entropy'],
                       'max_depth': range(3, 31, 3),
                       'min_samples_split' : range(2, 15, 3),
                       'min_samples_leaf' : [1],
                       'max_features' : [0.1, 0.3, 0.5, 0.7, 0.9]},

    'EXTRA_TREES' : {'n_estimators': range(50, 401, 30),
                     'max_depth': range(3, 31, 2),
                     'min_samples_leaf' : [1,2,3,4],
                     'bootstrap' : [True, False]},

    'ADABOOST' : {'n_estimators': range(50, 401, 30),
                  'learning_rate': [0.01,0.1,0.5,1.0]},

    'XGBOOST' : {'learning_rate': [0.01,0.1,0.5,1.0],
                 'max_depth': range(3, 31, 2),
                 'gamma' : [0.0,0.1,0.3,0.5,0.7,1.0]},

    'LIGHTGBM' : {'learning_rate': [0.01,0.1,0.5,1.0],
                'n_estimators': range(50, 401, 30),
                'min_split_gain' : [0.0,0.1,0.3,0.5,1.0]}
    }