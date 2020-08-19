import os
import pandas as pd

from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LassoCV

def LassoSelection(train_df, valid_df, opt):
    train_X = train_df.drop(opt.label_col, axis=1)
    train_y = train_df[opt.label_col]

    n_feats = train_X.shape[1] # Number of features

    clf = LassoCV(cv=opt.lasso_cv,
                  alphas=[opt.lasso_alpha]*n_feats,
                  tol=opt.lasso_tol)
    sfm = SelectFromModel(clf, threshold=opt.lasso_threshold)
    sfm.fit(train_X, train_y)

    selected_feats = train_X.columns[sfm.get_support()].to_list()
    print(">>> # of Lasso Selected Features : %d" % len(selected_feats))

    train_df_lasso = train_df[[opt.label_col]+selected_feats]
    valid_df_lasso = valid_df[[opt.label_col]+selected_feats]

    # Save selected features list as excel file
    file_name = os.path.join(opt.exp, 'Lasso_Selected_Features.xlsx')
    pd.DataFrame(selected_feats).to_excel(file_name, index=False, header=False)

    return train_df_lasso, valid_df_lasso


def FscoreSelection(train_df, valid_df, opt):
    train_X = train_df.drop(opt.label_col, axis=1)
    train_y = train_df[opt.label_col]

    f_selector = SelectKBest(f_classif, k=opt.fscore_n_feats)
    f_selector.fit(train_X, train_y)

    selected_feats = train_X.columns[f_selector.get_support()].to_list()
    print(">>> # of F-score Selected Features : %d" % len(selected_feats))

    train_df_fscore = train_df[[opt.label_col]+selected_feats]
    valid_df_fscore = valid_df[[opt.label_col]+selected_feats]

    # Save selected features list as excel file
    file_name = os.path.join(opt.exp, 'Fscore_Selected_Features.xlsx')
    pd.DataFrame(selected_feats).to_excel(file_name, index=False, header=False)

    return train_df_fscore, valid_df_fscore


def MISelection(train_df, valid_df, opt):
    train_X = train_df.drop(opt.label_col, axis=1)
    train_y = train_df[opt.label_col]

    mi_selector = SelectKBest(mutual_info_classif, k=opt.mi_n_feats)
    mi_selector.fit(train_X, train_y)

    selected_feats = train_X.columns[mi_selector.get_support()].to_list()
    print(">>> # of Mutual Information Selected Features : %d" % len(selected_feats))

    train_df_mi = train_df[[opt.label_col]+selected_feats]
    valid_df_mi = valid_df[[opt.label_col]+selected_feats]

    # Save selected features list as excel file
    file_name = os.path.join(opt.exp, 'MI_Selected_Features.xlsx')
    pd.DataFrame(selected_feats).to_excel(file_name, index=False, header=False)

    return train_df_mi, valid_df_mi