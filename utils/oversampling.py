import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler

def oversample_SMOTE(train_df, opt):
    # X,y split
    train_X = train_df.drop(opt.label_col, axis=1)
    train_y = train_df[opt.label_col]

    # Oversampling with SMOTE
    train_X_smote, train_y_smote = SMOTE().fit_sample(train_X, train_y)
    train_df_SMOTE = pd.concat([train_y_smote, train_X_smote], axis=1)

    return train_df_SMOTE

def oversample_ROSE(train_df, opt):
    # X,y split
    train_X = train_df.drop(opt.label_col, axis=1)
    train_y = train_df[opt.label_col]

    # Oversampling with ROSE
    train_X_rose, train_y_rose = RandomOverSampler().fit_sample(train_X, train_y)
    train_df_ROSE = pd.concat([train_y_rose, train_X_rose], axis=1)

    return train_df_ROSE