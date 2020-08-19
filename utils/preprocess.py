import pandas as pd
import numpy as np

def zscore_normalization(train_df, valid_df, opt):
    # Feature Values
    train_values = train_df.drop(opt.label_col, axis=1).values

    # Calculate MEAN and STD of each feature on train data values
    MEAN = np.nanmean(train_values, axis=0)
    STD  = np.nanstd(train_values, axis=0)

    # Apply Z-score Normalization to DataFrame
    train_df_normed = train_df.copy()
    valid_df_normed = valid_df.copy()

    train_df_normed.loc[:, train_df.columns != opt.label_col] = (train_df.drop(opt.label_col, axis=1)  - MEAN) / STD
    valid_df_normed.loc[:, valid_df.columns != opt.label_col] = (valid_df.drop(opt.label_col, axis=1)  - MEAN) / STD

    return train_df_normed, valid_df_normed


def fill_NaN_w_mean(train_df, valid_df):
    MEAN = train_df.mean()

    train_df_filled = train_df.fillna(MEAN).copy()
    valid_df_filled = valid_df.fillna(MEAN).copy()

    return train_df_filled, valid_df_filled