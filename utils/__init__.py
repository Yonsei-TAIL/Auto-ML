import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def load_data(opt):
    # Load data
    if opt.data_path.endswith('.csv'):
        data_df = pd.read_csv(opt.data_path)
    elif opt.data_path.endswith('.xlsx'):
        data_df = pd.read_excel(opt.data_path)
    else:
        raise ValueError('%s extension is not supported now. Please use csv or xlsx extension.'
                        % opt.data_path.split('.')[-1])
    
    # Drop columns not to use
    data_df = data_df.drop(opt.exclude_cols.split(','), axis=1)

    return data_df

def train_valid_split(data_df, opt):
    data_df = data_df.sample(frac=1.0) # Shuffle
    
    uniq_labels = data_df[opt.label_col].unique()
    
    train_data = pd.DataFrame()
    valid_data  = pd.DataFrame()

    for label in uniq_labels:
        df_w_uniq_label = data_df[data_df[opt.label_col] == label]

        n_rows = len(df_w_uniq_label)
        split_idx = int(n_rows * opt.train_valid_ratio)

        df_uniq_train = df_w_uniq_label.iloc[:split_idx].sort_index()
        df_uniq_test  = df_w_uniq_label.iloc[split_idx:].sort_index()

        train_data = pd.concat([train_data, df_uniq_train])
        valid_data  = pd.concat([valid_data,  df_uniq_test ])

    # Export split data to disk
    train_data.to_csv(os.path.join(opt.exp, 'train_dataset.csv'))
    valid_data.to_csv(os.path.join(opt.exp, 'valid_dataset.csv'))


    print("Dataset Split : %d Train - %d Valid (Ratio %s)" % (len(train_data), len(valid_data), opt.train_valid_ratio))
    return train_data, valid_data