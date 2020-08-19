import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_option(print_option=True):    
    p = argparse.ArgumentParser(description='')

    p.add_argument('--data_path', default='data/radiomic_features.csv', type=str, help='csv data path')
    p.add_argument('--label_col', type=str, required=True, help='column name of label in data excel file')
    p.add_argument('--exclude_cols', type=str, help='name of columns not to use \
                                                    (Use , for multiple columns : ex. patientID,date')
    p.add_argument('--train_valid_ratio', type=float, default=0.7, help='Ratio of train-valid dataset split')

    # Feature selection options
    p.add_argument('--lasso_cv', type=int, default=10)
    p.add_argument('--lasso_alpha', type=float, default=5e-3)
    p.add_argument('--lasso_tol', type=float, default=1e-2)
    p.add_argument('--lasso_threshold', type=float, default=5e-2)
    p.add_argument('--fscore_n_feats', type=int, default=30)
    p.add_argument('--mi_n_feats', type=int, default=30)

    # Hyper-parameter
    p.add_argument('--search_cv', type=int, default=10)
    p.add_argument('--search_scoring', type=str, default='f1')
    p.add_argument('--search_n_iter', type=int, default=100)

    p.add_argument('--exp', default='exp', type=str, help='output dir.')

    opt = p.parse_args()
    
    # Make output directory
    if not os.path.exists(opt.exp):
        os.makedirs(opt.exp)

    if print_option:
        print("\n============================= Options =============================\n")
    
        print('   Data path : %s' % opt.data_path)
        print('   Label column name : %s' % opt.label_col)
        print('   Exclude column name : %s' % opt.exclude_cols)
        print('   Train-Valid Ratio : %s' % opt.train_valid_ratio)
        print('   Output dir : %s' % opt.exp)

        print("\n===================================================================\n")

    return opt