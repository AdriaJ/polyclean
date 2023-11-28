import pandas as pd
import os
import pickle

METRICS_FILE_NAME = 'metrics.csv'
PROPS_FILE_NAME = 'props.csv'
TMP_DATA_DIR = 'tmpdir'


def load_dfs(df_dir_path=None, filenames=None):
    """
    Open the two dataframes and load them as pd.DataFrame objects.
    """
    if filenames is None:
        filenames = [METRICS_FILE_NAME, PROPS_FILE_NAME]
    if df_dir_path is None:
        df_dir_path = os.getcwd()
    metrics_path = os.path.join(df_dir_path, filenames[0])
    props_path = os.path.join(df_dir_path, filenames[1])
    if os.path.exists(metrics_path):
        # assumes that if one exists, the other exists as well
        metrics_df = pd.read_csv(metrics_path, header=[0,1])
        props_df = pd.read_csv(props_path)
    else:
        iterables = [['wsclean', 'pclean', 'apgd'],  # 'pclean-ng' to consider one day maybe, 'monofw' removed
                     ['time', 'mse', 'mad', 'objf', 'dcv']]
        columns = pd.MultiIndex.from_product(iterables, names=['methods', 'metrics'])
        metrics_df = pd.DataFrame(columns=columns)
        props_df = pd.DataFrame(columns=['rmax', 'npix', 'nvis', 'seed', 'lips_t'])
    return metrics_df, props_df


def write_dfs(metrics_df: pd.DataFrame, props_df: pd.DataFrame, df_dir_path=None):
    """
    Save the two pd.DataFrame objects on disk, at the current working location.
    """
    if df_dir_path is None:
        df_dir_path = os.getcwd()
    metrics_path = os.path.join(df_dir_path, METRICS_FILE_NAME)
    props_path = os.path.join(df_dir_path, PROPS_FILE_NAME)
    metrics_df.to_csv(metrics_path, index=False)
    props_df.to_csv(props_path, index=False)


if __name__ == "__main__":
    with open(os.path.join(TMP_DATA_DIR, 'rmax_npix_seed.pkl'), 'rb') as f:
        props = pickle.load(f)
    with open(os.path.join(TMP_DATA_DIR, 'lips_t.pkl'), 'rb') as f:
        tmp_dict = pickle.load(f)
    props.update(tmp_dict)

    with open(os.path.join(TMP_DATA_DIR, 'lasso_res.pkl'), 'rb') as f:
        metrics = pickle.load(f)
    with open(os.path.join(TMP_DATA_DIR, 'clean_res.pkl'), 'rb') as f:
        tmp_dict = pickle.load(f)
    metrics.update(tmp_dict)
    metrics = {(k1, k2): metrics[k1][k2] for k1 in metrics.keys() for k2 in metrics[k1].keys()}

    metrics_df, props_df = load_dfs()

    metrics_df.loc[len(metrics_df.index)] = metrics
    props_df.loc[len(metrics_df.index)] = props

    write_dfs(metrics_df, props_df)

    # d = {'a':
    #          {'i': 1,
    #           'j': 2},
    #      'b':
    #          {'i': 3,
    #           'j': 4},
    #      }
    # d1 = {(k1, k2): [d[k1][k2]] for k1 in d.keys() for k2 in d[k1].keys()}
    # d2 = {(k1, k2): d[k1][k2] for k1 in d.keys() for k2 in d[k1].keys()}
    # df = pd.DataFrame(d1)
    # df.drop(0, inplace=True)
    # df.loc[len(df.index)] = d2
    # d2.pop(('a', 'j'))
    # df.loc[len(df.index)] = d2
