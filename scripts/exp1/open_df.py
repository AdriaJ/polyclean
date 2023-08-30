"""
Open the two dataframes and load them as pd.DataFrame objects.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import use
import yaml

from fill_df import load_dfs

use("Qt5Agg")

df_dir_path = 'archive'

if __name__ == "__main__":
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    metrics_df, props_df = load_dfs(df_dir_path=df_dir_path)

    # metrics_df.loc[:, ('apgd', 'objf')] = metrics_df['apgd']['objf'].apply(lambda x: x[1:-1]).astype(float).values
    # metrics_df.loc[:, ('monofw', 'objf')] = metrics_df.loc[:, ('monofw', 'objf')].apply(lambda x: x[1:-1]).astype(float)
    # from fill_df import write_dfs
    # write_dfs(metrics_df, props_df, df_dir_path=df_dir_path)


    objf = pd.concat([props_df[['rmax']], metrics_df.xs('objf', level=1, axis=1).drop(columns=['wsclean'])], axis=1)
    dcv = pd.concat([props_df[['rmax']], metrics_df.xs('dcv', level=1, axis=1).drop(columns=['wsclean'])], axis=1)
    mse = pd.concat([props_df[['rmax']], metrics_df.xs('mse', level=1, axis=1)], axis=1)
    mad = pd.concat([props_df[['rmax']], metrics_df.xs('mad', level=1, axis=1)], axis=1)
    time = pd.concat([props_df[['rmax', 'lips_t']], metrics_df.xs('time', level=1, axis=1)], axis=1)

    logy = True
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    time.plot(x='rmax', y=['wsclean', 'pclean', 'apgd', 'monofw', 'lips_t'], ax=ax, logy=logy,
              style=['x', '+', 'o', 'o', 'd'], xticks=time['rmax'].values, grid=True)
    ax.scatter(time['rmax'], config['monofw_params']['max_time_factor'] * time['pclean'], marker='1', color='k',
               label='time limit', s=100)
    ax.set_ylabel('time (s)')
    ax.set_title("Time comparison")
    plt.show()

    # do the same plots for the other metrics
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    dcv.plot(x='rmax', y=['pclean', 'apgd', 'monofw'], ax=ax, logy=False, style=['o', '+', 'x'], xticks=time['rmax'].values, grid=True)
    ax.set_ylabel("Dual certificate value")
    ax.set_title("Dual certificate comparison")
    plt.show()

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.scatter(objf['rmax'], (objf['apgd'] - objf['pclean'])/objf['pclean'], marker='x', label='apgd')
    ax.scatter(objf['rmax'], (objf['monofw'] - objf['pclean'])/objf['pclean'], marker='+', label='monofw')
    ax.set_ylabel("Objective function relative difference")
    # draw a horizontal line on 0
    ax.axhline(0, color='k', linestyle='--')
    # set x ticks with rmax
    ax.set_xticks(objf['rmax'].values)
    ax.legend()
    ax.set_title("Objective function relative difference with PolyCLEAN")
    # objf.plot(x='rmax', y=['pclean', 'apgd', 'monofw'], ax=ax, logy=True, style=['o', '+', 'x'], xticks=time['rmax'].values, grid=True)
    plt.show()

