"""
Open the two dataframes and load them as pd.DataFrame objects.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import use
import yaml

from fill_df import load_dfs
# plt.style.use('ggplot')

# use("Qt5Agg")

df_dir_path = 'archive'

exp_name = '6reps_server2'  # '2reps_local'

if __name__ == "__main__":
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    time_factor = config['monofw_params']['max_time_factor']

    metrics_df, props_df = load_dfs(df_dir_path=df_dir_path, filenames=[n + exp_name + '.csv' for n in ['metrics_', 'props_']])

    # metrics_df.loc[:, ('apgd', 'objf')] = metrics_df['apgd']['objf'].apply(lambda x: x[1:-1]).astype(float).values
    # metrics_df.loc[:, ('monofw', 'objf')] = metrics_df.loc[:, ('monofw', 'objf')].apply(lambda x: x[1:-1]).astype(float)
    # from fill_df import write_dfs
    # write_dfs(metrics_df, props_df, df_dir_path=df_dir_path)


    objf = pd.concat([props_df[['rmax']], metrics_df.xs('objf', level=1, axis=1).drop(columns=['wsclean'])], axis=1)
    dcv = pd.concat([props_df[['rmax']], metrics_df.xs('dcv', level=1, axis=1).drop(columns=['wsclean'])], axis=1)
    mse = pd.concat([props_df[['rmax']], metrics_df.xs('mse', level=1, axis=1)], axis=1)
    mad = pd.concat([props_df[['rmax']], metrics_df.xs('mad', level=1, axis=1)], axis=1)
    time = pd.concat([props_df[['rmax', 'lips_t']], metrics_df.xs('time', level=1, axis=1)], axis=1)

    # style = ['o']*4 #['xr', '+b', '*g', '*']
    # # same without Lipschitz
    # logy = True
    # plt.figure(figsize=(10, 10))
    # ax = plt.gca()
    # time.plot(x='rmax', y=['wsclean', 'pclean', 'apgd', 'monofw'], ax=ax, logy=logy,
    #           style=style, xticks=time['rmax'].values, grid=True)
    # ax.scatter(time['rmax'], time_factor * time['pclean'], marker='1', color='k',
    #            label='time limit', s=100)
    # ax.set_ylabel('time (s)')
    # ax.set_title("Time comparison")
    # plt.show()
    # # plt.savefig(os.path.join(df_dir_path, "time_comparison.png"))

    ### TIME COMPARISON ###
    meds = time.groupby('rmax').median()
    quart1 = time.groupby('rmax').quantile(.25)
    quart3 = time.groupby('rmax').quantile(.75)
    col = ['wsclean', 'pclean', 'apgd',] # 'monofw']
    side_size = props_df[['rmax', 'npix', 'nvis']].groupby('rmax').agg(lambda x: x.iloc[0])
    side_size['imsize_mpix'] = (side_size['npix'] ** 2) / 1.e+6

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    for c, color in zip(col, plt.rcParams['axes.prop_cycle'].by_key()['color']):
        ax.scatter(side_size.loc[meds.index]['imsize_mpix'], meds[c], marker='o', label=c, color=color)
        ax.fill_between(side_size.loc[meds.index]['imsize_mpix'], quart1[c], quart3[c], alpha=.2, color=color)
    ax.set_ylabel('time (s)')
    ax.set_title("Time comparison")
    ax.set_xlabel('Image size (MPix)')
    ax.set_xticks(side_size.loc[meds.index]['imsize_mpix'],
                  labels=[f"{s:.1f}" for s in side_size.loc[meds.index]['imsize_mpix']], minor=True)
    ax.set_xticks([1, 10, 100], labels=['', '', 100.0])
    ax.legend()
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    # ax.minorticks_off()
    ax2 = ax.secondary_xaxis('top')
    ax2.set_ticks(side_size.loc[meds.index]['imsize_mpix'], labels=[f"{r/1000}" for r in meds.index])
    ax2.set_xlabel("rmax (km)")
    ax2.minorticks_off()
    plt.show()



    ### DCV ###
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    dcv.plot(x='rmax', y=['pclean', 'apgd', 'monofw'], ax=ax, logy=False, style=['o', '+', 'x'], xticks=time['rmax'].values, grid=True)
    ax.set_ylabel("Dual certificate value")
    ax.set_title("Dual certificate comparison")
    plt.show()

    ### Objective function ###
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.scatter(objf['rmax'], (objf['apgd'] - objf['pclean'])/objf['pclean'], marker='x', label='apgd')
    ax.scatter(objf['rmax'], (objf['monofw'] - objf['pclean'])/objf['pclean'], marker='+', label='monofw')
    ax.set_ylabel("Objective function relative difference")
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xticks(objf['rmax'].values)
    ax.legend()
    ax.set_title("Objective function relative difference with PolyCLEAN")
    plt.show()

    ### MSE & MAD ###
    from matplotlib.lines import Line2D
    col = ['wsclean', 'pclean', 'apgd']
    meds = [m.groupby('rmax').median() for m in (mse, mad)]
    quart1 = [m.groupby('rmax').quantile(.25) for m in (mse, mad)]
    quart3 = [m.groupby('rmax').quantile(.75) for m in (mse, mad)]
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_yscale('log')
    legend_elements = [Line2D([0], [0], marker='x', color='grey',  label='MSE', lw=0),
                       Line2D([0], [0], marker='+', color='grey', label='MAD', lw=0)]
    for c, color in zip(col, plt.rcParams['axes.prop_cycle'].by_key()['color']):
        legend_elements.append(Line2D([0], [0], color=color, label=c, lw=2))
        for i in range(2):
            m = meds[i]
            marker = 'x' if i == 0 else '+'
            ax.scatter(m.index, m[c], marker=marker, color=color)
            ax.fill_between(m.index, quart1[i][c], quart3[i][c], alpha=.2, color=color)
    ax.legend(handles=legend_elements)
    ax.set_title("MSE & MAD comparison")
    plt.show()

    # side_size.plot(marker='+')
    # (side_size['npix'].values[1:] - side_size['npix'].values[:-1])/(side_size.index.values[1:] - side_size.index.values[:-1])

    time.groupby('rmax').mean().sum(axis=0)

    latex = False
    if latex:
        lips = time.groupby('rmax')['lips_t'].mean()
        summary = side_size.copy().reset_index()
        summary['lips_t'] = lips.reset_index()['lips_t']
        summary.rename(columns={'rmax': 'rmax (km)', 'npix': 'npix (side size)', 'nvis': 'nvis (k)', 'imsize_mpix': 'imsize (MPix)', 'lips_t': 'lipschitz time (s)'}, inplace=True)
        print(summary.to_latex(index=False, float_format='%.1f', formatters={'nvis (k)': lambda x: f"{x/1000:.1f}"}))
        mse.drop(columns='monofw', inplace=True)
        mad.drop(columns='monofw', inplace=True)
        mse = mse.groupby('rmax').median().reset_index()
        mse['rmax'] = mse.rmax.astype(int)
        print(mse.to_latex(index=False, float_format='%.2e'))
        mad = mad.groupby('rmax').median().reset_index()
        mad['rmax'] = mad.rmax.astype(int)
        print(mad.to_latex(index=False, float_format='%.2e'))
