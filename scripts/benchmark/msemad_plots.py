"""
Create plots for MSE and MAD metrics.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import use
import yaml

from fill_df import load_dfs
# plt.style.use('ggplot')

# use("Qt5Agg")

df_dir_path = '/home/jarret/Downloads/res/'
filenames = ['srf2reps10', 'srf5reps10', 'srf10reps10']
srfs = [2, 5, 10]

if __name__ == "__main__":
    msedfs = []
    maddfs = []
    for k in srfs:
        df_path = os.path.join(df_dir_path, 'srf' + str(k) + 'reps10')

        metrics_df, props_df = load_dfs(df_dir_path=df_path,)

        mse = pd.concat([props_df[['rmax']], metrics_df.xs('mse', level=1, axis=1)], axis=1)
        mse['srf'] = k
        mad = pd.concat([props_df[['rmax']], metrics_df.xs('mad', level=1, axis=1)], axis=1)
        mad['srf'] = k
        msedfs.append(mse)
        maddfs.append(mad)
    mse = pd.concat(msedfs)
    mad = pd.concat(maddfs)

    from mpl_toolkits.axes_grid1 import Grid

    fig = plt.figure(figsize=(16, 8))
    # mse
    grid = Grid(fig, 211,  # similar to subplot(111)
                     nrows_ncols=(1, 3),  # creates 2x2 grid of axes
                     axes_pad=0.2,  # pad between axes in inch.
                     )
    for ax, k in zip(grid, srfs):
        ax.set_yscale('log')
        ax.set_axisbelow(True)
        # ax.grid(axis='both', ls='--', color='grey', alpha=.5, zorder=0)
        # ax.grid( axis='x', zorder=-1, ls='--', color='grey', alpha=.5)
        col = ['wsclean', 'pclean', 'apgd']
        msek = mse[mse['srf'] == k]
        med = msek.groupby('rmax').median()
        quart1 = msek.groupby('rmax').quantile(.25)
        quart3 = msek.groupby('rmax').quantile(.75)
        for c, color in zip(col, plt.rcParams['axes.prop_cycle'].by_key()['color']):
            ax.scatter(med.index, med[c], color=color, marker='+', zorder=2)
            ax.fill_between(med.index, quart1[c], quart3[c], alpha=.2, color=color, zorder=1, label=c)
        ax.set_xticks([1000, 2000, 3000, 6000], ['']*4)
        ax.set_title("SRF " + str(k))
        ax.set_ylabel('MSE')
    ax.legend(fontsize=14, markerscale=2)

    #mad
    grid = Grid(fig, 212,  # similar to subplot(111)
                        nrows_ncols=(1, 3),  # creates 2x2 grid of axes
                        axes_pad=0.2,  # pad between axes in inch.
                        )
    for ax, k in zip(grid, srfs):
        ax.set_yscale('log')
        # ax.grid(axis='both', ls='--', color='grey', alpha=.5)
        col = ['wsclean', 'pclean', 'apgd']
        made = mad[mad['srf'] == k]
        med = made.groupby('rmax').median()
        quart1 = made.groupby('rmax').quantile(.25)
        quart3 = made.groupby('rmax').quantile(.75)
        for c, color in zip(col, plt.rcParams['axes.prop_cycle'].by_key()['color']):
            ax.scatter(med.index, med[c], color=color, marker='+')
            ax.fill_between(med.index, quart1[c], quart3[c], alpha=.2, color=color, label=c)
        ax.set_ylabel('MAD')
        ax.set_xlabel('rmax (m)')
        ax.set_xticks([1000, 2000, 3000, 6000])
    ax.legend(fontsize=14, markerscale=2)

    plt.subplots_adjust(hspace=.1)
    plt.savefig(os.path.join("/home/jarret/PycharmProjects/polyclean/figures/bench", "metrics_bench.pdf"))
    fig.show()
