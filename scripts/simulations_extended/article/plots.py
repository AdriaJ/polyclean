import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def truncate_colormap(cmap, minval, maxval, n=100):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mplc.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_3_images(im_list, title_list, suptitle="", normalize=True, offset_cm=0., vlim=None, alpha=.8):
    chan, pol = 0, 0
    set_vlim = vlim is None
    if normalize:
        vmax = max([np.abs(im.pixels.data).max() for im in im_list])
        if set_vlim:
            vlim = max(0, -min([im.pixels.data.min() for im in im_list]))
    else:
        vmax = None
    any_lim = False

    fig = plt.figure(figsize=(15, 5))
    axes = fig.subplots(1, 3, sharex=True, sharey=True,
                        subplot_kw={'projection': im_list[0].image_acc.wcs.sub([1, 2]), 'frameon': False})
    for i in range(3):
        ax = axes[i]
        arr = np.real(im_list[i]["pixels"].data[chan, pol, :, :])
        if not normalize:
            if i == 0:
                vmax = np.abs(arr).max()
                if set_vlim:
                    vlim = max(0, -arr.min())
            else:
                vmax = max([np.abs(im.pixels.data).max() for im in im_list[1:]])
                if set_vlim:
                    vlim = max(0, -min([im.pixels.data.min() for im in im_list[1:]]))
        # ims = ax.imshow(arr, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
        im_pos = np.ma.masked_array(arr, arr < vlim, fill_value=vlim)
        if (arr < vlim).any():
            im_neg = np.ma.masked_array(arr, arr > vlim, fill_value=vlim)
            cmapn = truncate_colormap('Greys', offset_cm, 1. - offset_cm)
            aximn = ax.imshow(im_neg, origin="lower", cmap=cmapn, interpolation='none', alpha=alpha,
                              vmin=-vlim, vmax=vlim)
            any_lim = True
        cmapp = truncate_colormap('hot', offset_cm, 1.)
        aximp = ax.imshow(im_pos, origin="lower", cmap=cmapp, interpolation='none', vmax=vmax, vmin=vlim)
        if i == 0:
            ax.set_ylabel(im_list[i].image_acc.wcs.wcs.ctype[1])
        else:
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')
        ax.set_xlabel(im_list[i].image_acc.wcs.wcs.ctype[0])
        ax.set_title(title_list[i])
        # fig.colorbar(ims, orientation="vertical", shrink=0.5, ax=ax)
        if i == 2:
            axinsp = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
            cbp = fig.colorbar(aximp, cax=axinsp, orientation="vertical")
            if any_lim:
                axinsn = inset_axes(axinsp, width="100%", height="100%", loc='center right', borderpad=-4)
                cbn = fig.colorbar(aximn, cax=axinsn, orientation="vertical")
        if not normalize and i == 0:
            axinsp = inset_axes(ax, width="3%", height="100%", loc='center left', borderpad=-8)
            cbp = fig.colorbar(aximp, cax=axinsp, orientation="vertical")
            if any_lim:
                axinsn = inset_axes(axinsp, width="100%", height="100%", loc='center left', borderpad=-4)
                cbn = fig.colorbar(aximn, cax=axinsn, orientation="vertical")

    fig.suptitle(suptitle)
    # plt.subplots_adjust(
    #     top=0.918,
    #     bottom=0.027,
    #     left=0.047,
    #     right=0.991,
    #     hspace=0.2,
    #     wspace=0.023)
    plt.show()

if __name__=="__main__":
    # load the images from the pkl files
    folders = ['pclean', 'rsclean', 'wsclean']
    names = ['comp.pkl', 'comp_conv.pkl', 'comp_conv_res.pkl']
    images = [[], [], []]
    for i, name in enumerate(names):
        for folder in folders:
            with open(os.path.join(os.getcwd(), 'reco_pkl', folder, name), 'rb') as handle:
                images[i].append(pickle.load(handle))

    with open(os.path.join(os.getcwd(), 'reco_pkl', 'dirty.pkl'), 'rb') as handle:
        dirty = pickle.load(handle)
    with open(os.path.join(os.getcwd(), 'reco_pkl', 'source.pkl'), 'rb') as handle:
        source = pickle.load(handle)
    with open(os.path.join(os.getcwd(), 'reco_pkl', 'source_conv.pkl'), 'rb') as handle:
        source_conv = pickle.load(handle)

    # RASCIL CLEAN
    plot_3_images([source, ] + images[0][:-1],
                  ['Source', 'PolyCLEAN', 'CLEAN'],
                  suptitle="Comparison components",
                  normalize=True, vlim=0.3, alpha=1)
    plot_3_images([source_conv, ] + images[1][:-1],
                  ['Source', 'PolyCLEAN', 'CLEAN'],
                  suptitle="Comparison components convolved sharp",
                  normalize=True)
    plot_3_images([source_conv, ] + images[2][:-1],
                  ['Source', 'PolyCLEAN', 'CLEAN'],
                  suptitle="Comparison restored sharp (components + residual)",
                  normalize=True, offset_cm=0.05)

    # WS CLEAN
    plot_3_images([source, ] + images[0][::2],
                  ['Source', 'PolyCLEAN', 'WS-CLEAN'],
                  suptitle="Comparison components",
                  normalize=True, vlim=0.3, alpha=1)
    plot_3_images([source_conv, ] + images[1][::2],
                  ['Source', 'PolyCLEAN', 'WS-CLEAN'],
                  suptitle="Comparison components convolved sharp",
                  normalize=True)
    plot_3_images([source_conv, ] + images[2][::2],
                  ['Source', 'PolyCLEAN', 'WS-CLEAN'],
                  suptitle="Comparison restored sharp (components + residual)",
                  normalize=True, offset_cm=0.05)
