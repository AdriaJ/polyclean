# load the 7 images from the pkl files in this directory
import pickle
import os

import matplotlib.colorbar
import numpy as np

import matplotlib
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


def plot_1_image(image, title="", cmaps=['hot', 'Greys'], alpha=.95, offset_cm=0., symm=True, ticks=None, vlim=None):
    if ticks is None:
        ticks = [1, 500, 1000, 2000, 3000, 4000]
    arr = image.pixels.data[0, 0]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.subplots(1, 1, subplot_kw={'projection': image.image_acc.wcs.sub([1, 2]), 'frameon': False})
    ax.set_xlabel(image.image_acc.wcs.wcs.ctype[0])
    ax.set_ylabel(image.image_acc.wcs.wcs.ctype[1])
    if vlim is None:
        vlim = -arr.min() if symm else 0.
    mask_comp = np.ma.masked_array(arr, arr < vlim, fill_value=vlim)
    mask_res = np.ma.masked_array(arr, arr > vlim, fill_value=vlim)
    cmapc = truncate_colormap(cmaps[0], offset_cm, 1.)
    aximc = ax.imshow(mask_comp, origin="lower", cmap=cmapc, interpolation='none', alpha=alpha,
                      norm=mplc.PowerNorm(gamma=0.5, vmin=vlim, vmax=1. * mask_comp.max()))
    cmapr = truncate_colormap(cmaps[1], 0., 1 - offset_cm)
    aximr = ax.imshow(mask_res, origin="lower", interpolation='none', alpha=alpha,
                      cmap=cmapr, norm='linear', vmin=-vlim, vmax=vlim)
    # norm=symm_sqrt_norm(-vlim, vlim))
    axinsc = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
    cbc = fig.colorbar(aximc, cax=axinsc,
                       orientation="vertical", ticks=[round(vlim)] + ticks)
    axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center right', borderpad=-6)
    cbr = fig.colorbar(aximr, cax=axinsr, orientation="vertical")
    fig.suptitle(title)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.0, right=0.93, hspace=0.15, wspace=0.15)
    fig.show()


if __name__ == "__main__":
    dir_path = os.path.join(os.getcwd(), 'reco_pkl')
    # files_dir = [f for f in os.listdir('.') if os.path.isdir(os.path.join('.', f))]
    files_dir = ['0.05', 'autothresh3', '0.02', 'autothresh2', '0.005', 'autothresh1']
    images = []
    for f in files_dir:
        with open(os.path.join(dir_path, 'restored.pkl'), 'rb') as file:
            images.append(pickle.load(file))
    with open(os.path.join(os.getcwd(), 'dirty.pkl'), 'rb') as file:
        dirty_im = pickle.load(file)

    plot_1_image(dirty_im, title="Dirty image")

    vmax = max([im.pixels.data.max() for im in images])
    vmin = min([im.pixels.data.min() for im in images])

    vlim = -vmin
    alpha = .95

    split = int(2 * vlim * 256 / (vmax + vlim))
    colors1 = plt.cm.hot(np.linspace(0.05**2, 1, 256 - split)**.5)
    colors2 = plt.cm.Greys(np.linspace(-0., 0.95, split))
    colors = np.vstack((colors2, colors1))
    mymap = mplc.LinearSegmentedColormap.from_list('my_colormap', colors)

    fig = plt.figure(figsize=(10, 17))
    axes = fig.subplots(3, 2, sharex=True, sharey=True,
                        subplot_kw={'projection': images[0].image_acc.wcs.sub([1, 2]), 'frameon': False})
    for i in range(len(files_dir)):
        ax = axes.ravel()[i]
        ims = ax.imshow(images[i].pixels.data[0,0], origin="lower", cmap=mymap, interpolation='none', alpha=alpha,
                        norm=mplc.PowerNorm(gamma=1., vmin=-vlim, vmax=vmax))
        # fig.colorbar()
    # ax = fig.add_subplot(311)
    cbar_ax = inset_axes(axes[0, 0], width="280%", height="10%", loc='upper left', borderpad=-5)
    # cax, _ = matplotlib.colorbar.make_axes(axes, location='top', orientation='horizontal')
    fig.colorbar(ims, cax=cbar_ax, orientation="horizontal", ticks=[-vlim, 0, vlim, 1000, 2000, 3000, 4000])
    plt.show()

    for im, f in zip(images, files_dir):
        plot_1_image(im, title=f)

    fig = plt.figure(figsize=(10, 17))
    cmapp = truncate_colormap('hot', 0.05, 1.)
    cmapn = truncate_colormap('Greys', 0., 0.95)
    n = mplc.PowerNorm(gamma=0.5, vmin=vlim, vmax=vmax)
    axes = fig.subplots(3, 2, sharex=True, sharey=True,
                        subplot_kw={'projection': images[0].image_acc.wcs.sub([1, 2]), 'frameon': False})
    for i in range(len(files_dir)):
        ax = axes.ravel()[i]
        arr = np.real(images[i]["pixels"].data[0, 0, :, :])
        im_pos = np.ma.masked_array(arr, arr < vlim, fill_value=vlim)
        im_neg = np.ma.masked_array(arr, arr > vlim, fill_value=vlim)
        aximn = ax.imshow(im_neg, origin="lower", cmap=cmapn, interpolation='none', alpha=alpha, vmin=-vlim, vmax=vlim)
        aximp = ax.imshow(im_pos, origin="lower", cmap=cmapp, interpolation='none', norm=n, alpha=alpha)
        ax.set_title(files_dir[i])
        if i in [1, 3, 5]:
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')
        else:
            ax.set_ylabel(images[0].image_acc.wcs.wcs.ctype[1])

        if i in [4, 5]:
            ax.set_xlabel(images[0].image_acc.wcs.wcs.ctype[0])
        else:
            ax.coords[0].set_ticklabel_visible(False)
            ax.coords[0].set_axislabel('')

    cbar_ax = inset_axes(axes[0, 0], width="250%", height="6%", loc='upper left', borderpad=-5)
    fig.colorbar(aximp, cax=cbar_ax, orientation="horizontal", ticks=[vlim, 1000, 2000, 3000, 4000])
    cbar_ax2 = inset_axes(cbar_ax, width="100%", height="100%", loc='upper center', borderpad=-4)
    fig.colorbar(aximn, cax=cbar_ax2, orientation="horizontal", ticks=[-vlim, -vlim/2, 0, vlim/2, vlim])
    plt.subplots_adjust(top=0.9, bottom=0.05, left=0.1, right=0.9, hspace=0.09, wspace=0.)
    plt.show()
