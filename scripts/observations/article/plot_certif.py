"""
Load the dual certificate and the reconstructed image and plot them with the appropriate colormaps.
"""
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plot_reconstructions import plot_1_image, truncate_colormap

if __name__ == "__main__":

    with open(os.path.join(os.getcwd(), "certif_pkl", "0.02", "certificate.pkl"), 'rb') as handle:
        certif = pickle.load(handle)
    with open(os.path.join(os.getcwd(), "certif_pkl", "0.02", "restored.pkl"), 'rb') as handle:
        restored = pickle.load(handle)

    plot_1_image(restored, title="Restored image", cmaps=['hot', 'Greys'], alpha=.95,
                 offset_cm=0., symm=True, ticks=None, vlim=172)

    arr = certif.pixels.data[0, 0]
    cmaps = ['hot', 'Greys']
    vlim = 0.8

    alpha = 0.95
    offset_cm = 0.0

    fig = plt.figure(figsize=(12, 10))
    ax = fig.subplots(1, 1, subplot_kw={'projection': certif.image_acc.wcs.sub([1, 2]), 'frameon': False})
    ax.set_xlabel(certif.image_acc.wcs.wcs.ctype[0])
    ax.set_ylabel(certif.image_acc.wcs.wcs.ctype[1])
    mask_comp = np.ma.masked_array(arr, arr < vlim, fill_value=vlim)
    mask_res = np.ma.masked_array(arr, arr > vlim, fill_value=vlim)
    cmapc = truncate_colormap(cmaps[0], offset_cm, 1.)
    aximc = ax.imshow(mask_comp, origin="lower", cmap=cmapc, interpolation='none', alpha=alpha,  norm='linear', vmax=1.)
    cmapr = truncate_colormap(cmaps[1], 0., 1 - offset_cm)
    aximr = ax.imshow(mask_res, origin="lower", interpolation='none', alpha=alpha, cmap=cmapr,
                      norm='linear', vmin=arr.min(), vmax=vlim, )
    axinsc = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
    cbc = fig.colorbar(aximc, cax=axinsc, orientation="vertical", extend='max')
    axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center right', borderpad=-6)
    cbr = fig.colorbar(aximr, cax=axinsr, orientation="vertical")
    fig.suptitle("Dual certificate image - maximum value: {:.3f}".format(arr.max()))
    fig.show()

    ## Zoom in
    # Find the area to keep
    slicex = slice(70, 250)  # 180
    slicey = slice(380, 610)  # 230

    # Plot the zoomed areas
    arr2 = arr[slicex, slicey]
    fig = plt.figure(figsize=(14, 10))
    ax = fig.subplots(1, 1, subplot_kw={'projection': certif.image_acc.wcs.sub([1, 2]), 'frameon': False})
    ax.set_xlabel(certif.image_acc.wcs.wcs.ctype[0])
    ax.coords[1].set_ticklabel_visible(False)
    ax.coords[1].set_axislabel('')
    ax.set_ylabel('')
    mask_comp = np.ma.masked_array(arr2, arr2 < vlim, fill_value=vlim)
    mask_res = np.ma.masked_array(arr2, arr2 > vlim, fill_value=vlim)
    cmapc = truncate_colormap(cmaps[0], offset_cm, 1.)
    aximc = ax.imshow(mask_comp, origin="lower", cmap=cmapc, interpolation='none', alpha=alpha,  norm='linear', vmax=1.)
    cmapr = truncate_colormap(cmaps[1], 0., 1 - offset_cm)
    aximr = ax.imshow(mask_res, origin="lower", interpolation='none', alpha=alpha, cmap=cmapr,
                      norm='linear', vmin=arr.min(), vmax=vlim, )
    axinsc = inset_axes(ax, width="3%", height="100%", loc='center left', borderpad=-3)
    cbc = fig.colorbar(aximc, cax=axinsc, orientation="vertical", extend='max')
    cbc.ax.yaxis.set_ticks_position('left')
    axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center left', borderpad=-6)
    cbr = fig.colorbar(aximr, cax=axinsr, orientation="vertical")
    cbr.ax.yaxis.set_ticks_position('left')
    fig.suptitle("Dual certificate image - maximum value: {:.3f}".format(arr.max()))
    fig.show()

    import matplotlib.colors as mplc
    files_dir = ['0.02', 'autothresh2']
    images = []
    for f in files_dir:
        with open(os.path.join(os.getcwd(), 'reco_pkl', f, 'comp_restored.pkl'), 'rb') as file:
            images.append(pickle.load(file))

    vmax = max([im.pixels.data.max() for im in images])
    vmin = min([im.pixels.data.min() for im in images])

    vlim = -vmin
    alpha = .95

    split = int(2 * vlim * 256 / (vmax + vlim))
    colors1 = plt.cm.hot(np.linspace(0.05 ** 2, 1, 256 - split) ** .5)
    colors2 = plt.cm.Greys(np.linspace(-0., 0.95, split))
    colors = np.vstack((colors2, colors1))
    mymap = mplc.LinearSegmentedColormap.from_list('my_colormap', colors)

    cmapp = truncate_colormap('hot', 0.05, 1.)
    cmapn = truncate_colormap('Greys', 0., 0.95)
    n = mplc.PowerNorm(gamma=0.5, vmin=vlim, vmax=vmax)

    for i in range(len(files_dir)):
        fig = plt.figure(figsize=(14, 10))
        ax = fig.subplots(1, 1, sharex=True, sharey=True,
                            subplot_kw={'projection': images[0].image_acc.wcs.sub([1, 2]), 'frameon': False})
        arr = np.real(images[i]["pixels"].data[0, 0, :, :])
        arr2 = arr[slicex, slicey]
        im_pos = np.ma.masked_array(arr2, arr2 < vlim, fill_value=vlim)
        im_neg = np.ma.masked_array(arr2, arr2 > vlim, fill_value=vlim)
        aximn = ax.imshow(im_neg, origin="lower", cmap=cmapn, interpolation='none', alpha=alpha, vmin=-vlim, vmax=vlim)
        aximp = ax.imshow(im_pos, origin="lower", cmap=cmapp, interpolation='none', norm=n, alpha=alpha)
        if i == 1:
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')
            ax.set_ylabel('')
            ax.set_xlabel(images[i].image_acc.wcs.wcs.ctype[0])
            axinsc = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
            cbc = fig.colorbar(aximp, cax=axinsc, orientation="vertical", extend='max')
            axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center right', borderpad=-6)
            cbr = fig.colorbar(aximn, cax=axinsr, orientation="vertical")
        else:
            ax.set_ylabel(images[i].image_acc.wcs.wcs.ctype[1])
            ax.set_xlabel(images[i].image_acc.wcs.wcs.ctype[0])
        fig.suptitle(files_dir[i])
        plt.show()

    fig = plt.figure(figsize=(14, 10))
    ax = fig.subplots(1, 1, sharex=True, sharey=True,
                        subplot_kw={'projection': images[0].image_acc.wcs.sub([1, 2]), 'frameon': False})
    arr = np.real(images[0]["pixels"].data[0, 0, :, :])
    arr2 = arr[slicex, slicey]
    im_pos = np.ma.masked_array(arr2, arr2 < vlim, fill_value=vlim)
    im_neg = np.ma.masked_array(arr2, arr2 > vlim, fill_value=vlim)
    aximn = ax.imshow(im_neg, origin="lower", cmap=cmapn, interpolation='none', alpha=alpha, vmin=-vlim, vmax=vlim)
    aximp = ax.imshow(im_pos, origin="lower", cmap=cmapp, interpolation='none', norm=n, alpha=alpha)
    ax.set_ylabel(images[0].image_acc.wcs.wcs.ctype[1])
    ax.set_xlabel(images[0].image_acc.wcs.wcs.ctype[0])
    axinsc = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
    cbc = fig.colorbar(aximp, cax=axinsc, orientation="vertical", extend='max')
    axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center right', borderpad=-6)
    cbr = fig.colorbar(aximn, cax=axinsr, orientation="vertical")
    fig.suptitle(files_dir[0])
    plt.show()
