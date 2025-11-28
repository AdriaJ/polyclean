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
    with open(os.path.join(os.getcwd(), 'dirty.pkl'), 'rb') as file:
        dirty_im = pickle.load(file)
    with open(os.path.join(os.getcwd(), "reco_pkl", "0.02", "restored_sharp.pkl"), 'rb') as file:
        model = pickle.load(file)

    arr = certif.pixels.data[0, 0]
    cmaps = ['hot', 'Greys']
    vlim = 0.8
    alpha = 0.95
    offset_cm = 0.0

    ## Zoom in
    # Find the area to keep
    slicex = slice(100, 580)  # 480
    slicey = slice(170, 650)  # 480

    # # Plot the zoomed areas
    # arr2 = arr[slicex, slicey]
    # fig = plt.figure(figsize=(14, 10))
    # ax = fig.subplots(1, 1, subplot_kw={'projection': certif.image_acc.wcs.sub([1, 2]), 'frameon': False})
    # ax.set_xlabel(certif.image_acc.wcs.wcs.ctype[0])
    # ax.coords[1].set_ticklabel_visible(False)
    # ax.coords[1].set_axislabel('')
    # ax.set_ylabel('')
    # mask_comp = np.ma.masked_array(arr2, arr2 < vlim, fill_value=vlim)
    # mask_res = np.ma.masked_array(arr2, arr2 > vlim, fill_value=vlim)
    # cmapc = truncate_colormap(cmaps[0], offset_cm, 1.)
    # aximc = ax.imshow(mask_comp, origin="lower", cmap=cmapc, interpolation='none', alpha=alpha,  norm='linear', vmax=1.)
    # cmapr = truncate_colormap(cmaps[1], 0.45, 1 - offset_cm)
    # aximr = ax.imshow(mask_res, origin="lower", interpolation='none', alpha=alpha, cmap=cmapr,
    #                   norm='linear', vmin=arr.min(), vmax=vlim, )
    # axinsc = inset_axes(ax, width="3%", height="100%", loc='center left', borderpad=-3)
    # cbc = fig.colorbar(aximc, cax=axinsc, orientation="vertical", extend='max')
    # cbc.ax.yaxis.set_ticks_position('left')
    # axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center left', borderpad=-6)
    # cbr = fig.colorbar(aximr, cax=axinsr, orientation="vertical")
    # cbr.ax.yaxis.set_ticks_position('left')
    # fig.suptitle("Dual certificate image - maximum value: {:.3f}".format(arr.max()))
    # fig.show()

    ## Dirty image
    import matplotlib.colors as mplc
    vmax = dirty_im['pixels'].data.max()
    vmin = dirty_im['pixels'].data.min()
    vlim = 172.
    cmapp = truncate_colormap('hot', 0.05, 1.)
    cmapn = truncate_colormap('Greys', 0., 0.95)
    n = mplc.PowerNorm(gamma=0.5, vmin=vlim, vmax=vmax)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.subplots(1, 1, sharex=True, sharey=True,
                        subplot_kw={'projection': dirty_im.image_acc.wcs.sub([1, 2]), 'frameon': False})
    arr = np.real(dirty_im["pixels"].data[0, 0, :, :])
    arr2 = arr[slicex, slicey]
    im_pos = np.ma.masked_array(arr2, arr2 < vlim, fill_value=vlim)
    im_neg = np.ma.masked_array(arr2, arr2 > vlim, fill_value=vlim)
    aximn = ax.imshow(im_neg, origin="lower", cmap=cmapn, interpolation='none', alpha=alpha, vmin=-vlim, vmax=vlim)
    aximp = ax.imshow(im_pos, origin="lower", cmap=cmapp, interpolation='none', norm=n, alpha=alpha)
    ax.coords[1].set_ticklabel_visible(False)
    ax.coords[1].set_axislabel('')
    ax.set_ylabel('')
    ax.coords[0].set_ticklabel_visible(False)
    ax.coords[0].set_axislabel('')
    ax.set_xlabel('')
    plt.show()

    # Fake point sources
    vlim=500
    cmapp = truncate_colormap('hot', 0.05, 1.)
    cmapn = truncate_colormap('Greys', 0.30, 0.95)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.subplots(1, 1, sharex=True, sharey=True,
                        subplot_kw={'projection': model.image_acc.wcs.sub([1, 2]), 'frameon': False})
    arr = np.real(model["pixels"].data[0, 0, :, :])
    arr2 = arr[slicex, slicey]
    im_pos = np.ma.masked_array(arr2, arr2 < vlim, fill_value=vlim)
    im_neg = np.ma.masked_array(arr2, arr2 > vlim, fill_value=vlim)
    aximn = ax.imshow(im_neg, origin="lower", cmap=cmapn, interpolation='none', alpha=alpha, vmin=-vlim, vmax=vlim)
    aximp = ax.imshow(im_pos, origin="lower", cmap=cmapp, interpolation='none', norm=n, alpha=alpha)
    ax.coords[1].set_ticklabel_visible(False)
    ax.coords[1].set_axislabel('')
    ax.set_ylabel('')
    ax.coords[0].set_ticklabel_visible(False)
    ax.coords[0].set_axislabel('')
    ax.set_xlabel('')
    plt.show()

    # Reconstruction
    vlim = 172
    cmapp = truncate_colormap('hot', 0.05, 1.)
    cmapn = truncate_colormap('Greys', 0., 0.95)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.subplots(1, 1, sharex=True, sharey=True,
                        subplot_kw={'projection': restored.image_acc.wcs.sub([1, 2]), 'frameon': False})
    arr = np.real(restored["pixels"].data[0, 0, :, :])
    arr2 = arr[slicex, slicey]
    im_pos = np.ma.masked_array(arr2, arr2 < vlim, fill_value=vlim)
    im_neg = np.ma.masked_array(arr2, arr2 > vlim, fill_value=vlim)
    aximn = ax.imshow(im_neg, origin="lower", cmap=cmapn, interpolation='none', alpha=alpha, vmin=-vlim, vmax=vlim)
    aximp = ax.imshow(im_pos, origin="lower", cmap=cmapp, interpolation='none', norm=n, alpha=alpha)
    ax.coords[1].set_ticklabel_visible(False)
    ax.coords[1].set_axislabel('')
    ax.set_ylabel('')
    ax.coords[0].set_ticklabel_visible(False)
    ax.coords[0].set_axislabel('')
    ax.set_xlabel('')
    plt.show()

