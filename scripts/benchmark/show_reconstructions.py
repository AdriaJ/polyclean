# Load reconstructed images from the fits files and plot them.
# These images come from the benchmark of section 6.1 and are displayed in the Appendix 1.
import pickle
import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from rascil.processing_components.image.operations import import_image_from_fits


def truncate_colormap(cmap, minval, maxval, n=100):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mplc.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def save_1_image(image, title="", savedir=None, show=False, cmaps=['hot', 'Greys'], alpha=.95, offset_cm=0., symm=True,
                 ticks=None, vlim=None):
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
    if savedir is None:
        savedir = os.getcwd()
    plt.savefig(os.path.join(savedir, title + '.png'))
    if show:
        fig.show()


srf = 10

dir_path = '/home/jarret/Downloads/res/srf' + str(srf) + 'reps1'

if __name__ == "__main__":
    names = ['rmax_' + str(r) for r in [2000, 3000, 6000]]

    # load image from fits file
    images = []
    for name in names:
        images.append(import_image_from_fits(os.path.join(dir_path, 'reconstructions', name, 'pclean_restored.fits')))

    # # save images
    # savedir = os.path.join('/home/jarret/Downloads/', 'srf' + str(srf))
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    # for im, name in zip(images, names):
    #     save_1_image(im, title=name, vlim=0.02, savedir=savedir, show=False)

    # Plot the image row-wise, 3 by 3
    rmaxs = [2000, 3000, 6000]
    srfs = [2, 5, 10]
    vlim = 0.02
    cmaps = ['hot', 'Greys']
    offset_cm = 0.05
    alpha = 0.95
    ticks = [0.1, 0.3, 0.6, 1.]
    ticks_bg = [-0.02, -0.01, 0, 0.01, 0.02]
    cmapc = truncate_colormap(cmaps[0], offset_cm, 1.)
    cmapr = truncate_colormap(cmaps[1], 0., 1 - offset_cm)

    for rmax in rmaxs:
        savename = os.path.join('/home/jarret/Downloads/', 'rmax' + str(rmax))
        images = []
        for srf in srfs:
            dirname = '/home/jarret/Downloads/res/srf' + str(srf) + 'reps1'
            images.append(import_image_from_fits(
                os.path.join(dirname, 'reconstructions', 'rmax_' + str(rmax), 'pclean_restored.fits')))

        # plot the 3 figures side by side
        fig = plt.figure(figsize=(13, 4))
        # axes = fig.subplots(1, 3, subplot_kw={'frameon': False})  # 'projection': images[0].image_acc.wcs.sub([1, 2])
        for i in range(len(images)):
            ax = fig.add_subplot(1, len(images), i + 1, projection=images[i].image_acc.wcs.sub([1, 2]))
            image = images[i]
            arr = image.pixels.data[0, 0]
            if i == 0:
                ax.set_ylabel(images[i].image_acc.wcs.wcs.ctype[1])
            else:
                ax.coords[1].set_ticklabel_visible(False)
                ax.coords[1].set_axislabel('')
                ax.set_ylabel('')
            ax.set_xlabel(image.image_acc.wcs.wcs.ctype[0])
            mask_comp = np.ma.masked_array(arr, arr < vlim, fill_value=vlim)
            mask_res = np.ma.masked_array(arr, arr > vlim, fill_value=vlim)
            aximc = ax.imshow(mask_comp, origin="lower", cmap=cmapc, interpolation='none', alpha=alpha,
                              norm=mplc.PowerNorm(gamma=0.5, vmin=vlim, vmax=1. * mask_comp.max()))
            aximr = ax.imshow(mask_res, origin="lower", interpolation='none', alpha=alpha,
                              cmap=cmapr, norm='linear', vmin=-vlim, vmax=vlim)
            if i == 2:
                axinsc = inset_axes(ax, width="4%", height="100%", loc='center right', borderpad=-3)
                cbc = fig.colorbar(aximc, cax=axinsc,
                                   orientation="vertical", ticks=[round(vlim)] + ticks)
                axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center right', borderpad=-4)
                cbr = fig.colorbar(aximr, cax=axinsr, orientation="vertical", ticks=ticks_bg)
        plt.subplots_adjust(top=0.92, bottom=0.12, left=0.05, right=0.88, hspace=0.1, wspace=0.15)
        plt.savefig(savename + '.png')
        # plt.show()


