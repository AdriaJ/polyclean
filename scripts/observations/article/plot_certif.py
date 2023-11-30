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
