"""
Load the dual certificate and estimate the size of the uncertainty area by
inverting the PSD (Power Spectral Density).
Convolve the model image with the 'certificate' beam estimated that way.
"""
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mptchs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import scipy.signal as sps
import scipy.fft as sfft

from ska_sdp_func_python.image import fit_psf, restore_cube

from plot_reconstructions import plot_1_image, truncate_colormap

factor = 0.02
nantennas = 50

if __name__ == "__main__":
    # with open(os.path.join(os.getcwd(), "certif_pkl", "0.02", "certificate.pkl"), 'rb') as handle:
    #     certif = pickle.load(handle)

    with open(os.path.join(os.getcwd(), "reco_pkl", f"{nantennas}antennas", str(factor), "certif.pkl"), 'rb') as handle:
        certif = pickle.load(handle)

    certif_arr = certif.pixels.data[0, 0]
    cmaps = ['hot', 'Greys']
    vlim = 0.8

    alpha = 0.95
    offset_cm = 0.0

    arr = certif_arr

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
    # rect = mptchs.Rectangle((380, 70), 230, 180, fill=False, edgecolor='aquamarine', lw=3, ls='--')
    rect = mptchs.Rectangle((1140, 210), 690, 540, fill=False, edgecolor='aquamarine', lw=3, ls='--')
    ax.add_patch(rect)
    axinsc = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
    cbc = fig.colorbar(aximc, cax=axinsc, orientation="vertical", extend='max')
    axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center right', borderpad=-6)
    cbr = fig.colorbar(aximr, cax=axinsr, orientation="vertical")
    fig.suptitle("Dual certificate image - maximum value: {:.3f}".format(arr.max()))
    fig.show()

    certif_arr[certif_arr < 0.9] = 0
    psd_beam = sfft.ifftshift(sfft.ifft2(np.abs(sfft.fft2(certif_arr))**2)).real
    # print(np.allclose(psd_beam.imag, 0))  # True
    psd_beam /= psd_beam.max()
    # psd_beam[psd_beam < 0.1] = 0

    vlim = min(-psd_beam.min(), 0)
    arr = psd_beam

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
    # rect = mptchs.Rectangle((380, 70), 230, 180, fill=False, edgecolor='aquamarine', lw=3, ls='--')
    rect = mptchs.Rectangle((1140, 210), 690, 540, fill=False, edgecolor='aquamarine', lw=3, ls='--')
    ax.add_patch(rect)
    axinsc = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
    cbc = fig.colorbar(aximc, cax=axinsc, orientation="vertical", extend='max')
    axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center right', borderpad=-6)
    cbr = fig.colorbar(aximr, cax=axinsr, orientation="vertical")
    fig.suptitle("Dual certificate image - maximum value: {:.3f}".format(arr.max()))
    fig.show()

    plt.figure()
    plt.hist(psd_beam.flatten(), bins=100)
    plt.yscale('log')
    plt.show()

    beam_image = certif.copy(deep=True)
    beam_image.pixels.data[0, 0] = psd_beam
    beam = fit_psf(beam_image)

    # with open(os.path.join(os.getcwd(), "reco_pkl", "0.02", "comp_restored.pkl"), 'rb') as handle:
    #     restored = pickle.load(handle)
    # with open(os.path.join(os.getcwd(), "reco_pkl", "0.02", "model.pkl"), 'rb') as handle:
    #     model = pickle.load(handle)

    with open(os.path.join(os.getcwd(), "reco_pkl", f"{nantennas}antennas", str(factor), "restored.pkl"), 'rb') as handle:
        restored = pickle.load(handle)
    with open(os.path.join(os.getcwd(), "reco_pkl", f"{nantennas}antennas", str(factor), "model.pkl"), 'rb') as handle:
        model = pickle.load(handle)


    model_arr = model.pixels.data[0, 0]

    # Convolve the model image with the 'certificate' beam
    certif_restored_arr = sps.fftconvolve(model_arr, psd_beam, mode='same')
    certif_restored = restored.copy(deep=True)
    certif_restored.pixels.data[0, 0] = certif_restored_arr

    psf_certif_restored = restore_cube(model, None, None, clean_beam=beam)

    plot_1_image(certif_restored, title="Restored image with certificate beam", cmaps=['hot', 'Greys'], alpha=.95,
                    offset_cm=0., symm=True, ticks=None, vlim=172)
    plot_1_image(restored, title="Restored image", cmaps=['hot', 'Greys'], alpha=.95,
                    offset_cm=0., symm=True, ticks=None, vlim=172)
    plot_1_image(psf_certif_restored, title="Restored image with certificate beam fitted with RASCIL", cmaps=['hot', 'Greys'], alpha=.95,
                    offset_cm=0., symm=True, ticks=None, vlim=172)


    # plot the zoomed area
    import matplotlib.colors as mplc

    # slicex = slice(70, 250)  # 180
    # slicey = slice(380, 610)  # 230

    slicex = slice(210, 750)
    slicey = slice(1140, 1830)

    image = psf_certif_restored

    # vmax, vmin = 4881.83, -25.39
    vmax, vmin = 4892.68, -4.33
    vlim = -vmin
    cmapp = truncate_colormap('hot', 0.05, 1.)
    cmapn = truncate_colormap('Greys', 0., 0.95)
    n = mplc.PowerNorm(gamma=0.5, vmin=vlim, vmax=vmax)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.subplots(1, 1, sharex=True, sharey=True,
                        subplot_kw={'projection': image.image_acc.wcs.sub([1, 2]), 'frameon': False})
    arr = np.real(image["pixels"].data[0, 0, :, :])
    arr2 = arr[slicex, slicey]
    im_pos = np.ma.masked_array(arr2, arr2 < vlim, fill_value=vlim)
    im_neg = np.ma.masked_array(arr2, arr2 > vlim, fill_value=vlim)
    aximn = ax.imshow(im_neg, origin="lower", cmap=cmapn, interpolation='none', alpha=alpha, vmin=-vlim, vmax=vlim)
    aximp = ax.imshow(im_pos, origin="lower", cmap=cmapp, interpolation='none', norm=n, alpha=alpha)
    # ax.set_ylabel(image.image_acc.wcs.wcs.ctype[1])
    ax.coords[1].set_ticklabel_visible(False)
    ax.coords[1].set_axislabel('')
    ax.set_ylabel('')
    ax.set_xlabel(image.image_acc.wcs.wcs.ctype[0])
    axinsc = inset_axes(ax, width="3%", height="100%", loc='center left', borderpad=-3)
    cbc = fig.colorbar(aximp, cax=axinsc, orientation="vertical", extend='max')
    cbc.ax.yaxis.set_ticks_position('left')
    axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center left', borderpad=-6)
    cbr = fig.colorbar(aximn, cax=axinsr, orientation="vertical")
    cbr.ax.yaxis.set_ticks_position('left')

    plt.show()
