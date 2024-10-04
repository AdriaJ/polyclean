import os
import pickle
import numpy as np
import pandas as pd
from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astropy.coordinates import SkyCoord
from astropy import units as u
from ska_sdp_func_python.sky_component import insert_skycomponent
from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_func_python.image import restore_cube, fit_psf




import polyclean.image_utils as ut
from ska_sdp_func_python.imaging import invert_visibility, create_image_from_visibility


peak_thresh = 10

npixel = 1024
fov_deg = 6.
context = "ng"

mode = "total"  # "peak"

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
    catalogpath = "/home/jarret/Documents/EPFL/PhD/ra_data/LOFAR150_BOOTES.fits"
    with fits.open(catalogpath) as hdul:
        hdul.info()
        c = hdul[1].columns
        if mode=="total":
            col_names = ['RA', 'DEC', 'Total_flux']  # Peak_flux
        elif mode=="peak":
            col_names = ['RA', 'DEC', 'Peak_flux']
        src_ra_dec_flux = np.vstack([hdul[1].data[n] for n in col_names])  # degrees
        # src_ra_dec_pflux = np.vstack([hdul[1].data[n] for n in ['RA', 'DEC', 'Peak_flux']])

    # plt.figure()
    # plt.scatter(src_ra_dec_pflux[-1], src_ra_dec_flux[-1], marker='x')
    # m = max(src_ra_dec_pflux[-1].max(), src_ra_dec_flux[-1].max())
    # plt.plot([0, m], [0, m], color='r', ls='--')
    # plt.xlabel("Peak flux [Jy]")
    # plt.ylabel("Total flux [Jy]")
    # plt.show()

    print("Select source from the catalog with peak flux higher than {:d} Jy:".format(peak_thresh))
    src_ra_dec_flux = src_ra_dec_flux[:, src_ra_dec_flux[-1] > peak_thresh]
    print("\t{:d} sources selected in the catalogue.".format(src_ra_dec_flux.shape[1]))

    # create dirty image from the measurements, to get an image model
    data_dir = "/home/jarret/PycharmProjects/polyclean/scripts/observations/article"
    with open(os.path.join(data_dir, "vis", "vis.pkl"), 'rb') as handle:
        vis = pickle.load(handle)
    print("Selected vis: ", vis.dims)

    phasecentre = vis.phasecentre
    freq = vis.frequency.data[0]
    channel_bandwidth = vis.channel_bandwidth.data[0]
    fov = fov_deg * np.pi / 180.
    cellsize = fov / npixel

    ## image model
    image_model = create_image_from_visibility(vis, npixel=npixel, cellsize=cellsize, override_cellsize=False)
    image_model = ut.image_add_ra_dec_grid(image_model)


    ## dirty image with RASCIL
    # directions = SkyCoord(
    #     ra=image_model.ra_grid.data.ravel() * u.rad,
    #     dec=image_model.dec_grid.data.ravel() * u.rad,
    #     frame="icrs", equinox="J2000", )
    # direction_cosines = np.stack(skycoord_to_lmn(directions, phasecentre), axis=-1)
    # uvwlambda = vis.visibility_acc.uvw_lambda.reshape(-1, 3)
    # flags_bool = (vis.weight.data != 0.).reshape(-1)
    # flagged_uvwlambda = uvwlambda[flags_bool]
    #
    # dirty, _ = invert_visibility(vis, image_model, context=context, dopsf=False, normalise=False)

    # Load sky components
    sc=[]
    for ra_dec_flux in src_ra_dec_flux.T:
        ra, dec, flux = ra_dec_flux
        # ra *= np.pi / 180
        # dec *= np.pi / 180
        sc_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs",) # equinox="J2000")
        sc.append(SkyComponent(sc_coord, flux=np.r_[flux].reshape(1, 1), frequency=np.r_[freq],
                          polarisation_frame=PolarisationFrame("stokesI")))
        # print(ra, dec, flux)

    gt_image = image_model.copy(deep=True)
    insert_skycomponent(gt_image, sc, insert_method='Nearest')

    # Convolve with dirty beam
    psf, _ = invert_visibility(vis, image_model, context=context, dopsf=True)
    clean_beam = fit_psf(psf)
    sharp_beam = clean_beam.copy()
    sharp_beam["bmin"] = clean_beam["bmin"] / 2
    sharp_beam["bmaj"] = clean_beam["bmaj"] / 2

    gt_conv = restore_cube(gt_image, None, None, clean_beam=clean_beam)
    gt_sharp = restore_cube(gt_image, None, None, clean_beam=sharp_beam)

    # Show the images
    plot_1_image(gt_conv, title="GT image convolved", offset_cm=0.05, vlim=172.1)
    plot_1_image(gt_sharp, title="GT image sharp", offset_cm=0.05, vlim=172.1)

    # save the images as pkl
    pkl_dir = "/home/jarret/PycharmProjects/polyclean/scripts/observations/article/reco_pkl"
    if not os.path.exists(os.path.join(pkl_dir, 'gt')):
        os.makedirs(os.path.join(pkl_dir, 'gt'))
    with open(os.path.join(pkl_dir, 'gt', 'gt_' + mode + '_cb' + '.pkl'), 'wb') as file:
        pickle.dump(gt_conv, file)
    with open(os.path.join(pkl_dir, 'gt', 'gt_' + mode + '_sharp' + '.pkl'), 'wb') as file:
        pickle.dump(gt_sharp, file)



    # plot pclean reconstruction
    with open(os.path.join(pkl_dir, '0.02', 'comp_restored' + '.pkl'), 'rb') as file:
        pclean_im = pickle.load(file)
    plot_1_image(pclean_im, title="PClean reconstruction", offset_cm=0.05, vlim=172.1)
