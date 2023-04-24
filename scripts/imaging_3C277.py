import logging
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import polyclean.image_utils as ut
import polyclean.polyclean as pc
import polyclean.reconstructions as reco
import pycsou.util.complex as pycuc

from astropy.coordinates import SkyCoord
from astropy import units as u
from rascil.processing_components.visibility import create_visibility_from_ms, list_ms
from ska_sdp_func_python.imaging import (
    predict_visibility,
    invert_visibility,
    create_image_from_visibility,
)
from ska_sdp_func_python.image import (
    deconvolve_cube,
    restore_cube,
    fit_psf,
)
from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame

matplotlib.use("Qt5Agg")

context = "ng"
filename2 = "3C277.1C.16channels.ms"  # 2498 times, 21 baselines, 16 channels, 1 polarization "I"
algorithm = 'hogbom'
npixel = 1024
nufft_eps = 1e-5

niter = 100

lambda_factor = 0.15
eps = 1e-5
tmax = 240.
min_iter = 5
ms_threshold = 0.9
init_correction_prec = 5e-2
final_correction_prec = min(1e-5, eps)
remove = True
min_correction_steps = 5
lock = False
diagnostics = True
log_diagnostics = False

if __name__ == "__main__":
    log = logging.getLogger("rascil-logger")
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))

    datapath = "/home/jarret/PycharmProjects/polyclean/src/rascil-main/data/vis/"
    filename1 = "ASKAP_example.ms"  # 19 times, 55 baselines, 192 channels, 4 polarization "XX"  "XY" "YX" "YY"
    msname = datapath + filename2
    print("Source and data descriptors in the measurement set: " + str(list_ms(msname)))

    bvis_list = create_visibility_from_ms(msname, datacolumn="DATA", channum=slice(6, 7))
    vis = bvis_list[0]

    print(vis.dims)

    selected_vis = vis.isel(
        {"time": slice(0, vis.dims["time"], 50),  # slice(vis.dims["time"]//2, vis.dims["time"]//2 + 1)
         # "frequency": slice(vis.dims["frequency"] // 2, vis.dims["frequency"] // 2 + 1),
         })
    phasecentre = selected_vis.phasecentre
    image_model = create_image_from_visibility(selected_vis, npixel=npixel)
    cellsize = abs(image_model.coords["x"].data[1] - image_model.coords["x"].data[0])
    fov = npixel * cellsize
    fov_deg = fov * 180.0 / np.pi

    ## CLEAN reconstruction

    clean_model = create_image_from_visibility(selected_vis, npixel=2 * npixel)
    large_dirty, sumwt_dirty = invert_visibility(selected_vis, clean_model, context=context)
    psf, sumwt = invert_visibility(selected_vis, image_model, context=context, dopsf=True)
    # ut.plot_image(dirty, title="Dirty image")
    dirty = image_model.copy(deep=True)
    dirty['pixels'].data[0, 0, ...] = \
        large_dirty['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]

    print("CLEAN: ...")
    start = time.time()
    tmp_clean_comp, tmp_clean_residual = deconvolve_cube(
        large_dirty,
        psf,
        niter=niter,
        threshold=0.001,
        fractional_threshold=0.001,
        window_shape="quarter",
        gain=0.7,
        algorithm=algorithm,
    )
    clean_comp = image_model.copy(deep=True)
    clean_comp['pixels'].data[0, 0, ...] = \
        tmp_clean_comp['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]
    clean_residual = image_model.copy(deep=True)
    clean_residual['pixels'].data[0, 0, ...] = \
        tmp_clean_residual['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]
    dt_clean = time.time() - start
    print("\nSolved in {:.3f}".format(dt_clean))

    clean_comp_restored = restore_cube(clean_comp, psf, None)
    # ut.plot_image(clean_comp_restored, title="CLEAN reco")
    clean_restored = restore_cube(clean_comp, psf, clean_residual)
    # ut.plot_image(clean_restored, title="CLEAN reco + residual")

    ## PolyCLEAN reconstruction

    image_model = ut.image_add_ra_dec_grid(image_model)
    directions = SkyCoord(
        ra=image_model.ra_grid.data.ravel() * u.rad,
        dec=image_model.dec_grid.data.ravel() * u.rad,
        frame="icrs",
        equinox="J2000",
    )
    direction_cosines = np.stack(skycoord_to_lmn(directions, phasecentre), axis=-1)
    uvwlambda = selected_vis.visibility_acc.uvw_lambda.reshape(-1, 3)
    flags_bool = np.any(uvwlambda != 0., axis=-1)
    flagged_uvwlambda = uvwlambda[flags_bool]

    forwardOp = pc.generatorVisOp(direction_cosines=direction_cosines,
                                  vlambda=flagged_uvwlambda,
                                  nufft_eps=nufft_eps,
                                  chunked=False)
    start = time.time()
    fOp_lipschitz = forwardOp.lipschitz(tol=1., tight=True)
    lipschitz_time = time.time() - start
    print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)\n".format(lipschitz_time))

    vis_array = pycuc.view_as_real(selected_vis.vis.data.reshape(-1)[flags_bool])

    dirty_hvox_arr = forwardOp.adjoint(vis_array)
    lambda_ = lambda_factor * np.abs(dirty_hvox_arr).max()

    # error_dirty = dirty.copy(deep=True)
    # error_dirty.pixels.data = dirty_hvox_arr.reshape(dirty.pixels.data.shape)/sumwt_dirty - dirty.pixels.data
    # ut.display_image(error_dirty, cmap="bwr")
    #
    # np.abs(dirty.pixels.data.reshape(-1) - dirty_hvox_arr/sumwt_dirty[0]).max()

    stop_crit = reco.stop_crit(tmax, min_iter, eps)
    pclean_parameters = {
        # "lambda_factor": lambda_factor,
        "ms_threshold": ms_threshold,
        "init_correction_prec": init_correction_prec,
        "final_correction_prec": final_correction_prec,
        "min_correction_steps": min_correction_steps,
        "remove_positions": remove,
        "nufft_eps": nufft_eps,
        "show_progress": False,
    }
    fit_parameters = {
        "stop_crit": stop_crit,
        "positivity_constraint": True,
        "diff_lipschitz": fOp_lipschitz ** 2,
        "lock_reweighting": lock
    }

    # Computations
    pclean = pc.PolyCLEAN(
        flagged_uvwlambda,
        direction_cosines,
        vis_array,
        lambda_=lambda_,
        **pclean_parameters,
    )
    print("PolyCLEAN: Solving...")
    pclean_time = time.time()
    pclean.fit(**fit_parameters)
    print("\tSolved in {:.3f} seconds".format(time.time() - pclean_time))
    if diagnostics:
        pclean.diagnostics(log=log_diagnostics)
    data, hist = pclean.stats()
    pclean_residual = forwardOp.adjoint(vis_array - forwardOp(data["x"]))

    print("PolyCLEAN final DCV: {:.3f}".format(data["dcv"]))
    print("Iterations: {}".format(int(hist['N_iter'][-1])))
    print("Final sparsity: {}".format(np.count_nonzero(data["x"])))

    pclean_comp = image_model.copy(deep=True)
    pclean_comp.pixels.data[0, 0] = data["x"].reshape((npixel,) * 2)

    pclean_comp_restored = restore_cube(pclean_comp, psf, None)
    # ut.plot_image(pclean_comp_restored, title="PolyCLEAN Reco", cmap="cubehelix_r")

    pclean_residual_im = image_model.copy(deep=True)
    pclean_residual_im.pixels.data = pclean_residual.reshape(pclean_residual_im.pixels.data.shape) / sumwt_dirty[0, 0]
    pclean_restored_plus_res = restore_cube(pclean_comp, psf, pclean_residual_im)
    # ut.plot_image(pclean_restored_plus_res, title="PolyCLEAN Reco + residual", cmap="cubehelix_r")

    pclean_restored = restore_cube(pclean_comp, psf, pclean_residual_im)

    dual_certificate_im = image_model.copy(deep=True)
    dual_certificate_im.pixels.data = pclean_residual.reshape(dual_certificate_im.pixels.data.shape) / lambda_
    # ut.plot_certificate(dual_certificate_im)

    # plot comparative solutions
    suptitle = "Reconstruction comparison"
    cm = "cubehelix_r"
    chan, pol = 0, 0
    vmin = 0.

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig = plt.figure(figsize=(26, 12))
    axes = fig.subplots(2, 4, sharex=True, sharey=True, subplot_kw={'projection': dirty.image_acc.wcs.sub([1, 2]),
                                                                    'frameon': False})

    dirty_array = np.real(dirty["pixels"].data[chan, pol, :, :]) * sumwt_dirty
    ax = axes[0, 0]
    ims = ax.imshow(dirty_array, origin="lower", cmap=cm, interpolation="none")
    ax.set_ylabel(dirty.image_acc.wcs.wcs.ctype[1])
    ax.coords[0].set_ticklabel_visible(False)
    ax.coords[0].set_axislabel('')
    ax.set_title("Dirty image")
    axins = inset_axes(ax, width="100%", height="4%", loc='lower center', borderpad=-5)
    fig.colorbar(ims, cax=axins, orientation="horizontal")

    dual_certif_arr = np.real(dual_certificate_im["pixels"].data[chan, pol, :, :])
    ax = axes[1, 0]
    ims = ax.imshow(dual_certif_arr, origin="lower", cmap=cm, interpolation="none")
    ax.set_ylabel(dirty.image_acc.wcs.wcs.ctype[1])
    ax.set_xlabel(dirty.image_acc.wcs.wcs.ctype[0])
    ax.contour(dual_certif_arr, levels=[1.], colors="c")
    ax.set_title("Dual certificate image")
    axins = inset_axes(ax, width="100%", height="4%", loc='lower center', borderpad=-5)
    fig.colorbar(ims, cax=axins, orientation="horizontal")

    upper_row = [tmp_clean_comp, clean_comp_restored, clean_restored]
    upper_row_title = ["CLEAN components", "Components convolved", "+ Residual"]
    upper_row_indices = [(0, 1), (0, 2), (0, 3)]
    lower_row = [pclean_comp, pclean_comp_restored, pclean_restored]
    lower_row_title = ["PolyCLEAN components", "Components convolved", "+ Residual"]
    lower_row_indices = [(1, 1), (1, 2), (1, 3)]
    vmaxs = [max(up.pixels.data.max(), low.pixels.data.max()) for (up, low) in zip(upper_row, lower_row)]

    for idx, im in enumerate(upper_row):
        ax = axes[upper_row_indices[idx]]
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[0].set_axislabel('')
        ax.coords[1].set_ticklabel_visible(False)
        ax.coords[1].set_axislabel('')
        im_array = np.real(im["pixels"].data[chan, pol, :, :])
        ims = ax.imshow(im_array, origin="lower", cmap=cm, vmin=vmin, vmax=vmaxs[idx], interpolation="none")
        ax.set_title(upper_row_title[idx])
        axins = inset_axes(ax, width="100%", height="4%", loc='lower center', borderpad=-5)
        fig.colorbar(ims, cax=axins, orientation="horizontal")

    for idx, im in enumerate(lower_row):
        ax = axes[lower_row_indices[idx]]
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[0].set_axislabel('')
        ax.coords[1].set_ticklabel_visible(False)
        ax.coords[1].set_axislabel('')
        im_array = np.real(im["pixels"].data[chan, pol, :, :])
        ims = ax.imshow(im_array, origin="lower", cmap=cm, vmin=vmin, vmax=vmaxs[idx], interpolation="none")
        ax.set_title(lower_row_title[idx])
        # fig.colorbar(ims, ax=ax, orientation="vertical", shrink=0.7)

    fig.suptitle(suptitle)
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.03, right=0.97, hspace=0.25, wspace=0.08)
    plt.show()
