import logging
import sys
import pickle
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import time
import polyclean.image_utils as ut
import polyclean.polyclean as pc
import polyclean.reconstructions as reco
import pyxu.util.complex as pxc

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs.utils import skycoord_to_pixel

from ska_sdp_func_python.imaging import invert_visibility, create_image_from_visibility
from ska_sdp_func_python.image import restore_cube, fit_psf
from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.visibility.vis_utils import generate_baselines
from ska_sdp_datamodels.sky_model import SkyComponent

nantennas = 28
ntimes = 50
npixel = 1024
fov_deg = 6.
context = "ng"

nufft_eps = 1e-3
lambda_factor = 0.005
eps = 1e-4
tmax = 120. * 2
min_iter = 5
ms_threshold = 0.9
init_correction_prec = 1e-2
final_correction_prec = min(1e-5, eps)
remove = True
min_correction_steps = 5
max_correction_steps = 1000
lock = False
diagnostics = True
log_diagnostics = False

save = False
save_unified = False
save_im_pkl = True

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


def plot_certificate(certificate_image, title="Dual certificate", alpha=0.95, offset_cm=.2):
    arr = certificate_image.pixels.data[0, 0]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.subplots(1, 1, subplot_kw={'projection': image_model.image_acc.wcs.sub([1, 2]), 'frameon': False})
    ax.set_xlabel(image_model.image_acc.wcs.wcs.ctype[0])
    ax.set_ylabel(image_model.image_acc.wcs.wcs.ctype[1])
    vlim = 0.
    mask_comp = np.ma.masked_array(arr, arr < vlim, fill_value=vlim)
    mask_res = np.ma.masked_array(arr, arr > vlim, fill_value=vlim)
    cmapc = truncate_colormap(cmaps[0], 0., 1 - offset_cm)
    aximc = ax.imshow(mask_comp, origin="lower", cmap=cmapc, interpolation='none', alpha=alpha,
                      norm='linear', vmax=1.)
    cmapr = cmaps[1]  # truncate_colormap(cmaps[1], 0., 1)
    aximr = ax.imshow(mask_res, origin="lower", interpolation='none', alpha=alpha, cmap=cmapr,
                      norm='linear', vmin=arr.min(), vmax=vlim, )
    ax.contour(arr, levels=[.9], colors="b")
    axinsc = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
    cbc = fig.colorbar(aximc, cax=axinsc, orientation="vertical", extend='max')
    cbc.ax.hlines(0.9, 0, 1, color='b')
    axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center right', borderpad=-6)
    cbr = fig.colorbar(aximr, cax=axinsr, orientation="vertical")
    fig.suptitle(title)
    fig.show()


def plot_2_images(
        im_list,
        title_list,
        suptitle="",
        normalize=False,
        vmin=1e-3,
        norm='log',
        cm='cubehelix_r',
        cmn="CMRmap_r",
        sc=None,
):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    chan, pol = 0, 0
    vmax = im_list[0]['pixels'].data.max()
    fig = plt.figure(figsize=(13, 6))
    axes = fig.subplots(1, 2, sharex=True, sharey=True,
                        subplot_kw={'projection': im_list[0].image_acc.wcs.sub([1, 2]),
                                    'frameon': False})
    if norm == 'sqrt':
        n = mplc.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    for i, ax in enumerate(axes.flat):
        # ax = axes[i]
        arr = np.real(im_list[i]["pixels"].data[chan, pol, :, :])
        im_pos = np.ma.masked_array(arr, arr < vmin, fill_value=0.)
        if neg := (arr < -vmin).any():
            im_neg = np.ma.masked_array(arr, arr > -vmin, fill_value=0.)
            if norm == 'sqrt':
                aximn = ax.imshow(-im_neg, origin="lower", cmap=cmn, interpolation='none', alpha=0.8, norm=n)
            else:
                aximn = ax.imshow(-im_neg, origin="lower", cmap=cmn, interpolation='none', alpha=0.8, norm=norm,
                                  vmin=vmin, vmax=vmax)
        if norm == 'sqrt':
            aximp = ax.imshow(im_pos, origin="lower", cmap=cm, interpolation='none', norm=n)
        else:
            aximp = ax.imshow(im_pos, origin="lower", cmap=cm, interpolation='none', norm=norm, vmax=vmax, vmin=vmin)
        # ims = ax.imshow(arr, origin="lower", cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        if i == 0:
            ax.set_ylabel(im_list[i].image_acc.wcs.wcs.ctype[1])
        else:
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')
        if sc is not None:
            for component in sc:
                x, y = skycoord_to_pixel(component.direction, im_list[0].image_acc.wcs, 0, "wcs")
                ax.scatter(x, y, marker=".", color="red", s=3, alpha=.9)
        ax.set_xlabel(im_list[i].image_acc.wcs.wcs.ctype[0])
        ax.set_title(title_list[i])
        # fig.colorbar(ims, orientation="vertical", shrink=0.5, ax=ax)
        if i == 1:
            axinsp = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
            cbp = fig.colorbar(aximp, cax=axinsp, orientation="vertical")
            if neg:
                axinsn = inset_axes(axinsp, width="100%", height="100%", loc='center right', borderpad=-4)
                cbn = fig.colorbar(aximn, cax=axinsn, orientation="vertical")
                cbn.ax.invert_yaxis()
    fig.suptitle(suptitle)
    plt.subplots_adjust(top=0.88, bottom=0.11, left=0.06, right=0.9, hspace=0.17, wspace=0.05)
    plt.show()


if __name__ == "__main__":
    log = logging.getLogger("rascil-logger")
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))

    path = "/home/jarret/Documents/EPFL/PhD/ra_data/"
    pklname = path + "bootes.pkl"
    with open(pklname, 'rb') as handle:
        total_vis = pickle.load(handle)

    print(total_vis.dims)

    vis = total_vis.isel({
        "time": slice(0, total_vis.dims["time"], ntimes),
        # "time": slice(total_vis.dims["time"]//10, total_vis.dims["time"]//10 + 1),
        # "frequency": slice(vis.dims["frequency"] // 2, vis.dims["frequency"] // 2 + 1),
    })
    vis = vis.sel({"baselines": list(generate_baselines(nantennas)), })
    print("Selected vis: ", vis.dims)
    # Broken antennas: 12, 13, 16, 17, 47
    # Unusable baseline: (22, 23)
    # ut.myplot_uvcoverage(vis, title="Subsampled UV coverage")

    phasecentre = vis.phasecentre
    fov = fov_deg * np.pi / 180.
    cellsize = fov / npixel
    image_model = create_image_from_visibility(vis, npixel=npixel, cellsize=cellsize, override_cellsize=False)

    print("Field of view in degrees: {:.3f}".format(fov_deg))

    ## PolyCLEAN reconstruction

    image_model = ut.image_add_ra_dec_grid(image_model)
    directions = SkyCoord(
        ra=image_model.ra_grid.data.ravel() * u.rad,
        dec=image_model.dec_grid.data.ravel() * u.rad,
        frame="icrs", equinox="J2000", )
    direction_cosines = np.stack(skycoord_to_lmn(directions, phasecentre), axis=-1)
    uvwlambda = vis.visibility_acc.uvw_lambda.reshape(-1, 3)
    flags_bool = (vis.weight.data != 0.).reshape(-1)
    flagged_uvwlambda = uvwlambda[flags_bool]

    forwardOp = pc.generatorVisOp(direction_cosines=direction_cosines, vlambda=flagged_uvwlambda,
                                  nufft_eps=nufft_eps, chunked=False)
    start = time.time()
    fOp_lipschitz = forwardOp.estimate_lipschitz(method='svd', tol=1.)  # ~8000 in 18 min in chunked mode, no memory issue
    dt_lipschitz = time.time() - start
    print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)\n".format(dt_lipschitz))

    vis_array = pxc.view_as_real(vis.vis.data.reshape(-1)[flags_bool])
    sum_vis = vis_array.shape[0] // 2
    dirty_array = forwardOp.adjoint(vis_array)  # 28s in chunked mode
    lambda_ = lambda_factor * np.abs(dirty_array).max()

    stop_crit = reco.stop_crit(tmax, min_iter, eps)
    pclean_parameters = {
        # "lambda_factor": lambda_factor,
        "ms_threshold": ms_threshold,
        "init_correction_prec": init_correction_prec,
        "final_correction_prec": final_correction_prec,
        "min_correction_steps": min_correction_steps,
        "max_correction_steps": max_correction_steps,
        "remove_positions": remove,
        "nufft_eps": nufft_eps,
        "show_progress": False, }
    fit_parameters = {
        "stop_crit": stop_crit,
        "positivity_constraint": True,
        "diff_lipschitz": fOp_lipschitz ** 2,
        "lock_reweighting": lock,
        "precision_rule": lambda k: 10 ** (-k / 10), }

    # Computations
    pclean = pc.PolyCLEAN(
        data=vis_array,
        uvwlambda=flagged_uvwlambda,
        direction_cosines=direction_cosines,
        lambda_=lambda_,
        **pclean_parameters,
    )
    print("PolyCLEAN: Solving...")
    pclean_time = time.time()
    pclean.fit(**fit_parameters)
    print("\tSolved in {:.3f} seconds".format(dt_pclean := time.time() - pclean_time))
    if diagnostics:
        pclean.diagnostics(log=log_diagnostics)
    data, hist = pclean.stats()
    pclean_residual = forwardOp.adjoint(vis_array - forwardOp(data["x"]))

    print("PolyCLEAN final DCV: {:.3f}".format(data["dcv"]))
    print("Iterations: {}".format(int(hist['N_iter'][-1])))
    print("Final sparsity of the components: {}".format(np.count_nonzero(data["x"])))

    psf, _ = invert_visibility(vis, image_model, context=context, dopsf=True)
    clean_beam = fit_psf(psf)

    pclean_comp = image_model.copy(deep=True)
    pclean_comp.pixels.data[0, 0] = data["x"].reshape((npixel,) * 2)
    pclean_comp_restored = restore_cube(pclean_comp, None, None, clean_beam=clean_beam)

    pclean_residual_im = image_model.copy(deep=True)
    pclean_residual_im.pixels.data = pclean_residual.reshape(pclean_residual_im.pixels.data.shape) / sum_vis
    pclean_restored = restore_cube(pclean_comp, None, pclean_residual_im, clean_beam=clean_beam)

    dirty_image = image_model.copy(deep=True)
    dirty_image.pixels.data = dirty_array.reshape(dirty_image.pixels.data.shape) / sum_vis

    ## Plot and save the results

    folder_path = "/home/jarret/PycharmProjects/polyclean/figures/lofar_ps/polyclean/" + str(lambda_factor)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    plot_1_image(dirty_image, "Dirty image", alpha=0.95)
    if save:
        plt.savefig(folder_path + "/dirty.png")
    plot_1_image(pclean_restored, "PolyCLEAN {:.3f} - {:.2f}s".format(lambda_factor, dt_pclean), alpha=0.95)
    if save:
        plt.savefig(folder_path + "/restored.png")

    if save_unified:
        vlim = -(pclean_restored.pixels.data.min() + dirty_image.pixels.data.min()) / 2
        plot_1_image(dirty_image, "Dirty image", alpha=0.95, vlim=vlim)
        plt.savefig(folder_path + "/dirty_unified.png")
        plot_1_image(pclean_restored, "PolyCLEAN {:.3f} - {:.2f}s".format(lambda_factor, dt_pclean), alpha=0.95, vlim=vlim)
        plt.savefig(folder_path + "/restored_unified.png")

    pclean_comp_restored.image_acc.export_to_fits(folder_path + "/components.fits")
    pclean_restored.image_acc.export_to_fits(folder_path + "/restored.fits")

    do_sharp_beam = True
    if do_sharp_beam:
        sharp_beam = clean_beam.copy()
        sharp_beam["bmin"] = clean_beam["bmin"] / 2
        sharp_beam["bmaj"] = clean_beam["bmaj"] / 2

        pclean_comp_sharp = restore_cube(pclean_comp, None, None, clean_beam=sharp_beam)
        pclean_sharp = restore_cube(pclean_comp, None, pclean_residual_im, clean_beam=sharp_beam)
        plot_1_image(pclean_sharp, "PolyCLEAN sharp{:.3f} - {:.2f}s".format(lambda_factor, dt_pclean), alpha=0.95)
        if save:
            plt.savefig(folder_path + "/restored_sharp.png")

        pclean_comp_sharp.image_acc.export_to_fits(folder_path + "/components_sharp.fits")
        pclean_sharp.image_acc.export_to_fits(folder_path + "/restored_sharp.fits")

    if save_im_pkl:
        import pickle
        folder_path = "/home/jarret/PycharmProjects/polyclean/scripts/observations/reco_pkl/" + str(lambda_factor)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(folder_path + "/restored.pkl", 'wb') as handle:
            pickle.dump(pclean_restored, handle)
        if do_sharp_beam:
            with open(folder_path + "/restored_sharp.pkl", 'wb') as handle:
                pickle.dump(pclean_sharp, handle)



    ## Dual certificate

    ### Load the catalogue locations
    catalog = "bootes_catalog.npz"
    filecatalog = path + catalog
    with np.load(filecatalog, allow_pickle=True) as f:
        lst = f.files
        src_ra_dec_flux_rad = f[lst[0]]
    peak_thresh = 30  # 20
    print("Select source from the catalog with peak flux higher than {:d} Jy:".format(peak_thresh))
    src_ra_dec_flux_rad = src_ra_dec_flux_rad[:, src_ra_dec_flux_rad[-1] > peak_thresh]
    print("\t{:d} sources selected.".format(src_ra_dec_flux_rad.shape[1]))

    sky_coords = SkyCoord(ra=src_ra_dec_flux_rad[0] * u.rad,
                          dec=src_ra_dec_flux_rad[1] * u.rad,
                          frame="icrs", equinox="J2000")
    sc = [SkyComponent(sky_coords[i], flux=src_ra_dec_flux_rad[-1, i].reshape((1, 1)),
                       frequency=np.r_[total_vis.frequency],
                       shape='Point',
                       polarisation_frame=PolarisationFrame("stokesI")
                       ) for i in range(len(sky_coords))]  # if coord_in_fov(sky_coords[i], phasecentre, fov_deg)]
    # catalogue_image = image_model.copy(deep=True)
    # insert_skycomponent(catalogue_image, sc, insert_method='Nearest')

    dual_certificate_im = image_model.copy(deep=True)
    dual_certificate_im.pixels.data = pclean_residual.reshape(dual_certificate_im.pixels.data.shape) / lambda_

    # fig = plt.figure(figsize=(12, 12))
    # chan, pol = 0, 0
    # cmap = "cubehelix_r"
    # ax = fig.subplots(1, 1, subplot_kw={'projection': dual_certificate_im.image_acc.wcs.sub([1, 2]), 'frameon': False})
    # dual_certif_arr = np.real(dual_certificate_im["pixels"].data[chan, pol, :, :])
    # ims = ax.imshow(dual_certif_arr, origin="lower", cmap=cmap, interpolation="none")
    # ax.set_ylabel(image_model.image_acc.wcs.wcs.ctype[1])
    # ax.set_xlabel(image_model.image_acc.wcs.wcs.ctype[0])
    # ax.contour(dual_certif_arr, levels=[.9], colors="c")
    # fig.suptitle("Dual certificate image - maximum value: {:.3f}".format(dual_certif_arr.max()))
    # # for component in sc:
    # #     x, y = skycoord_to_pixel(component.direction, dual_certificate_im.image_acc.wcs, 0, "wcs")
    # #     ax.scatter(x, y, marker="+", color="red", s=30, alpha=.9)
    # axins = inset_axes(ax, width="4%", height="100%", loc='center right', borderpad=-5)
    # cb = fig.colorbar(ims, cax=axins, orientation="vertical")
    # cb.ax.hlines(0.9, 0, 1, color='c')
    # plt.show()
    #
    # folder_path = "/home/jarret/PycharmProjects/polyclean/figures/lofar_ps/dual_certificate/" + str(lambda_factor)
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    # if save:
    #     plt.savefig(folder_path + "/certif_0.9.pdf")

    comp_im = pclean_comp_restored.pixels.data[0, 0]
    res_im = pclean_residual_im.pixels.data[0, 0]

    def symm_sqrt_norm(vmin, vmax):  # norm=symm_sqrt_norm(arr.min(), vlim)
        def _forward(x):
            return np.sqrt(np.abs(x)) * np.sign(x)

        def _inverse(x):
            return np.sign(x) * x ** 2

        return mplc.FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax)


    arr = dual_certificate_im.pixels.data[0, 0]
    folder_path = "/home/jarret/PycharmProjects/polyclean/figures/lofar_ps/dual_certificate/" + str(lambda_factor)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cmaps = ['hot', 'Greys']

    ## option 1
    alpha = 0.95
    offset_cm = 0.0

    fig = plt.figure(figsize=(12, 10))
    ax = fig.subplots(1, 1, subplot_kw={'projection': image_model.image_acc.wcs.sub([1, 2]), 'frameon': False})
    ax.set_xlabel(image_model.image_acc.wcs.wcs.ctype[0])
    ax.set_ylabel(image_model.image_acc.wcs.wcs.ctype[1])
    vlim = 0.8
    mask_comp = np.ma.masked_array(arr, arr < vlim, fill_value=vlim)
    mask_res = np.ma.masked_array(arr, arr > vlim, fill_value=vlim)
    cmapc = truncate_colormap(cmaps[0], offset_cm, 1.)
    aximc = ax.imshow(mask_comp, origin="lower", cmap=cmapc, interpolation='none', alpha=alpha,
                      norm='linear', vmax=1.)
    # norm=mplc.PowerNorm(gamma=0.5, vmin=vlim, vmax=1. * mask_comp.max()))
    cmapr = truncate_colormap(cmaps[1], 0., 1 - offset_cm)
    aximr = ax.imshow(mask_res, origin="lower", interpolation='none', alpha=alpha, cmap=cmapr,
                      norm='linear', vmin=arr.min(), vmax=vlim, )
    # norm=symm_sqrt_norm(arr.min(), vlim),)
    axinsc = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
    cbc = fig.colorbar(aximc, cax=axinsc,
                       orientation="vertical", extend='max')  # ticks=[50, 1000, 2000, 3000, 4000])
    axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center right', borderpad=-6)
    cbr = fig.colorbar(aximr, cax=axinsr, orientation="vertical")
    # ax.contour(arr, levels=[.9], colors="b")
    # cbc.ax.hlines(0.9, 0, 1, color='b')
    fig.suptitle("Dual certificate image - maximum value: {:.3f}".format(arr.max()))
    for s in sc:
        x, y = skycoord_to_pixel(s.direction, image_model.image_acc.wcs, 0, "wcs")
        ax.scatter(x, y, marker="+", color="blue", s=60, alpha=.9)
    fig.show()
    if save:
        plt.savefig(folder_path + "/certif_0.8_sources.png")

    # ## option 2
    # alpha = 0.95
    # offset_cm = 0.0
    #
    # fig = plt.figure(figsize=(12, 10))
    # ax = fig.subplots(1, 1, subplot_kw={'projection': image_model.image_acc.wcs.sub([1, 2]), 'frameon': False})
    # ax.set_xlabel(image_model.image_acc.wcs.wcs.ctype[0])
    # ax.set_ylabel(image_model.image_acc.wcs.wcs.ctype[1])
    # vlim = 0.
    # mask_comp = np.ma.masked_array(arr, arr < vlim, fill_value=vlim)
    # mask_res = np.ma.masked_array(arr, arr > vlim, fill_value=vlim)
    # cmapc = truncate_colormap(cmaps[0], offset_cm, 1.)
    # aximc = ax.imshow(mask_comp, origin="lower", cmap=cmapc, interpolation='none', alpha=alpha,
    #                   norm='linear', vmax=1.)
    # # norm=mplc.PowerNorm(gamma=0.5, vmin=vlim, vmax=1. * mask_comp.max()))
    # cmapr = truncate_colormap(cmaps[1], 0., 1 - offset_cm)
    # aximr = ax.imshow(mask_res, origin="lower", interpolation='none', alpha=alpha, cmap=cmapr,
    #                   norm='linear', vmin=arr.min(), vmax=vlim, )
    # # norm=symm_sqrt_norm(arr.min(), vlim),)
    # axinsc = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
    # cbc = fig.colorbar(aximc, cax=axinsc,
    #                    orientation="vertical", extend='max') # ticks=[50, 1000, 2000, 3000, 4000])
    # ax.contour(arr, levels=[.9], colors="b")
    # cbc.ax.hlines(0.9, 0, 1, color='b')
    # axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center right', borderpad=-6)
    # cbr = fig.colorbar(aximr, cax=axinsr, orientation="vertical")
    # fig.suptitle("Dual certificate image - maximum value: {:.3f}".format(arr.max()))
    # fig.show()
    # if save:
    #     plt.savefig(folder_path + "/certif_0.9_option2.png")
