import logging
import sys
import pickle
import os
import datetime as dt
import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import polyclean.image_utils as ut
import polyclean.polyclean as pc

import pycsou.util.complex as pycuc
import pycsou.opt.stop as pycos

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs.utils import skycoord_to_pixel

from ska_sdp_func_python.imaging import create_image_from_visibility
from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.visibility.vis_utils import generate_baselines
from ska_sdp_datamodels.sky_model import SkyComponent



matplotlib.use("Qt5Agg")

nantennas = 28
ntimes = 5
npixel = 1024
fov_deg = 6.
context = "ng"

nufft_eps = 1e-3
lambda_factor = 0.02
eps = 0.1
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
    fOp_lipschitz = forwardOp.lipschitz(tol=1., tight=True)  # ~8000 in 18 min in chunked mode, no memory issue
    dt_lipschitz = time.time() - start
    print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)\n".format(dt_lipschitz))

    vis_array = pycuc.view_as_real(vis.vis.data.reshape(-1)[flags_bool])
    sum_vis = vis_array.shape[0]//2
    dirty_array = forwardOp.adjoint(vis_array)  # 28s in chunked mode
    lambda_ = lambda_factor * np.abs(dirty_array).max()

    duration_stop = pycos.MaxDuration(t=dt.timedelta(seconds=tmax))
    min_iter_stop = pycos.MaxIter(n=min_iter)
    stop_crit = pycos.AbsError(
        eps=eps,
        var="dcv",
        f=lambda x: np.abs(x) - 1,
        norm=1,
        satisfy_all=True,
    )
    stop_crit = (stop_crit & min_iter_stop) | duration_stop

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
        "precision_rule": lambda k: 10**(-k/10), }

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
    print("\tSolved in {:.3f} seconds".format(dt_pclean := time.time() - pclean_time))
    if diagnostics:
        pclean.diagnostics(log=log_diagnostics)
    data, hist = pclean.stats()
    pclean_residual = forwardOp.adjoint(vis_array - forwardOp(data["x"]))

    print("PolyCLEAN final DCV: {:.3f}".format(data["dcv"]))
    print("Iterations: {}".format(int(hist['N_iter'][-1])))
    print("Final sparsity of the components: {}".format(np.count_nonzero(data["x"])))

    ## Dual certificate

    ### Load the catalogue locations
    catalog = "bootes_catalog.npz"
    filecatalog = path + catalog
    with np.load(filecatalog, allow_pickle=True) as f:
        lst = f.files
        src_ra_dec_flux_rad = f[lst[0]]
    peak_thresh = 20
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
                       ) for i in range(len(sky_coords))]

    dual_certificate_im = image_model.copy(deep=True)
    dual_certificate_im.pixels.data = pclean_residual.reshape(dual_certificate_im.pixels.data.shape) / lambda_

    fig = plt.figure(figsize=(12, 12))
    chan, pol = 0, 0
    cmap = "cubehelix_r"
    ax = fig.subplots(1, 1, subplot_kw={'projection': dual_certificate_im.image_acc.wcs.sub([1, 2]), 'frameon': False})
    dual_certif_arr = np.real(dual_certificate_im["pixels"].data[chan, pol, :, :])
    ims = ax.imshow(dual_certif_arr, origin="lower", cmap=cmap, interpolation="none")
    ax.set_ylabel(image_model.image_acc.wcs.wcs.ctype[1])
    ax.set_xlabel(image_model.image_acc.wcs.wcs.ctype[0])
    ax.contour(dual_certif_arr, levels=[.9], colors="c")
    fig.suptitle("Dual certificate image - maximum value: {:.3f} - {:.2f}s".format(data["dcv"], dt_pclean))
    for component in sc:
        x, y = skycoord_to_pixel(component.direction, dual_certificate_im.image_acc.wcs, 0, "wcs")
        ax.scatter(x, y, marker="+", color="red", s=30, alpha=.9)
    axins = inset_axes(ax, width="4%", height="100%", loc='center right', borderpad=-5)
    cb = fig.colorbar(ims, cax=axins, orientation="vertical")
    cb.ax.hlines(0.9, 0, 1, color='c')
    plt.show()

    folder_path = f"/home/jarret/PycharmProjects/polyclean/examples/figures/certificate/{1/ntimes:.3f}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path + "/certif_0.9.pdf")
    plt.savefig(folder_path + "/certif_0.9.png")

    d = hist['AbsError[dcv]'][1:]
    plt.figure()
    plt.title("Value of the dcv - 1")
    plt.scatter(np.arange(1, d.size+1), d)
    plt.hlines(0.1, 1, d.size+.5, color='r')
    plt.yscale('log')
    plt.show()
