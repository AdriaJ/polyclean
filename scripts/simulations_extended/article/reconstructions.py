import numpy as np
import time
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astropy import units as u
from astropy.coordinates import SkyCoord

from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_func_python.imaging import predict_visibility, invert_visibility, create_image_from_visibility
from ska_sdp_func_python.image import restore_list, fit_psf
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame

from rascil.processing_components.simulation import create_test_image
from rascil.processing_components.visibility.base import export_visibility_to_ms
from rascil.processing_components.image.operations import import_image_from_fits

import pyxu.util.complex as pxc

import polyclean.reconstructions as reco
import polyclean.image_utils as ut
import polyclean.polyclean as pc
import polyclean.kernels as pck

from polyclean.clean_utils import mjCLEAN
from gauss_pclean import GaussPolyCLEAN

# Simulation parameters
rmax = 1000.
fov_deg = 6.5
npixel = 256
ntimes = 11
times = (np.arange(ntimes) - (ntimes - 1) / 2) * np.pi / (3 * (ntimes - 1) / 2)
nufft_eps = 1e-3

psnrdb = 20

# LASSO parameters
scales = [0, 2, 5, 8]
r = 1.2
scale_bias = [r ** k for k in range(len(scales))]
n_supp = 2
norm_kernels = 1
lambda_factor = .02
tmax = 240.
eps = 1e-5

# PolyCLEAN parameters
min_iter = 5
ms_threshold = 0.8
init_correction_prec = 5e-2
final_correction_prec = min(1e-4, eps)
remove = True
min_correction_steps = 5
max_correction_steps = 150
positivity_constraint = True
diagnostics = False

# CLEAN parameters
niter = 10_000
n_major = 10
gain = .7
algorithm = 'msclean'
context = "ng"

# WS-CLEAN parameters
thresh = 1

if __name__ == "__main__":
    # Image simulation
    frequency = np.array([1e8])
    channel_bandwidth = np.array([1e6])
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000")
    fov = fov_deg * np.pi / 180.
    cellsize = fov / npixel
    m31image = create_test_image(cellsize=cellsize, frequency=frequency, phasecentre=phasecentre, )
    m31image = ut.image_add_ra_dec_grid(m31image)
    directions = SkyCoord(ra=m31image.ra_grid.data.ravel() * u.rad, dec=m31image.dec_grid.data.ravel() * u.rad,
                          frame="icrs", equinox="J2000")
    direction_cosines = np.stack(skycoord_to_lmn(directions, phasecentre), axis=-1)

    # Visibility simulations
    lowr3 = create_named_configuration("LOWBD2", rmax=rmax)
    vt = create_visibility(lowr3, times, frequency, channel_bandwidth=channel_bandwidth, weight=1.0,
                           phasecentre=phasecentre, polarisation_frame=PolarisationFrame("stokesI"))
    image_model = create_image_from_visibility(vt, cellsize=cellsize, npixel=npixel)  # verify same size as m31image
    uvwlambda = vt.visibility_acc.uvw_lambda.reshape(-1, 3)
    flags_bool = np.any(uvwlambda != 0., axis=-1)
    flagged_uvwlambda = uvwlambda[flags_bool]
    measOp = pc.generatorVisOp(direction_cosines=direction_cosines,
                               vlambda=flagged_uvwlambda,
                               nufft_eps=nufft_eps)

    noiseless_measurements = measOp(m31image.pixels.data.reshape(-1))
    noise_scale = np.abs(noiseless_measurements).max() * 10 ** (-psnrdb / 20) / np.sqrt(2)
    noise = np.random.normal(0, noise_scale, noiseless_measurements.shape)
    measurements = noiseless_measurements + noise
    dirty_image = measOp.adjoint(measurements)

    skernels = pck.stackedKernels((npixel,) * 2, scales,
                                  n_supp=n_supp,
                                  tight_lipschitz=False,
                                  verbose=True,
                                  norm=norm_kernels,
                                  bias_list=scale_bias)
    forwardOp = measOp * skernels
    start = time.time()
    fOp_lipschitz = forwardOp.estimate_lipschitz(method='svd', tol=1.)
    lipschitz_time = time.time() - start
    print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)".format(lipschitz_time))

    ### Reconstructions
    stop_crit = reco.stop_crit(tmax, min_iter, eps)
    lambda_ = lambda_factor * np.abs(skernels.adjoint(dirty_image)).max()

    # Polyatomic FW
    pclean_parameters = {
        # "lambda_factor": lambda_factor,
        "ms_threshold": ms_threshold,
        "init_correction_prec": init_correction_prec,
        "final_correction_prec": final_correction_prec,
        "min_correction_steps": min_correction_steps,
        "max_correction_steps": max_correction_steps,
        "remove_positions": remove,
        "show_progress": False,
        "shortcut": False,
    }
    fit_parameters = {
        "stop_crit": stop_crit,
        "positivity_constraint": positivity_constraint,
        "diff_lipschitz": fOp_lipschitz ** 2,
        "precision_rule": lambda k: 10 ** (-k / 10),
    }

    print("GaussPolyCLEAN: Solving...")
    pclean = GaussPolyCLEAN(
        scales,
        data=measurements,
        uvwlambda=flagged_uvwlambda,
        direction_cosines=direction_cosines,
        kernel_bias=scale_bias,
        n_supp=n_supp,
        norm_kernels=norm_kernels,
        lambda_=lambda_,
        **pclean_parameters,
    )

    pclean_time = time.time()
    pclean.fit(**fit_parameters)
    print("\tSolved in {:.3f} seconds".format(dt_pclean := time.time() - pclean_time))
    if diagnostics:
        pclean.diagnostics()
    solution_pc, hist_pc = pclean.stats()

    # Reconstruction in the image domain
    sol_pc = skernels(solution_pc["x"])
    pclean_comp = image_model.copy(deep=True)
    pclean_comp.pixels.data[0, 0] = sol_pc.reshape((npixel,) * 2)
    pclean_residual = measOp.adjoint(measurements - measOp(sol_pc))
    sum_vis = measurements.shape[0] // 2
    pclean_residual_im = image_model.copy(deep=True)
    pclean_residual_im.pixels.data[0, 0] = pclean_residual.reshape((npixel,) * 2) / sum_vis

    # MS-CLEAN
    predicted_visi = predict_visibility(vt, m31image, context=context)
    predicted_visi.vis.data[predicted_visi.visibility_acc.flagged_weight.astype(bool)] += pxc.view_as_complex(noise)
    clean_model = create_image_from_visibility(predicted_visi, cellsize=cellsize, npixel=2 * npixel)
    dirty, sumwt_dirty = invert_visibility(predicted_visi, clean_model, context=context)
    psf, sumwt = invert_visibility(predicted_visi, clean_model, context=context, dopsf=True)

    print("CLEAN: Solving...")
    start = time.time()
    tmp_clean_comp, tmp_clean_residual = mjCLEAN(
        dirty,
        psf,
        n_major=n_major,
        n_minor=niter,
        vt=vt,
        threshold=0.001,
        fractional_threshold=0.001,
        window_shape="quarter",
        gain=gain,
        algorithm=algorithm,
    )
    clean_comp = image_model.copy(deep=True)
    clean_comp['pixels'].data[0, 0, ...] = \
        tmp_clean_comp['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]
    clean_residual = image_model.copy(deep=True)
    clean_residual['pixels'].data[0, 0, ...] = \
        tmp_clean_residual['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]
    dt = time.time() - start
    print("\tSolved in {:.3f} seconds".format(dt))

    # WS-CLEAN
    ws_dir = "wsclean-dir"
    print("WS-CLEAN: Solving...")
    if not os.path.exists(ws_dir):
        os.makedirs(ws_dir)
    filename = "ssms.ms"
    export_visibility_to_ms(ws_dir + "/" + filename, [predicted_visi], )

    start = time.time()
    os.system(
        f"wsclean -multiscale -auto-threshold {thresh} -size {npixel:d} {npixel:d} -scale {fov_deg / npixel:.6f} -mgain 0.7 "
        f"-niter {niter:d} -name ws -weight natural -quiet -no-dirty wsclean-dir/ssms.ms")
    print("\tRun in {:.3f}s".format(dt_wsclean := time.time() - start))
    os.system(f"mv ws-* wsclean-dir/")

    ### Reconstruction quality
    print("\nPolyCLEAN final value: {:.3e}".format(hist_pc["Memorize[objective_func]"][-1]))
    print("PolyCLEAN final DCV: {:.3f}".format(solution_pc["dcv"]))

    clean_beam = fit_psf(psf)
    srf = 2
    sharp_beam = clean_beam.copy()
    sharp_beam["bmin"] = clean_beam["bmin"] / srf
    sharp_beam["bmaj"] = clean_beam["bmaj"] / srf
    m31_convolved = restore_list([m31image, ], None, None, clean_beam=sharp_beam)[0]

    ws_comp = import_image_from_fits(ws_dir + '/' + f"ws-model.fits")
    ws_residual = import_image_from_fits(ws_dir + '/' + f"ws-residual.fits")

    comps = [pclean_comp, clean_comp, ws_comp]
    residuals = [pclean_residual_im, clean_residual, ws_residual]
    convolved = restore_list(comps, None, None, sharp_beam)
    convolved_res = restore_list(comps, None, residuals, sharp_beam)
    cropped_dirty, _ = invert_visibility(predicted_visi, image_model, context=context)
    m31_convolved = restore_list([m31image, ], None, None, clean_beam=sharp_beam)[0]

    # Metrics
    print("\nPolyCLEAN:")
    print("\tFinal sparsity (coefficients): {}/{}".format(np.count_nonzero(solution_pc["x"]), skernels.shape[1]))
    print("\tComponents: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31image, pclean_comp), ut.MAD(m31image, pclean_comp)))
    print("\tComponents convolved sharp: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31_convolved, convolved[0]),
                                                                        ut.MAD(m31_convolved, convolved[0])))
    print("\tComponents restored sharp: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31_convolved, convolved_res[0]),
                                                                       ut.MAD(m31_convolved, convolved_res[0])))
    print("\tTotal weight: {:.2f}/{:.2f}".format(sol_pc.sum(), m31image.pixels.data.sum()))

    print("\nMS-CLEAN:")
    print("\tComponents: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31image, clean_comp), ut.MAD(m31image, clean_comp)))
    print("\tComponents convolved sharp: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31_convolved, convolved[-1]),
                                                                        ut.MAD(m31_convolved, convolved[-1])))
    print("\tComponents restored sharp: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31_convolved, convolved_res[-1]),
                                                                       ut.MAD(m31_convolved, convolved_res[-1])))
    print("\tTotal weight: {:.2f}/{:.2f}".format(clean_comp.pixels.data.sum(), m31image.pixels.data.sum()))


    # Save the images
    if not os.path.exists("reco_pkl"):
        os.makedirs("reco_pkl")

    dirty_im = image_model.copy(deep=True)
    dirty_im.pixels.data[0, 0] = dirty_image.reshape((npixel,) * 2)
    with open(os.path.join(os.getcwd(), "reco_pkl", "dirty.pkl"), 'wb') as handle:
        pickle.dump(dirty_im, handle)

    with open(os.path.join(os.getcwd(), "reco_pkl", "source.pkl"), 'wb') as handle:
        pickle.dump(m31image, handle)
    with open(os.path.join(os.getcwd(), "reco_pkl", "source_conv.pkl"), 'wb') as handle:
        pickle.dump(m31_convolved, handle)

    names = ['pclean', 'rsclean', 'wsclean']
    for i in range(3):
        folder_path = os.path.join(os.getcwd(), 'reco_pkl', names[i])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(os.path.join(folder_path, "comp.pkl"), 'wb') as handle:
            pickle.dump(comps[i], handle)
        with open(os.path.join(folder_path, "comp_conv.pkl"), 'wb') as handle:
            pickle.dump(convolved[i], handle)
        with open(os.path.join(folder_path, "comp_conv_res.pkl"), 'wb') as handle:
            pickle.dump(convolved_res[i], handle)


    ## Deep analysis of the decomposition
    ## Separate the different images in the dictionary decomposition

    components_pc = pck.stack_list_components(solution_pc['x'], skernels, scales, (npixel, npixel))
    sources_list = pck.stack_list_sources(solution_pc['x'], skernels, scales, n_supp)

    n_comp = len(components_pc)
    vmin, vmax = sol_pc.min(), sol_pc.max()
    marker = '.'
    vlim = 0.1
    split = int(2 * vlim * 256 / (vmax + vlim))
    colors1 = plt.cm.hot(np.linspace(0.05, 1, 256 - split))
    colors2 = plt.cm.Greys(np.linspace(-0.3, 0.95, split))
    colors = np.vstack((colors2, colors1))
    mymap = mplc.LinearSegmentedColormap.from_list('my_colormap', colors)

    print("Min and max values of the solution images: {:.2f} / {:.2f}".format(vmin, vmax))
    fig = plt.figure(figsize=(4 * n_comp, 9))
    axes = fig.subplots(2, n_comp, sharex=True, sharey=True)
    for i in range(n_comp):
        ax = axes[0, i]
        ims = ax.scatter(*np.where(sources_list[i].T > 0), marker=marker, s=10, color='r')
        ims = ax.scatter(*np.where(sources_list[i].T < 0), marker=marker, s=10, color='b')
        ax.set_xlim([0, npixel])
        ax.set_ylim([0, npixel])
        ax.set_title("scale " + str(scales[i]))
        ax = axes[1, i]
        arr = components_pc[i]
        ims = ax.imshow(arr, origin="lower", cmap=mymap, vmin=-vlim, vmax=vmax, interpolation='none')
        # fig.colorbar(ims, orientation="vertical", shrink=0.8, ax=ax)
        if i == 3:
            axins = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
            cb = fig.colorbar(ims, cax=axins, orientation="vertical", )  # ticks=[-vlim, 0, vlim, vmax])
    fig.suptitle("PolyCLEAN  sources (up) and components (bottom)")
    plt.show()

    # Dual certificate(s)
    plot_dual_certif = False
    if plot_dual_certif:
        vmin, vmax = None, None
        certif = np.abs(forwardOp.adjoint(measurements - measOp(sol_pc))) / lambda_  # belong to the coefficients domain
        subcertif_list = pck.stack_list_sources(certif, skernels, scales, n_supp)
        fig = plt.figure(figsize=(4 * n_comp, 9))
        axes = fig.subplots(2, n_comp, sharex=True, sharey=True)
        for i in range(n_comp):
            ax = axes[0, i]
            ims = ax.imshow(subcertif_list[i], origin="lower", cmap='cubehelix_r', vmin=vmin, vmax=vmax)
            ax.set_title("scale " + str(scales[i]))
            ax = axes[1, i]
            ims = ax.imshow(np.where(subcertif_list[i] > 1., subcertif_list[i], 0), origin="lower", cmap='cubehelix_r',
                            vmin=vmin, vmax=vmax)
            # ax.set_title("scale " + str(scales[i]))
            # fig.colorbar(ims, orientation="vertical", shrink=0.5, ax=ax)
        plt.suptitle("Dual certificates")
        plt.show()