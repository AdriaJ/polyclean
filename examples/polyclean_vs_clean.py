import logging
import sys
import time

import numpy as np
import datetime as dt
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

from rascil.data_models import PolarisationFrame
from rascil.processing_components import (
    create_visibility,
    deconvolve_cube,
    create_named_configuration,
    create_image_from_visibility,
    fit_psf,
    restore_cube,
    predict_visibility,
    invert_visibility
)

from rascil.processing_components.util import skycoord_to_lmn

import pyxu.opt.stop as pxos
import pyxu.util.complex as pxc
import pyxu.operator as pxop
import pyxu.opt.solver.pgd as pxpgd

import pyfwl

import polyclean.image_utils as ut
from polyclean import PolyCLEAN, generatorVisOp, diagnostics_polyclean  # , psf_kernel


# matplotlib.use("Qt5Agg")
log = logging.getLogger("rascil-logger")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

##########################################################
seed = np.random.randint(0, 1000)  # np.random.randint(0, 1000)  # 492
rmax = 300.  # 2000.
times = np.zeros([1])
# np.linspace(-2, +2, 5) * (np.pi / 12.0)
# np.zeros([1])
# np.linspace(-3, +3, 13) * (np.pi / 12.0)  # radians
fov_deg = 5
npixel = 256  #512  # 384 #  128 * 2
npoints = 100
context = "ng"
nufft_eps = 1e-3

lambda_factor = .05
lambda_factor_plus = .05

eps = 1e-4
tmax = 240.
min_iter = 20
ms_threshold = 0.8
init_correction_prec = 5e-2
final_correction_prec = 1e-4
remove = True
min_correction_steps = 3
minor_cycle_length = 0

if __name__ == "__main__":
    duration_stop = pxos.MaxDuration(t=dt.timedelta(seconds=tmax))
    min_iter_stop = pxos.MaxIter(n=min_iter)
    stop_crit = pxos.RelError(
        eps=eps,
        var="objective_func",
        f=None,
        norm=2,
        satisfy_all=True,
    )
    stop = (stop_crit & min_iter_stop) | duration_stop

    if seed is None:
        seed = np.random.randint(1000)
    print("Seed: {}".format(seed))
    rng = np.random.default_rng(seed)

    # Construct LOW core configuration
    lowr3 = create_named_configuration("LOWBD2", rmax=rmax)

    frequency = np.array([1e8])
    channel_bandwidth = np.array([1e6])
    phasecentre = SkyCoord(
        ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
    )
    vt = create_visibility(
        lowr3,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        weight=1.0,
        phasecentre=phasecentre,
        polarisation_frame=PolarisationFrame("stokesI"),
    )

    ##########################################################
    ## Generation of the source

    fov = fov_deg * np.pi / 180
    cellsize = fov / npixel
    flux_sigma = 0.4
    radius = 0.9 * fov

    sky_im, sc = ut.generate_point_sources(npoints,
                                           fov_deg,
                                           npixel,
                                           flux_sigma,
                                           phasecentre=phasecentre,
                                           frequency=frequency,
                                           channel_bandwidth=channel_bandwidth,
                                           seed=seed)

    ##########################################################

    ## Image deconvolution

    # CLEAN
    predicted_visi = predict_visibility(vt, sky_im, context=context)
    image_model = create_image_from_visibility(predicted_visi, cellsize=cellsize, npixel=npixel)

    clean_model = create_image_from_visibility(predicted_visi, cellsize=cellsize, npixel=2 * npixel)
    dirty, sumwt_dirty = invert_visibility(predicted_visi, clean_model, context=context)
    psf, sumwt = invert_visibility(predicted_visi, image_model, context=context, dopsf=True)
    tmp_clean_comp, tmp_clean_residual = deconvolve_cube(
        dirty,
        psf,
        niter=10000,
        threshold=0.001,
        fractional_threshold=0.001,
        window_shape="quarter",
        gain=0.7,
        scales=[0, 3, 10, 30],
        algorithm='hogbom',
    )
    clean_comp = image_model.copy(deep=True)
    clean_comp['pixels'].data[0, 0, ...] = \
        tmp_clean_comp['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]
    clean_residual = image_model.copy(deep=True)
    clean_residual['pixels'].data[0, 0, ...] = \
        tmp_clean_residual['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]

    ##########################################################

    # computation of the visibilities
    directions = SkyCoord(
        ra=sky_im.ra_grid.data.ravel() * u.rad,
        dec=sky_im.dec_grid.data.ravel() * u.rad,
        frame="icrs",
        equinox="J2000",
    )
    direction_cosines = np.stack(skycoord_to_lmn(directions, phasecentre), axis=-1)
    vlambda = vt.visibility_acc.uvw_lambda.reshape(-1, 3)
    flagged_weights_bool = np.any(vlambda != 0., axis=-1)
    flagged_vlambda = vlambda[flagged_weights_bool]
    forwardOp = generatorVisOp(direction_cosines=direction_cosines,
                               vlambda=flagged_vlambda,
                               nufft_eps=nufft_eps)
    measurements = forwardOp(sky_im.pixels.data.reshape(-1))
    start = time.time()
    fOp_lipschitz = forwardOp.estimate_lipschitz(method='svd', tol=1.)
    lipschitz_time = time.time() - start
    print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)".format(lipschitz_time))
    # start = time.time()
    # fOp_lipschitz2 = generatorVisOp(direction_cosines=direction_cosines,
    #                            vlambda=flagged_vlambda,
    #                            nufft_eps=10.).estimate_lipschitz(method='svd', tol=10.)
    # print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)".format(time.time() - start))
    # print(fOp_lipschitz, fOp_lipschitz2)

    ## Image reconstruction

    # PolyCLEAN # measurements should have the format of complex numbers returned by NUFFT
    pclean = PolyCLEAN(
        vt,
        image_model,
        data=measurements,
        lambda_factor=lambda_factor,
        nufft_eps=nufft_eps,
        flagged_bool_mask=None,
        ms_threshold=ms_threshold,
        init_correction_prec=init_correction_prec,
        final_correction_prec=final_correction_prec,
        remove_positions=remove,
        min_correction_steps=min_correction_steps,
        show_progress=False,
    )
    print("PolyCLEAN: Solving...")
    pclean_time = time.time()
    pclean.fit(stop_crit=stop, positivity_constraint=True, diff_lipschitz=fOp_lipschitz ** 2)
    print("\tSolved in {:.3f} seconds".format(time.time() - pclean_time))

    data, hist = pclean.stats()
    pclean_comp = sky_im.copy(deep=True)
    pclean_comp["pixels"].data[0, 0] = data["x"].reshape((npixel, npixel))

    # PolyCLEAN+
    # PolyCLEAN # measurements should have the format of complex numbers returned by NUFFT
    pclean_plus = PolyCLEAN(
        vt,
        image_model,
        data=measurements,
        lambda_factor=lambda_factor_plus,
        nufft_eps=nufft_eps,
        flagged_bool_mask=None,
        ms_threshold=ms_threshold,
        init_correction_prec=init_correction_prec,
        final_correction_prec=final_correction_prec,
        remove_positions=remove,
        min_correction_steps=min_correction_steps,
        show_progress=False,
    )
    print("PolyCLEAN+: Solving...")
    pclean_time = time.time()
    pclean_plus.fit(stop_crit=stop, positivity_constraint=True, diff_lipschitz=fOp_lipschitz ** 2)
    print("\tSolved in {:.3f} seconds".format(time.time() - pclean_time))

    data_plus, hist_plus = pclean_plus.stats()
    pclean_plus_comp = sky_im.copy(deep=True)
    pclean_plus_comp["pixels"].data[0, 0] = data_plus["x"].reshape((npixel, npixel))

    # APGD (for time comparison)

    data_fid_synth = 0.5 * pxop.SquaredL2Norm(dim=forwardOp.shape[0]).argshift(-measurements) * forwardOp
    regul_synth = pclean.lambda_ * pyfwl.L1NormPositivityConstraint(shape=(1, None))
    apgd = pxpgd.PGD(data_fid_synth, regul_synth, show_progress=False)
    print("APGD: Solving ...")
    apgd.fit(
        x0=np.zeros(forwardOp.shape[1], dtype="float64"),
        # stop_crit=(min_iter_stop & apgd.default_stop_crit()) | duration_stop,
        stop_crit=(min_iter_stop & pxos.AbsError(eps=hist["Memorize[objective_func]"][-1],
                                                  var="objective_func")) | duration_stop,
        track_objective=True,
        tau=1 / (fOp_lipschitz ** 2),
    )
    apgd_data, apgd_history = apgd.stats()
    print("\tSolved in {:.3f} seconds".format(apgd_history['duration'][-1]))
    apgd_comp = sky_im.copy(deep=True)
    apgd_comp["pixels"].data[0, 0] = apgd_data["x"].reshape((npixel, npixel))

    # # Make sure the solver is working fine
    # plt.figure()
    # plt.scatter(apgd_history['duration'], apgd_history['Memorize[objective_func]'], label="APGD", s=20, marker="+")
    # plt.title('Reconstruction: LASSO objective function')
    # plt.legend()
    # plt.show()

    ## Restauration of the images
    clean_beam = fit_psf(psf)
    convolved_sky_im = restore_cube(sky_im, None, None, clean_beam=clean_beam)
    clean_comp_restored = restore_cube(clean_comp, None, None, clean_beam=clean_beam)
    pclean_comp_restored = restore_cube(pclean_comp, None, None, clean_beam=clean_beam)
    pclean_plus_comp_restored = restore_cube(pclean_plus_comp, None, None, clean_beam=clean_beam)
    apgd_comp_restored = restore_cube(apgd_comp, None, None, clean_beam=clean_beam)

    cropped_dirty = image_model.copy(deep=True)
    cropped_dirty['pixels'].data[0, 0, ...] = \
        dirty['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]

    print("\nPolyCLEAN final value: {:.3e}".format(hist["Memorize[objective_func]"][-1]))
    print("\nAPGD final value: {:.3e}".format(apgd_history["Memorize[objective_func]"][-1]))

    print("PolyCLEAN final DCV: {:.3f}".format(data["dcv"]))
    print("Iterations: {}".format(int(hist['N_iter'][-1])))
    print("Final sparsity: {}".format(np.count_nonzero(data["x"])))
    print("\n")

    print("\nPolyCLEAN+ final value: {:.3e}".format(hist_plus["Memorize[objective_func]"][-1]))
    print("PolyCLEAN+ final DCV: {:.3f}".format(data_plus["dcv"]))
    print("Iterations: {}".format(int(hist_plus['N_iter'][-1])))
    print("Final sparsity: {}".format(np.count_nonzero(data_plus["x"])))
    print("\n")

    print("Final sparsity CLEAN: {}".format(np.count_nonzero(clean_comp.pixels.data)))
    print("Final sparsity APGD: {}".format(np.count_nonzero(apgd_data["x"])))

    print("MSE with the source components, convolved with the synthetic beam:")
    print("\tCLEAN : {:.3e}".format(ut.MSE(convolved_sky_im, clean_comp_restored)[0, 0]))
    print("\tLASSO (PolyCLEAN) : {:.3e}".format(ut.MSE(convolved_sky_im, apgd_comp_restored)[0, 0]))
    print("\tLASSO (APGD) : {:.3e}".format(ut.MSE(convolved_sky_im, pclean_comp_restored)[0, 0]))
    print("\tLASSO (PolyCLEAN+) : {:.3e}".format(ut.MSE(convolved_sky_im, pclean_plus_comp_restored)[0, 0]))

    ut.display_image_error(convolved_sky_im, cropped_dirty, clean_comp_restored, pclean_comp_restored,
                           titles=["CLEAN", "PolyCLEAN"], sc=sc,
                           suptitle="Components convolved with synthetic beam",
                           normalize=True, cm="cubehelix_r", sc_marker=".", sc_color="k", sc_size=5)

    ut.compare_3_images(sky_im, clean_comp, pclean_comp,
                        titles=["CLEAN", "PolyCLEAN"], sc=sc,
                        suptitle="Raw components", normalize=True)

    diagnostics_polyclean(pclean, hist, log=False)

    # try to estimate shrinkage effect on the true values
    # qq_plot_point_sources(sky_im, clean_comp, pclean_comp, ["CLEAN", "PolyCLEAN"])

    ## Least squares reweighting

    import pyxu.operator as pxop
    import pyxu.opt.solver as pxsol

    duration_stop_rw = pxos.MaxDuration(t=dt.timedelta(seconds=hist_plus["duration"][-1] / 5))
    sol = data_plus["x"]
    support = np.nonzero(sol)[0]
    rs_forwardOp = generatorVisOp(direction_cosines=direction_cosines[support, :],
                                  vlambda=flagged_vlambda,
                                  nufft_eps=nufft_eps)
    rs_data_fid = .5 * pxop.SquaredL2Norm(dim=measurements.shape[0]).argshift(-measurements) * rs_forwardOp
    rs_regul = pxop.PositiveOrthant(dim=rs_forwardOp.shape[1])
    rw_apgd = pxsol.PGD(rs_data_fid, rs_regul, show_progress=False)
    rw_apgd.fit(x0=sol[support],
                stop_crit=(min_iter_stop & pxos.RelError(1e-4)) | duration_stop_rw,
                track_objective=True,
                tau=1 / (fOp_lipschitz ** 2))
    print("Least squares reweighting:")
    print("\tSolved in {:.3f} seconds".format(rw_apgd.stats()[1]['duration'][-1]))
    res2 = rw_apgd.stats()[0]["x"]

    print("Final sparsity reweighted: {}".format(np.count_nonzero(res2)))

    reweighted_image = image_model.copy(deep=True)
    reweighted_image.pixels.data[0, 0][np.unravel_index(support, shape=(npixel,) * 2)] = res2
    # qq_plot_point_sources(sky_im, pclean_comp, reweighted_image, titles=["PolyCLEAN", "Reweighted"])
    reweighted_restored = restore_cube(reweighted_image, None, None, clean_beam=clean_beam)
    print("\tLASSO with reweighting : {:.3e}".format(ut.MSE(convolved_sky_im, reweighted_restored)[0, 0]))

    ut.qq_plot_point_sources_stacked(sky_im, clean_comp, pclean_comp, reweighted_image, title="rmax = {}".format(rmax))

    history = rw_apgd.stats()[1]
    # plt.figure()
    # plt.scatter(history['duration'], history['Memorize[objective_func]'], marker='.', s=15)
    # plt.show()

    # display_image(cropped_dirty, cmap="Greys")

    timings = ["see CLEAN",
               lipschitz_time + apgd_history["duration"][-1],
               lipschitz_time + hist["duration"][-1],
               lipschitz_time + hist_plus["duration"][-1] + history["duration"][-1]]
    mse = [ut.MSE(convolved_sky_im, clean_comp_restored)[0, 0],
           ut.MSE(convolved_sky_im, apgd_comp_restored)[0, 0],
           ut.MSE(convolved_sky_im, pclean_comp_restored)[0, 0],
           ut.MSE(convolved_sky_im, pclean_plus_comp_restored)[0, 0],
           ut.MSE(convolved_sky_im, reweighted_restored)[0, 0]]
    sparsity = [np.count_nonzero(clean_comp.pixels.data),
                np.count_nonzero(apgd_data["x"]),
                np.count_nonzero(data["x"]),
                np.count_nonzero(res2)]

    # display Poly CLEAN dual certificate
    op = pclean.forwardOp
    dual_certificate = (1 / pclean.lambda_) * op.adjoint(measurements - op(data["x"])).reshape((npixel,) * 2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=sky_im.image_acc.wcs.sub([1, 2]))
    im = ax.imshow(dual_certificate, interpolation="none", origin="lower", cmap="cubehelix_r")
    ax.set_ylabel(sky_im.image_acc.wcs.wcs.ctype[1])
    ax.set_xlabel(sky_im.image_acc.wcs.wcs.ctype[0])
    ax.contour(dual_certificate, levels=[.8], colors='c')
    fig.colorbar(im)
    plt.show()

    ut.display_image_error(convolved_sky_im, cropped_dirty, clean_comp_restored, reweighted_restored,
                           titles=["CLEAN", "PolyCLEAN"], sc=sc,
                           suptitle="Components convolved with synthetic beam",
                           normalize=True, cm="cubehelix_r", sc_marker=".", sc_color="k", sc_size=5)
