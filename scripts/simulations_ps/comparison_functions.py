"""
Comparison of the different image reconstruction methods in simulation, using a funcitonal implementation of the methods
 (i.e. no direct access to the class instances of the solvers).
"""

import numpy as np
import time
from astropy import units as u
from astropy.coordinates import SkyCoord

from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
# Metrics
from ska_sdp_func_python.imaging.imaging import invert_visibility
from ska_sdp_func_python.image.deconvolution import restore_cube, fit_psf, restore_list

import polyclean.reconstructions as reco
import polyclean.image_utils as ut
import polyclean.polyclean as pc

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

seed = 195  # np.random.randint(0, 1000)  # np.random.randint(0, 1000)  # 195
rmax = 700.  # 2000.
times = np.zeros([1])
fov_deg = 5
npixel = 1024  # 512  # 384 #  128 * 2
npoints = 200
nufft_eps = 1e-3

lambda_factor = .05
lambda_factor_pcp = lambda_factor
clean_iter = 500
diag = False

eps = 1e-4
tmax = 240.
min_iter = 5
ms_threshold = 0.8
init_correction_prec = 5e-2
final_correction_prec = min(1e-4, eps)
remove = True
min_correction_steps = 5

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(1000)
    print("Seed: {}".format(seed))
    rng = np.random.default_rng(seed)

    ### Simulation of the source

    frequency = np.array([1e8])
    channel_bandwidth = np.array([1e6])
    phasecentre = SkyCoord(
        ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
    )

    sky_im, sc = ut.generate_point_sources(npoints,
                                           fov_deg,
                                           npixel,
                                           flux_sigma=.8,
                                           radius_rate=.9,
                                           phasecentre=phasecentre,
                                           frequency=frequency,
                                           channel_bandwidth=channel_bandwidth,
                                           seed=seed)
    # parametrisation of the image
    directions = SkyCoord(
        ra=sky_im.ra_grid.data.ravel() * u.rad,
        dec=sky_im.dec_grid.data.ravel() * u.rad,
        frame="icrs",
        equinox="J2000",
    )
    direction_cosines = np.stack(skycoord_to_lmn(directions, phasecentre), axis=-1)

    ### Loading of the configuration
    lowr3 = create_named_configuration("LOWBD2", rmax=rmax)
    # baselines computation
    vt = create_visibility(
        lowr3,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        weight=1.0,
        phasecentre=phasecentre,
        polarisation_frame=PolarisationFrame("stokesI"),
    )
    uvwlambda = vt.visibility_acc.uvw_lambda.reshape(-1, 3)
    flags_bool = np.any(uvwlambda != 0., axis=-1)
    flagged_uvwlambda = uvwlambda[flags_bool]

    ### Simulation of the measurements
    forwardOp = pc.generatorVisOp(direction_cosines=direction_cosines,
                                  vlambda=flagged_uvwlambda,
                                  nufft_eps=nufft_eps)
    start = time.time()
    fOp_lipschitz = forwardOp.estimate_lipschitz(method='svd', tol=1.)
    lipschitz_time = time.time() - start
    print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)\n".format(lipschitz_time))

    measurements = forwardOp(sky_im.pixels.data.reshape(-1))
    dirty_image = forwardOp.adjoint(measurements)

    ### Reconstructions

    methods = ["PolyCLEAN",
               "PolyCLEAN+",
               "APGD",
               "CLEAN"]
    stop_crit = reco.stop_crit(tmax, min_iter, eps)
    lambda_ = lambda_factor * np.abs(dirty_image).max()
    lambda_pcp = lambda_factor_pcp * np.abs(dirty_image).max()

    # Parameters
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
    pcleanp_parameters = {
        # "lambda_factor": lambda_factor_pcp,
        "ms_threshold": ms_threshold,
        "init_correction_prec": init_correction_prec,
        "final_correction_prec": final_correction_prec,
        "min_correction_steps": min_correction_steps,
        "remove_positions": remove,
        "nufft_eps": nufft_eps,
        "show_progress": False,
        "overtime_lsr": 0.2,
    }
    apgd_parameters = {"nufft_eps": nufft_eps}
    clean_parameters = reco.get_clean_default_params()
    clean_parameters["niter"] = clean_iter
    fit_parameters_pc = {
        "stop_crit": stop_crit,
        "positivity_constraint": True,
        "diff_lipschitz": fOp_lipschitz ** 2
    }

    # Computations
    data_pc, hist_pc = reco.reco_pclean(
        flagged_uvwlambda, direction_cosines, measurements, lambda_, pclean_parameters, fit_parameters_pc, diagnostics=diag)
    data_pcp, hist_pcp1, hist_pcp2 = reco.reco_pclean_plus(
        flagged_uvwlambda, direction_cosines, measurements, lambda_pcp, pcleanp_parameters, fit_parameters_pc, diagnostics=diag)

    fit_parameters_apgd = {
        "x0": np.zeros(forwardOp.shape[1], dtype="float64"),
        "stop_crit": reco.stop_crit(tmax, min_iter, eps, value=hist_pc["Memorize[objective_func]"][-1]),
        "track_objective": True,
        "tau": 1 / (fOp_lipschitz ** 2)
    }
    data_apgd, hist_apgd = reco.reco_apgd(
        flagged_uvwlambda, direction_cosines, measurements, lambda_, apgd_parameters, fit_parameters_apgd)

    clean_comp, clean_residual, clean_duration = reco.reco_clean(sky_im, vt, clean_parameters)

    ### Performances of the solvers
    # PolyCLEAN vs APGD
    print("\nPolyCLEAN final value: {:.3e}".format(hist_pc["Memorize[objective_func]"][-1]))
    print("APGD final value: {:.3e}".format(hist_apgd["Memorize[objective_func]"][-1]))
    print("PolyCLEAN final DCV: {:.3f}".format(data_pc["dcv"]))
    print("APGD final DCV: {:.3f}".format(data_apgd["dcv"]))

    plt.figure()
    plt.scatter(hist_pc['duration'], hist_pc['Memorize[objective_func]'], label="PolyCLEAN", s=20, marker="+")
    plt.scatter(hist_apgd['duration'], hist_apgd['Memorize[objective_func]'], label="APGD", s=20, marker="+")
    plt.title('Reconstruction: LASSO objective function')
    plt.legend()
    plt.show()

    # PolyCLEAN vs PolyCLEAN+
    print("\nPolyCLEAN:")
    print("Iterations: {}".format(int(hist_pc['N_iter'][-1])))
    print("Final sparsity: {}".format(np.count_nonzero(data_pc["x"])))
    print("\n")
    print("PolyCLEAN+:")
    print("Iterations: {} + {}".format(int(hist_pcp1['N_iter'][-1]), int(hist_pcp2['N_iter'][-1])))
    print("Final sparsity: {}".format(np.count_nonzero(data_pcp["x"])))
    print("\n")

    # Final sparsity
    print("Final sparsity CLEAN: {}".format(np.count_nonzero(clean_comp.pixels.data)))
    print("Final sparsity APGD: {}".format(np.count_nonzero(data_apgd["x"])))

    ### Quality of the reconstructions

    pclean_comp = sky_im.copy(deep=True)
    pclean_comp.pixels.data[0, 0] = data_pc["x"].reshape((npixel,) * 2)
    pcleanp_comp = sky_im.copy(deep=True)
    pcleanp_comp.pixels.data[0, 0] = data_pcp["x"].reshape((npixel,) * 2)
    apgd_comp = sky_im.copy(deep=True)
    apgd_comp.pixels.data[0, 0] = data_apgd["x"].reshape((npixel,) * 2)

    psf, sumwt = invert_visibility(vt, sky_im, context="ng", dopsf=True)
    clean_beam = fit_psf(psf)
    """
    The clean beam is saved in deg, deg, deg, accounting for, after conversion in pixels, the xstd, ystd and rotation of
    the kernel. THe functions are then normalised in amplitude.
    Sharper kernel implies reducing both the first two components of clean_beam.
    """

    dirty = sky_im.copy(deep=True)
    dirty.pixels.data[0, 0] = dirty_image.reshape((npixel,) * 2)/(forwardOp.shape[0]/2)

    components = [pclean_comp, pcleanp_comp, apgd_comp, clean_comp]

    sky_im_restored = restore_cube(sky_im, None, None, clean_beam)
    restored_comps = restore_list(components, None, None, clean_beam)

    timings = [lipschitz_time + hist_pc["duration"][-1],
               lipschitz_time + hist_pcp1["duration"][-1] + hist_pcp2["duration"][-1],
               lipschitz_time + hist_apgd["duration"][-1],
               clean_duration,
               0.]
    sparsity = [np.count_nonzero(data_pc["x"]),
                np.count_nonzero(data_pcp["x"]),
                np.count_nonzero(data_apgd["x"]),
                np.count_nonzero(clean_comp.pixels.data), ]
    mse_comp = [ut.MSE(sky_im, im)[0, 0] for im in components] + [ut.MSE(sky_im, dirty)[0, 0], ]
    mse = [ut.MSE(sky_im_restored, im)[0, 0] for im in restored_comps] + [ut.MSE(sky_im_restored, dirty)[0, 0], ]
    cb_sharp = clean_beam.copy()
    cb_sharp["bmin"] = clean_beam["bmin"] / np.sqrt(2)
    cb_sharp["bmaj"] = clean_beam["bmaj"] / np.sqrt(2)
    sky_im_sharp = restore_cube(sky_im, None, None, cb_sharp)
    sharp_comps = restore_list(components, None, None, cb_sharp)
    mse_sharp = [ut.MSE(sky_im_sharp, im)[0, 0] for im in sharp_comps] + [ut.MSE(sky_im_sharp, dirty)[0, 0], ]
    cb_vsharp = clean_beam.copy()
    cb_vsharp["bmin"] = clean_beam["bmin"] / 2
    cb_vsharp["bmaj"] = clean_beam["bmaj"] / 2
    sky_im_vsharp = restore_cube(sky_im, None, None, cb_vsharp)
    vsharp_comps = restore_list(components, None, None, cb_vsharp)
    mse_vsharp = [ut.MSE(sky_im_vsharp, im)[0, 0] for im in vsharp_comps] + [ut.MSE(sky_im_vsharp, dirty)[0, 0], ]
    cb_wide = clean_beam.copy()
    cb_wide["bmin"] = clean_beam["bmin"]*np.sqrt(2)
    cb_wide["bmaj"] = clean_beam["bmaj"]*np.sqrt(2)
    sky_im_wide = restore_cube(sky_im, None, None, cb_wide)
    wide_comps = restore_list(components, None, None, cb_wide)
    mse_wide = [ut.MSE(sky_im_wide, im)[0, 0] for im in wide_comps] + [ut.MSE(sky_im_wide, dirty)[0, 0], ]

    intensity_sum = [im.pixels.data.sum() for im in components] + [sky_im.pixels.data.sum(), ]


    print("Methods :", *(m + " - " for m in methods), "Dirty image")
    print("Time :", *(f"{t:.1f} - " for t in timings))
    print("Sparsity :", *(f"{s} - " for s in sparsity))
    print("MSE wide :", *(f"{m:.2e} - " for m in mse_wide))
    print("MSE cb :", *(f"{m:.2e} - " for m in mse))
    print("MSE sharp :", *(f"{m:.2e} - " for m in mse_sharp))
    print("MSE v-sharp :", *(f"{m:.2e} - " for m in mse_vsharp))
    print("MSE components :", *(f"{m:.2e} - " for m in mse_comp))

    import pandas as pd
    df = [timings, sparsity, mse_wide, mse, mse_sharp, mse_vsharp, mse_comp, intensity_sum]
    table = pd.DataFrame(df, columns=methods + ["Dirty Im"], index=["Time", "Sparsity", "MSE wide", "MSE clean beam", "MSE sharp", "MSE very sharp", "MSE components", "Sum"])
    cols = table.columns.tolist()
    cols = cols[-1:-3:-1] + cols[:-2]
    table = table[cols]

    # for i, label in enumerate(["Time", "Sparsity", "MSE wide", "MSE clean beam", "MSE sharp", "MSE very sharp", "MSE components"]):
    #     table.loc[i] =


    # dirty, sumwt = invert_visibility(vt, sky_im, context="ng")
    # print(ut.MSE(sky_im_restored, dirty)[0][0])

    # Visualization

    # for i in range(3):
    #     ut.display_image_error(sky_im_restored, dirty, restored_comps[-1], restored_comps[i],
    #                        titles=["CLEAN", methods[i]], sc=sc,
    #                        suptitle="Components convolved with synthetic beam",
    #                        normalize=True, cm="cubehelix_r", sc_marker=".", sc_color="k", sc_size=5)

    i=1
    ut.display_image_error(sky_im_restored, dirty, restored_comps[-1], restored_comps[i],
                       titles=["CLEAN", methods[i]], sc=sc,
                       suptitle="Components convolved with synthetic beam",
                       normalize=True, cm="cubehelix_r", sc_marker=".", sc_color="k", sc_size=5)
    ut.display_image_error(sky_im_vsharp, dirty, vsharp_comps[-1], vsharp_comps[i],
                       titles=["CLEAN", methods[i]], sc=sc,
                       suptitle="Components convolved with very sharp beam",
                       normalize=True, cm="cubehelix_r", sc_marker=".", sc_color="k", sc_size=5)
    ut.display_image_error(sky_im_restored, dirty, restored_comps[-1], restored_comps[0],
                           titles=["CLEAN", methods[0]], sc=sc,
                           suptitle="Components convolved with synthetic beam",
                           normalize=True, cm="cubehelix_r", sc_marker=".", sc_color="k", sc_size=5)

    # ### sharp psf
    # clean_beam["bmin"] /= 2
    # clean_beam["bmaj"] /= 2
    # sky_im_sharp = restore_cube(sky_im, None, None, clean_beam)
    # sharp_comps = restore_list(components, None, None, clean_beam)
    # mse_sharp = [ut.MSE(sky_im_sharp, im)[0, 0] for im in sharp_comps]
    #
    # ut.display_image_error(sky_im_sharp, dirty, sharp_comps[-1], sharp_comps[i],
    #                    titles=["CLEAN", methods[i]], sc=sc,
    #                    suptitle="Components convolved with synthetic beam",
    #                    normalize=True, cm="cubehelix_r", sc_marker=".", sc_color="k", sc_size=5)
    #
    # ut.plot_image(sky_im_sharp, sc=sc)

    ### QQplot

    ut.qq_plot_point_sources_stacked(sky_im, clean_comp, pclean_comp, pcleanp_comp, title="rmax = {}".format(rmax))
    ut.plot_image(pclean_comp, sc=sc, title="pCLEAN")
    ut.plot_image(pcleanp_comp, sc=sc, title="pCLEAN+")
    ut.plot_image(apgd_comp, sc=sc, title="APGD")
    # todo plot components on the same graph
