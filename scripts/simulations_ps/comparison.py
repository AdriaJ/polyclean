"""
Compares the reconstructions obtained with different algorithms on simulated point sources data.
"""
import numpy as np
import time
from astropy import units as u
from astropy.coordinates import SkyCoord

from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_func_python.imaging import predict_visibility, invert_visibility, create_image_from_visibility
from ska_sdp_func_python.image.deconvolution import restore_cube, fit_psf, restore_list, deconvolve_cube

import pyxu.operator as pxop
import pyxu.opt.solver as pxsol

import pyfwl

import polyclean.reconstructions as reco
import polyclean.image_utils as ut
import polyclean.polyclean as pc


# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

seed = 195  # np.random.randint(0, 1000)  # np.random.randint(0, 1000)  # 195
rmax = 800.  # 2000.
times = np.zeros([1])
fov_deg = 5
npixel = 1024  # 512  # 384 #  128 * 2
npoints = 200
nufft_eps = 1e-3

lambda_factor = .01
lambda_factor_pcp = lambda_factor

clean_iter = 500
clean_gain = 0.7
diagnostics = False

eps = 1e-4
tmax = 120.
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
        "overtime_lsr": 0.2,
        "rate_tk": 0.,
        "eps_lsr": eps,
    }
    clean_parameters = {
        "niter": clean_iter,
        "threshold": 0.001,
        "fractional_threshold": 0.001,
        "window_shape": "quarter",
        "gain": clean_gain,
        "algorithm": 'hogbom',
    }
    context = 'ng'
    fit_parameters_pc = {
        "stop_crit": stop_crit,
        "positivity_constraint": True,
        "diff_lipschitz": fOp_lipschitz ** 2
    }

    ## Computations
    # PolyCLEAN
    pclean = pc.PolyCLEAN(
        flagged_uvwlambda,
        direction_cosines,
        measurements,
        lambda_=lambda_,
        **pclean_parameters,
    )
    print("PolyCLEAN: Solving...")
    pclean_time = time.time()
    pclean.fit(**fit_parameters_pc)
    print("\tSolved in {:.3f} seconds".format(time.time() - pclean_time))
    if diagnostics:
        pclean.diagnostics(log=False)
    data_pc, hist_pc = pclean.stats()

    # PolyCLEAN+
    s = reco.stop_crit(tmax=hist_pc["duration"][-1] * pcleanp_parameters.get("overtime_lsr", .2),
                       min_iter=pcleanp_parameters.get("min_iter_lsr", 5),
                       eps=pcleanp_parameters.get("eps_lsr", 1e-4))
    sol_pc = data_pc["x"]
    support = np.nonzero(sol_pc)[0]  # todo test np.where(np.abs(sol) > 1e-3)[0]
    rs_forwardOp = pclean.rs_forwardOp(support)
    rs_data_fid = .5 * pxop.SquaredL2Norm(dim=rs_forwardOp.shape[0]).argshift(-measurements) * rs_forwardOp
    rate_tk = pcleanp_parameters.get("rate_lsr", 0.)
    if rate_tk > 0.:
        rs_data_fid = rs_data_fid + 0.5 * rate_tk * fit_parameters_pc["diff_lipschitz"] * pxop.SquaredL2Norm(
            dim=rs_forwardOp.shape[1])
    rs_regul = pxop.PositiveOrthant(dim=rs_forwardOp.shape[1])
    lsr_apgd = pxsol.PGD(rs_data_fid, rs_regul, show_progress=False)
    print("Least squares reweighting (PolyCLEAN+): ...")
    lsr_apgd.fit(x0=sol_pc[support],
                 stop_crit=s,
                 track_objective=True,
                 tau=1 / (fit_parameters_pc["diff_lipschitz"] * (1 + rate_tk)))
    data_pcp, hist_lsr = lsr_apgd.stats()
    sol_pcp = np.zeros_like(sol_pc)
    sol_pcp[support] = data_pcp['x']
    print("\tSolved in {:.3f} seconds ({:d} iterations)".format(hist_lsr['duration'][-1],
                                                                int(hist_lsr['N_iter'][-1])))

    # APGD
    fit_parameters_apgd = {
        "x0": np.zeros(forwardOp.shape[1], dtype="float64"),
        "stop_crit": reco.stop_crit(tmax, min_iter, eps, value=hist_pc["Memorize[objective_func]"][-1]),
        "track_objective": True,
        "tau": 1 / (fOp_lipschitz ** 2)
    }
    # forwardOp = pc.generatorVisOp(direction_cosines,
    #                               flagged_uvwlambda,
    #                               nufft_eps)
    data_fid_synth = 0.5 * pxop.SquaredL2Norm(dim=forwardOp.shape[0]).argshift(-measurements) * forwardOp
    regul_synth = lambda_ * pyfwl.L1NormPositivityConstraint(shape=(1, forwardOp.shape[1]))
    apgd = pxsol.PGD(data_fid_synth, regul_synth, show_progress=False)
    print("APGD: Solving ...")
    start = time.time()
    apgd.fit(**fit_parameters_apgd)
    print("\tSolved in {:.3f} seconds".format(time.time() - start))
    data_apgd, hist_apgd = apgd.stats()
    dcv = abs(forwardOp.adjoint(measurements - forwardOp(data_apgd["x"]))).max() / lambda_
    data_apgd["dcv"] = dcv

    # CLEAN
    cellsize = abs(sky_im.coords["x"].data[1] - sky_im.coords["x"].data[0])
    npixel = sky_im.dims["x"]

    predicted_visi = predict_visibility(vt, sky_im, context=context)
    image_model = create_image_from_visibility(predicted_visi, cellsize=cellsize, npixel=npixel)

    clean_model = create_image_from_visibility(predicted_visi, cellsize=cellsize, npixel=2 * npixel)
    dirty, sumwt_dirty = invert_visibility(predicted_visi, clean_model, context=context)
    psf, sumwt = invert_visibility(predicted_visi, image_model, context=context, dopsf=True)
    print("CLEAN: Solving...")
    start = time.time()
    tmp_clean_comp, tmp_clean_residual = deconvolve_cube(
        dirty,
        psf,
        **clean_parameters,
    )
    clean_comp = image_model.copy(deep=True)
    clean_comp['pixels'].data[0, 0, ...] = \
        tmp_clean_comp['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]
    clean_residual = image_model.copy(deep=True)
    clean_residual['pixels'].data[0, 0, ...] = \
        tmp_clean_residual['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]
    clean_duration = time.time() - start
    print("\tSolved in {:.3f} seconds".format(clean_duration))

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
    print("Iterations: {} + {}".format(int(hist_pc['N_iter'][-1]), int(hist_lsr['N_iter'][-1])))
    print("Final sparsity: {}".format(np.count_nonzero(sol_pcp)))
    print("\n")

    # Final sparsity
    print("Final sparsity CLEAN: {}".format(np.count_nonzero(clean_comp.pixels.data)))
    print("Final sparsity APGD: {}".format(np.count_nonzero(data_apgd["x"])))

    ### Quality of the reconstructions

    pclean_comp = sky_im.copy(deep=True)
    pclean_comp.pixels.data[0, 0] = data_pc["x"].reshape((npixel,) * 2)
    pcleanp_comp = sky_im.copy(deep=True)
    pcleanp_comp.pixels.data[0, 0] = sol_pcp.reshape((npixel,) * 2)
    apgd_comp = sky_im.copy(deep=True)
    apgd_comp.pixels.data[0, 0] = data_apgd["x"].reshape((npixel,) * 2)

    psf, sumwt = invert_visibility(vt, sky_im, context="ng", dopsf=True)
    clean_beam = fit_psf(psf)
    """
    The clean beam is saved in deg, deg, deg, accounting for, after conversion in pixels, the xstd, ystd and rotation of
    the kernel. The functions are then normalised in amplitude.
    Sharper kernel implies reducing both the first two components of clean_beam.
    """

    dirty = sky_im.copy(deep=True)
    dirty.pixels.data[0, 0] = dirty_image.reshape((npixel,) * 2) / (forwardOp.shape[0] / 2)

    components = [pclean_comp, pcleanp_comp, apgd_comp, clean_comp]

    sky_im_restored = restore_cube(sky_im, None, None, clean_beam)
    restored_comps = restore_list(components, None, None, clean_beam)

    timings = [lipschitz_time + hist_pc["duration"][-1],
               lipschitz_time + hist_pc["duration"][-1] + hist_lsr["duration"][-1],
               lipschitz_time + hist_apgd["duration"][-1],
               clean_duration,
               0.]
    sparsity = [np.count_nonzero(data_pc["x"]),
                np.count_nonzero(data_pcp["x"]),
                np.count_nonzero(data_apgd["x"]),
                np.count_nonzero(clean_comp.pixels.data), ]
    mse_comp = [ut.MSE(sky_im, im) for im in components] + [ut.MSE(sky_im, dirty), ]
    mse = [ut.MSE(sky_im_restored, im) for im in restored_comps] + [ut.MSE(sky_im_restored, dirty), ]
    cb_sharp = clean_beam.copy()
    cb_sharp["bmin"] = clean_beam["bmin"] / np.sqrt(2)
    cb_sharp["bmaj"] = clean_beam["bmaj"] / np.sqrt(2)
    sky_im_sharp = restore_cube(sky_im, None, None, cb_sharp)
    sharp_comps = restore_list(components, None, None, cb_sharp)
    mse_sharp = [ut.MSE(sky_im_sharp, im) for im in sharp_comps] + [ut.MSE(sky_im_sharp, dirty), ]
    cb_vsharp = clean_beam.copy()
    cb_vsharp["bmin"] = clean_beam["bmin"] / 2
    cb_vsharp["bmaj"] = clean_beam["bmaj"] / 2
    sky_im_vsharp = restore_cube(sky_im, None, None, cb_vsharp)
    vsharp_comps = restore_list(components, None, None, cb_vsharp)
    mse_vsharp = [ut.MSE(sky_im_vsharp, im) for im in vsharp_comps] + [ut.MSE(sky_im_vsharp, dirty), ]
    cb_wide = clean_beam.copy()
    cb_wide["bmin"] = clean_beam["bmin"] * np.sqrt(2)
    cb_wide["bmaj"] = clean_beam["bmaj"] * np.sqrt(2)
    sky_im_wide = restore_cube(sky_im, None, None, cb_wide)
    wide_comps = restore_list(components, None, None, cb_wide)
    mse_wide = [ut.MSE(sky_im_wide, im) for im in wide_comps] + [ut.MSE(sky_im_wide, dirty), ]

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
    table = pd.DataFrame(df, columns=methods + ["Dirty Im"],
                         index=["Time", "Sparsity", "MSE wide", "MSE clean beam", "MSE sharp", "MSE very sharp",
                                "MSE components", "Sum"])
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

    i = 0
    ut.display_image_error(sky_im_restored, dirty, restored_comps[-1], restored_comps[i],
                           titles=["CLEAN", methods[i]], sc=sc,
                           suptitle="Components convolved with synthetic beam",
                           normalize=True, cm="cubehelix_r", sc_marker=".", sc_color="k", sc_size=5)
    ut.display_image_error(sky_im_vsharp, dirty, vsharp_comps[-1], vsharp_comps[i],
                           titles=["CLEAN", methods[i]], sc=sc,
                           suptitle="Components convolved with very sharp beam",
                           normalize=True, cm="cubehelix_r", sc_marker=".", sc_color="k", sc_size=5)
    # ut.display_image_error(sky_im_restored, dirty, restored_comps[-1], restored_comps[0],
    #                        titles=["CLEAN", methods[0]], sc=sc,
    #                        suptitle="Components convolved with synthetic beam",
    #                        normalize=True, cm="cubehelix_r", sc_marker=".", sc_color="k", sc_size=5)

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
    # ut.qq_plot_point_sources_stacked(sky_im, clean_comp, pclean_comp, pcleanp_comp, title="rmax = {}".format(rmax))
    ## Components
    # ut.plot_image(pclean_comp, sc=sc, title="pCLEAN")
    # ut.plot_image(pcleanp_comp, sc=sc, title="pCLEAN+")
    # ut.plot_image(apgd_comp, sc=sc, title="APGD")
    # ut.plot_image(clean_comp, sc=sc, title="CLEAN")
    # todo plot components on the same graph

    bayes_test = False
    if bayes_test:
        vis_from_solution = forwardOp(pclean_comp.pixels.data[0, 0].flatten())
        sigma = (measurements - vis_from_solution).std()/2
        sigma_bis2 = ((measurements - vis_from_solution) ** 2).sum() \
                     / (measurements.shape[0] - np.count_nonzero(pclean_comp.pixels.data[0, 0]))
        snrdb = 20 * np.log10(vis_from_solution.std() / sigma)
        print(f"A posteri estimated sigma: {sigma:.2e}")
        print(f"Corresponding SNR: {snrdb:.2f}dB")

        min_map = pclean.objective_func() / (sigma ** 2)
        alpha = np.linspace(0, 1, 100, endpoint=False)[1:]
        tau_alpha = 4 * np.sqrt(np.log(3 / alpha))
        gamma_alpha = min_map + tau_alpha * npixel + npixel ** 2

        plt.figure()
        plt.scatter(alpha, gamma_alpha)
        plt.hlines(min_map, alpha[0], alpha[-1], colors='red', label="value MAP")
        plt.legend()
        plt.show()

        # ut.plot_image(sharp_comps[1])
        # ut.plot_image(pclean_comp)

        solution = pclean_comp.pixels.data[0, 0]
        feature1 = solution[632:654, 565:590]  # high intensity component
        feature2 = solution[639:652, 274:285]  # medium intensity
        # plt.figure()
        # plt.imshow(feature2, origin="lower", cmap='cubehelix_r')
        # plt.show()

        hypothesis1 = solution.copy()
        hypothesis1[632:654, 565:590] = 0.
        value_hypothesis1 = (pclean._data_fidelity(hypothesis1.flatten()) +
                             pclean._penalty(hypothesis1.flatten())) / sigma ** 2

        hypothesis2 = solution.copy()
        hypothesis2[639:652, 274:285] = 0.
        value_hypothesis2 = (pclean._data_fidelity(hypothesis2.flatten()) +
                             pclean._penalty(hypothesis2.flatten())) / sigma ** 2

        empty = np.zeros_like(solution)
        value_empty = (pclean._data_fidelity(empty.flatten()) +
                       pclean._penalty(empty.flatten())) / sigma ** 2
        plt.figure()
        plt.scatter(alpha, gamma_alpha, label="threshold", marker='x', s=1)
        plt.hlines(min_map, alpha[0], alpha[-1], colors='red', label="value MAP")
        plt.hlines(value_hypothesis1, alpha[0], alpha[-1], colors='blue', label="value test 1")
        plt.hlines(value_hypothesis2, alpha[0], alpha[-1], colors='green', label="value test 2")
        plt.hlines(value_empty, alpha[0], alpha[-1], colors='orange', label="empty image")
        plt.legend()
        plt.yscale('linear')
        plt.xlabel("alpha")
        plt.show()

    save_images = False
    if save_images:
        sum_vis = (vt.weight.data > 0).sum()
        pclean_residual = forwardOp.adjoint(measurements - forwardOp(sol_pc))
        pclean_residual_im = image_model.copy(deep=True)
        pclean_residual_im.pixels.data = pclean_residual.reshape(pclean_residual_im.pixels.data.shape) / sum_vis
        pcleanp_residual = forwardOp.adjoint(measurements - forwardOp(sol_pcp))
        pcleanp_residual_im = image_model.copy(deep=True)
        pcleanp_residual_im.pixels.data = pcleanp_residual.reshape(pclean_residual_im.pixels.data.shape) / sum_vis

        pclean_restored = restore_cube(pclean_comp, None, pclean_residual_im, clean_beam=cb_sharp)
        pcleanp_restored = restore_cube(pcleanp_comp, None, pcleanp_residual_im, clean_beam=cb_sharp)
        clean_restored = restore_cube(clean_comp, None, clean_residual, clean_beam=clean_beam)


        folder_path = "/home/jarret/PycharmProjects/polyclean/scripts/simulations_ps/reco_fits/"
        pclean_restored.image_acc.export_to_fits(folder_path + "pclean_restored_{}.fits".format(lambda_factor))
        pclean_residual_im.image_acc.export_to_fits(folder_path + "pclean_residual_{}.fits".format(lambda_factor))
        pcleanp_restored.image_acc.export_to_fits(folder_path + "pcleanp_restored_{}.fits".format(lambda_factor))
        pcleanp_residual_im.image_acc.export_to_fits(folder_path + "pcleanp_residual_{}.fits".format(lambda_factor))
        clean_restored.image_acc.export_to_fits(folder_path + "clean_restored_{:d}.fits".format(clean_iter))
        clean_residual.image_acc.export_to_fits(folder_path + "clean_residual_{:d}.fits".format(clean_iter))

        # Export sc
        src_ra_dec_flux = np.array([[s.direction.ra.value, s.direction.dec.value, s.flux[0, 0]] for s in sc]).T
        src_ra_dec_flux[2, :] /= src_ra_dec_flux[2].max()
        np.savez(folder_path + "sources.npz", src_ra_dec_flux=src_ra_dec_flux)