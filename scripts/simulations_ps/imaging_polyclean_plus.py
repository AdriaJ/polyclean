import numpy as np
import time
from astropy import units as u
from astropy.coordinates import SkyCoord

from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame

import polyclean.reconstructions as reco
import polyclean.image_utils as ut
import polyclean.polyclean as pc

import pyxu.operator as pxop
import pyxu.opt.solver as pxsol

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


seed = 248  # np.random.randint(0, 1000)
rmax = 500.  # 2000.
times = np.zeros([1])
fov_deg = 5
npixel = 512  # 512  # 256
npoints = 200
nufft_eps = 1e-3

lambda_factor = .05

eps = 1e-3
tmax = 240.
min_iter = 5
ms_threshold = 0.8
init_correction_prec = 5e-2
final_correction_prec = min(eps, 1e-4)
remove = True
min_correction_steps = 3

diagnostics=True

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(1000)
    print("Seed: {}".format(seed))
    rng = np.random.default_rng(seed)

    ### Simulation of the source

    frequency = np.array([1e8])
    channel_bandwidth = np.array([1e6])
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000")
    sky_im, sc = ut.generate_point_sources(npoints,
                                           fov_deg,
                                           npixel,
                                           flux_sigma=.4,
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
    fOp_lipschitz = forwardOp.lipschitz(tol=1., tight=True)
    lipschitz_time = time.time() - start
    print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)".format(lipschitz_time))

    measurements = forwardOp(sky_im.pixels.data.reshape(-1))
    dirty_image = forwardOp.adjoint(measurements)

    ### Reconsruction
    # Parameters
    lambda_ = lambda_factor * np.abs(dirty_image).max()
    stop_crit = reco.stop_crit(tmax, min_iter, eps)
    pcleanp_parameters = {
        "lambda_factor": lambda_factor,
        "ms_threshold": ms_threshold,
        "init_correction_prec": init_correction_prec,
        "final_correction_prec": final_correction_prec,
        "min_correction_steps": min_correction_steps,
        "remove_positions": remove,
        "nufft_eps": nufft_eps,
        "show_progress": False,
        "overtime_lsr": 0.2,
        "rate_tk": 0.,
    }
    fit_parameters = {
        "stop_crit": stop_crit,
        "positivity_constraint": True,
        "diff_lipschitz": fOp_lipschitz ** 2,
    }

    # Computations
    pclean = pc.PolyCLEAN(
        flagged_uvwlambda,
        direction_cosines,
        measurements,
        lambda_=lambda_,
        **pcleanp_parameters,
    )
    print("PolyCLEAN: Solving...")
    pclean_time = time.time()
    pclean.fit(**fit_parameters)
    print("\tSolved in {:.3f} seconds".format(time.time() - pclean_time))
    if diagnostics:
        pclean.diagnostics()
    solution, hist = pclean.stats()

    s = reco.stop_crit(tmax=hist["duration"][-1] * pcleanp_parameters.get("overtime_lsr", .2),
                  min_iter=pcleanp_parameters.get("min_iter_lsr", 5),
                  eps=pcleanp_parameters.get("eps_lsr", 1e-4))

    sol = solution["x"]
    support = np.nonzero(sol)[0]
    rs_forwardOp = pclean.rs_forwardOp(support)
    rs_data_fid = .5 * pxop.SquaredL2Norm(dim=rs_forwardOp.shape[0]).argshift(-measurements) * rs_forwardOp
    rate_tk = pcleanp_parameters.get("rate_lsr", 0.)
    if rate_tk > 0.:
        rs_data_fid = rs_data_fid + 0.5 * rate_tk * fit_parameters["diff_lipschitz"] * pxop.SquaredL2Norm(dim=rs_forwardOp.shape[1])
    rs_regul = pxop.PositiveOrthant(dim=rs_forwardOp.shape[1])
    lsr_apgd = pxsol.PGD(rs_data_fid, rs_regul, show_progress=False)
    print("Least squares reweighting:")
    lsr_apgd.fit(x0=sol[support],
                 stop_crit=s,
                 track_objective=True,
                 tau=1 / (fit_parameters["diff_lipschitz"] * (1 + rate_tk)))
    data_lsr, hist_lsr = lsr_apgd.stats()
    print("\tSolved in {:.3f} seconds ({:d} iterations)".format(hist_lsr['duration'][-1],
                                                              int(hist_lsr['N_iter'][-1])))
    res = np.zeros_like(sol)
    res[support] = data_lsr["x"]
    solution["x_old"] = solution.pop("x")
    solution["x"] = res

    ### Results
    print("PolyCLEAN final DCV (before post processing): {:.3f}".format(solution["dcv"]))
    print("Iterations: {}".format(int(hist['N_iter'][-1])))
    print("Final sparsity: {}".format(np.count_nonzero(solution["x"])))
    print("Value of the LASSO objective function:\n\t- PolyCLEAN: {:.3f}\n\t- PolyCLEAN+: {:.3f}".format(
        pclean._data_fidelity(solution["x_old"])[0] +  pclean._penalty(solution["x_old"])[0],
        pclean._data_fidelity(solution["x"])[0] + pclean._penalty(solution["x"])[0]
    ))

    # Visualization
    from ska_sdp_func_python.imaging import invert_visibility
    from ska_sdp_func_python.image import restore_cube, fit_psf

    pcleanp_comp = sky_im.copy(deep=True)
    pcleanp_comp.pixels.data[0, 0] = solution["x"].reshape((npixel,) * 2)
    psf, sumwt = invert_visibility(vt, sky_im, context="ng", dopsf=True)
    clean_beam = fit_psf(psf)
    pcleanp_restored = restore_cube(pcleanp_comp, None, None, clean_beam)
    sky_im_restored = restore_cube(sky_im, None, None, clean_beam)

    print("MSE: {:.4e}".format(ut.MSE(sky_im_restored, pcleanp_restored)[0, 0]))

    ut.plot_source_reco_diff(sky_im_restored, pcleanp_restored, title="PolyCLEAN+ Convolved", suptitle="Comparison", sc=sc)

    # dual = forwardOp.adjoint(measurements - forwardOp(data["x_old"]))/lambda_
    # dual_im = sky_im.copy(deep=True)
    # dual_im.pixels.data[0, 0] = dual.reshape((npixel,) * 2)
    # ut.plot_certificate(dual_im, sc=sc, title="Dual certificate at convergence", level=1.)

    # ut.compare_3_images(sky_im_restored, pclean_comp, pclean_restored, titles=["components", "convolution"], sc=sc)

    ut.plot_image(pcleanp_comp, sc=sc, title="PolyCLEAN+ Components")

    # import pyxu.operator as pxop
    # op = pxop.NUFFT.type3(x=np.array([]).reshape((0, 3)),  # direction_cosines,
    #                        z=2 * np.pi * flagged_uvwlambda,
    #                        real=True, isign=-1, eps=nufft_eps,
    #                        chunked=True,
    #                        parallel=False,
    #                        )
    # x_chunks, z_chunks = op.auto_chunk(max_mem=10)
    # op.allocate(x_chunks, z_chunks, direct_eval_threshold=10_000)
    # op.diagnostic_plot("z")

    ## Bayesian tests Highest Posterior Density
    alpha = .05

    tau_a = 4. * np.sqrt(np.log(3/alpha))
    approx_HPD_thresh = hist["Memorize[objective_func]"][-1] + tau_a * npixel + npixel**2

    x_srg = solution["x_old"].copy().reshape((npixel,)*2)
    x_srg[npixel//2-10:npixel//2+10, npixel//2-10:npixel//2+10] = 0.1
    pclean._data_fidelity(x_srg.flatten())[0] +  pclean._penalty(x_srg.flatten())[0]

    # TODO: try more tests, seems good thought + illustrate
