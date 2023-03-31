import matplotlib.pyplot as plt
import numpy as np
import time
from astropy import units as u
from astropy.coordinates import SkyCoord

from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
# Visualization
from ska_sdp_func_python.imaging.imaging import invert_visibility
from ska_sdp_func_python.image.deconvolution import restore_cube, fit_psf

import polyclean.reconstructions as reco
import polyclean.image_utils as ut
import polyclean.polyclean as pc

import matplotlib
matplotlib.use("Qt5Agg")


seed = 64  # np.random.randint(0, 1000)  # np.random.randint(0, 1000)  # 492
rmax = 500.  # 2000.
times = np.zeros([1])
fov_deg = 5
npixel = 512  # 512  # 384 #  128 * 2
npoints = 200
nufft_eps = 1e-3

lambda_factor = .05

eps = 1e-4
tmax = 240.
min_iter = 5
ms_threshold = 0.8
init_correction_prec = 5e-2
final_correction_prec = 1e-4
remove = True
min_correction_steps = 3
lock = False
lock_steps = 6

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
    fOp_lipschitz = forwardOp.lipschitz(tol=1., tight=True)
    lipschitz_time = time.time() - start
    print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)".format(lipschitz_time))

    measurements = forwardOp(sky_im.pixels.data.reshape(-1))
    dirty_image = forwardOp.adjoint(measurements)

    ### Reconsruction
    # Parameters
    lambda_ = lambda_factor * np.abs(dirty_image).max()
    stop_crit = reco.stop_crit(tmax, min_iter, eps)
    if lock:
        min_correction_steps = lock_steps
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
    data, hist = reco.reco_pclean(flagged_uvwlambda, direction_cosines, measurements, lambda_,
                                  pclean_parameters, fit_parameters)

    ### Results
    print("PolyCLEAN final DCV: {:.3f}".format(data["dcv"]))
    print("Iterations: {}".format(int(hist['N_iter'][-1])))
    print("Final sparsity: {}".format(np.count_nonzero(data["x"])))



    pclean_comp = sky_im.copy(deep=True)
    pclean_comp.pixels.data[0, 0] = data["x"].reshape((npixel, )*2)
    psf, sumwt = invert_visibility(vt, sky_im, context="ng", dopsf=True)
    clean_beam = fit_psf(psf)
    pclean_restored = restore_cube(pclean_comp, None, None, clean_beam)
    sky_im_restored = restore_cube(sky_im, None, None, clean_beam)
    # print("MSE with convolved source: {:.3e}".format(ut.MSE(sky_im_restored, pclean_restored)[0, 0]))

    ut.plot_source_reco_diff(sky_im_restored, pclean_restored, title="PolyCLEAN Convolved", suptitle="Comparison: lock", sc=sc)

    # ut.compare_3_images(sky_im_restored, pclean_comp, pclean_restored, titles=["components", "convolution"], sc=sc)

    from ska_sdp_func_python.imaging import predict_visibility
    predicted_visi = predict_visibility(vt, sky_im, context="ng")
    dirty_rascil, _ = invert_visibility(predicted_visi, sky_im, context="ng", dopsf=False, normalise=True)
    # print("MSE dirty image: {:.3e}".format(ut.MSE(sky_im_restored, dirty_rascil)[0, 0]))
    print("Errors:\n\tDirty image: {:.2e}\n\tCLEAN image: {:.2e}\n\tCLEAN components: {:.2e}".format(
        ut.MSE(dirty_rascil, sky_im_restored)[0][0],
        ut.MSE(sky_im_restored, sky_im_restored)[0][0],
        ut.MSE(sky_im, pclean_comp)[0][0]))

    # print(np.allclose(dirty_image/dirty_image.max(), dirty_rascil.pixels.data.flatten()/dirty_image.max()))

    # ut.plot_image(dirty_rascil, title="Rascil")
    # dirty_copy = dirty_rascil.copy(deep=True)
    # dirty_copy.pixels.data = dirty_image.reshape((1, 1, npixel, npixel))
    # ut.plot_image(dirty_copy, title="Nufft")
    # diff = dirty_rascil.copy(deep=True)
    # diff.pixels.data = (dirty_image.reshape((1, 1, npixel, npixel)) - dirty_rascil.pixels.data)/dirty_image.max()
    # ut.plot_image(diff, title="Diff", cmap="bwr")

    import matplotlib.pyplot as plt
    fluxs = np.array([comp.flux[0,0] for comp in sc])
    fluxs /= fluxs.max()
    plt.figure()
    plt.hist(fluxs, bins=50)
    plt.show()
