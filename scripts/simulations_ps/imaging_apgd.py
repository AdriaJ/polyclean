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

import pyxu.operator as pxop
import pyxu.opt.solver as pxsol

import pyfwl

import polyclean.reconstructions as reco
import polyclean.image_utils as ut
import polyclean.polyclean as pc

# matplotlib.use("Qt5Agg")


seed = 64  # np.random.randint(0, 1000)  # np.random.randint(0, 1000)  # 492
rmax = 800.  # 2000.
times = (np.arange(7)-3) * np.pi/9  # 7 angles from -pi/3 to pi/3
fov_deg = 5
npixel = 1024  # 512  # 384 #  128 * 2
npoints = 200
nufft_eps = 1e-3
psnrdb = 20

lambda_factor = .01

eps = 1e-5
tmax = 240.
min_iter = 5

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
    print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)".format(lipschitz_time))

    noiseless_measurements = forwardOp(sky_im.pixels.data.reshape(-1))
    noise_scale = np.abs(noiseless_measurements).max() * 10 ** (-psnrdb / 20) / np.sqrt(2)
    noise = np.random.normal(0, noise_scale, noiseless_measurements.shape)
    measurements = noiseless_measurements + noise

    dirty_image = forwardOp.adjoint(measurements)

    ### Reconsruction
    # Parameters
    lambda_ = lambda_factor * np.abs(dirty_image).max()
    stop_crit = reco.stop_crit(tmax, min_iter, eps, value=401674.)
    apgd_parameters = {"nufft_eps": nufft_eps}
    fit_parameters = {
        "x0": np.zeros(forwardOp.shape[1], dtype="float64"),
        "stop_crit": reco.stop_crit(tmax, min_iter, eps, ),
        "track_objective": True,
        "tau": 1 / (fOp_lipschitz ** 2)
    }

    # Solving
    data_fid_synth = 0.5 * pxop.SquaredL2Norm(dim=forwardOp.shape[0]).argshift(-measurements) * forwardOp
    regul_synth = lambda_ * pyfwl.L1NormPositivityConstraint(shape=(1, forwardOp.shape[1]))
    apgd = pxsol.PGD(data_fid_synth, regul_synth, show_progress=False)
    print("APGD: Solving ...")
    start = time.time()
    apgd.fit(**fit_parameters)
    print("\tSolved in {:.3f} seconds".format(dt := time.time() - start))
    sol, hist = apgd.stats()
    dcv = abs(forwardOp.adjoint(measurements - forwardOp(sol["x"]))).max() / lambda_
    sol["dcv"] = dcv
    print("Dual certificate value at convergence: {:.3f}".format(dcv))
    print(f"Final value of the objective function: {hist['Memorize[objective_func]'][-1]:.3e}")
    print("Final sparsity of the components: {}".format(np.count_nonzero(sol["x"])))


    # Evaluation of the reconstruction
    components = sky_im.copy(deep=True)
    components.pixels.data[0, 0] = sol["x"].reshape((npixel,) * 2)
    psf, sumwt = invert_visibility(vt, sky_im, context="ng", dopsf=True)
    clean_beam = fit_psf(psf)
    comps_convolved = restore_cube(components, None, None, clean_beam)
    sky_im_restored = restore_cube(sky_im, None, None, clean_beam)
    ut.plot_source_reco_diff(sky_im_restored, comps_convolved, title="APGD Convolved", sc=sc)

    from ska_sdp_func_python.imaging import predict_visibility
    predicted_visi = predict_visibility(vt, sky_im, context="ng")
    dirty_rascil, _ = invert_visibility(predicted_visi, sky_im, context="ng", dopsf=False, normalise=True)

    print("CLEAN beam (MSE/MAD):\n\tDirty image: {:.2e}/{:.2e}\n\tComponents convolved: {:.2e}/{:.2e}\n\tRaw components: {:.2e}/{:.2e}".format(
        ut.MSE(dirty_rascil, sky_im_restored), ut.MAD(dirty_rascil, sky_im_restored),
        ut.MSE(comps_convolved, sky_im_restored), ut.MAD(comps_convolved, sky_im_restored),
        ut.MSE(sky_im, components), ut.MAD(sky_im, components)
        )
    )

    sharp_beam = clean_beam.copy()
    sharp_beam["bmin"] = clean_beam["bmin"] / 2
    sharp_beam["bmaj"] = clean_beam["bmaj"] / 2
    comp_sharp = restore_cube(components, None, None, sharp_beam)
    sky_im_sharp = restore_cube(sky_im, None, None, sharp_beam)

    print("Sharp beam (MSE/MAD):\n\tDirty image: {:.2e}/{:.2e}\n\tComponents convolved: {:.2e}/{:.2e}".format(
        ut.MSE(dirty_rascil, sky_im_sharp), ut.MAD(dirty_rascil, sky_im_sharp),
        ut.MSE(comp_sharp, sky_im_sharp), ut.MAD(comp_sharp, sky_im_sharp),
        )
    )

    import matplotlib.pyplot as plt
    plt.figure()
    d = hist['RelError[objective_func]'][1:]
    plt.scatter(np.arange(d.shape[0]), d, marker='x')
    m = hist['Memorize[objective_func]'][1:]
    plt.scatter(np.arange(d.shape[0]), (m - m[-1])/(m[0] - m[-1]), marker='x')
    plt.yscale('log')
    plt.show()