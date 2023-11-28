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


# matplotlib.use("Qt5Agg")

seed = 64  # np.random.randint(0, 1000)  # np.random.randint(0, 1000)  # 492
rmax = 300.  # 2000.
times = (np.arange(7) - 3) * np.pi / 9  # 7 angles from -pi/3 to pi/3
fov_deg = 5
npixel = 720  # 512  # 384 #  128 * 2
npoints = 200
nufft_eps = 1e-3
chunked = False
psnrdb = 20

lambda_factor = .01

eps = 1e-4
tmax = 240.
min_iter = 5
ms_threshold = 0.8
init_correction_prec = 5e-2
final_correction_prec = min(1e-4, eps)
remove = True
min_correction_steps = 3
max_correction_steps = 1000
lock = False
lock_steps = 6
diagnostics = True
log_diagnostics = False

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
                                  nufft_eps=nufft_eps,
                                  chunked=chunked)
    start = time.time()
    fOp_lipschitz = forwardOp.estimate_lipschitz(method='svd', tol=10.)
    lipschitz_time = time.time() - start
    print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)".format(lipschitz_time))
    # print(fOp_lipschitz)

    noiseless_measurements = forwardOp(sky_im.pixels.data.reshape(-1))
    noise_scale = np.abs(noiseless_measurements).max() * 10 ** (-psnrdb / 20) / np.sqrt(2)
    noise = np.random.normal(0, noise_scale, noiseless_measurements.shape)
    measurements = noiseless_measurements + noise

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
        "max_correction_steps": max_correction_steps,
        "remove_positions": remove,
        "show_progress": False,
    }
    fit_parameters = {
        "stop_crit": stop_crit,
        "positivity_constraint": True,
        "diff_lipschitz": fOp_lipschitz ** 2,
        "lock_reweighting": lock,
        "precision_rule": lambda k: 10 ** (-k / 10),
    }

    # Computations
    pclean = pc.PolyCLEAN(
        data=measurements,
        uvwlambda=flagged_uvwlambda,
        direction_cosines=direction_cosines,
        lambda_=lambda_,
        chunked=chunked,
        nufft_eps=nufft_eps,
        **pclean_parameters
    )
    print("PolyCLEAN: Solving...")
    pclean_time = time.time()
    pclean.fit(**fit_parameters)
    print("\tSolved in {:.3f} seconds".format(dt_pclean := time.time() - pclean_time))
    if diagnostics:
        pclean.diagnostics(log=log_diagnostics)
    data, hist = pclean.stats()
    pclean_residual = forwardOp.adjoint(measurements - forwardOp(data["x"]))

    ### Results
    print("PolyCLEAN final DCV: {:.3f}".format(data["dcv"]))
    print(f"Final value of the objective function: {hist['Memorize[objective_func]'][-1]:.3e}")
    print("Iterations: {}".format(int(hist['N_iter'][-1])))
    print("Final sparsity of the components: {}".format(np.count_nonzero(data["x"])))

    pclean_comp = sky_im.copy(deep=True)
    pclean_comp.pixels.data[0, 0] = data["x"].reshape((npixel,) * 2)
    psf, sumwt = invert_visibility(vt, sky_im, context="ng", dopsf=True)
    clean_beam = fit_psf(psf)
    pclean_restored = restore_cube(pclean_comp, None, None, clean_beam)
    sky_im_restored = restore_cube(sky_im, None, None, clean_beam)
    pclean_residual_im = sky_im.copy(deep=True)
    pclean_residual_im.pixels.data = pclean_residual.reshape(pclean_residual_im.pixels.data.shape) / (
                measurements.shape[0] // 2)

    ut.plot_source_reco_diff(sky_im_restored, pclean_restored, title="PolyCLEAN Convolved", suptitle="Comparison",
                             sc=sc)

    # ut.compare_3_images(sky_im_restored, pclean_comp, pclean_restored, titles=["components", "convolution"], sc=sc)

    from ska_sdp_func_python.imaging import predict_visibility

    predicted_visi = predict_visibility(vt, sky_im, context="ng")
    dirty_rascil, _ = invert_visibility(predicted_visi, sky_im, context="ng", dopsf=False, normalise=True)

    print(
        "CLEAN beam (MSE/MAD):\n\tDirty image: {:.2e}/{:.2e}\n\tComponents convolved: {:.2e}/{:.2e}\n\tRaw components: {:.2e}/{:.2e}".format(
            ut.MSE(dirty_rascil, sky_im_restored), ut.MAD(dirty_rascil, sky_im_restored),
            ut.MSE(pclean_restored, sky_im_restored), ut.MAD(pclean_restored, sky_im_restored),
            ut.MSE(sky_im, pclean_comp), ut.MAD(sky_im, pclean_comp)
        )
    )

    sharp_beam = clean_beam.copy()
    sharp_beam["bmin"] = clean_beam["bmin"] / 2
    sharp_beam["bmaj"] = clean_beam["bmaj"] / 2
    pclean_comp_sharp = restore_cube(pclean_comp, None, None, sharp_beam)
    sky_im_sharp = restore_cube(sky_im, None, None, sharp_beam)

    print("Sharp beam (MSE/MAD):\n\tDirty image: {:.2e}/{:.2e}\n\tComponents convolved: {:.2e}/{:.2e}".format(
        ut.MSE(dirty_rascil, sky_im_sharp), ut.MAD(dirty_rascil, sky_im_sharp),
        ut.MSE(pclean_comp_sharp, sky_im_sharp), ut.MAD(pclean_comp_sharp, sky_im_sharp),
    )
    )

    # print(np.allclose(dirty_image/dirty_image.max(), dirty_rascil.pixels.data.flatten()/dirty_image.max()))

    # ut.plot_image(dirty_rascil, title="Rascil")
    # dirty_copy = dirty_rascil.copy(deep=True)
    # dirty_copy.pixels.data = dirty_image.reshape((1, 1, npixel, npixel))
    # ut.plot_image(dirty_copy, title="Nufft")
    # diff = dirty_rascil.copy(deep=True)
    # diff.pixels.data = (dirty_image.reshape((1, 1, npixel, npixel)) - dirty_rascil.pixels.data)/dirty_image.max()
    # ut.plot_image(diff, title="Diff", cmap="bwr")

    # import matplotlib.pyplot as plt
    # fluxs = np.array([comp.flux[0,0] for comp in sc])
    # fluxs /= fluxs.max()
    # plt.figure()
    # plt.hist(fluxs, bins=50)
    # plt.show()

    from scripts.observations.pclean import plot_1_image
    sharp_beam = clean_beam.copy()
    sharp_beam["bmin"] = clean_beam["bmin"] / 10
    sharp_beam["bmaj"] = clean_beam["bmaj"] / 10
    test_sky_im = restore_cube(sky_im, None, None, sharp_beam)
    # test_sky_im['pixels'].data = 1000 * test_sky_im['pixels'].data

    def plot_1_image(image, title="", cmaps=['hot', 'Greys'], alpha=.95, offset_cm=0., symm=True):
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import matplotlib.colors as mplc
        arr = image.pixels.data[0, 0]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.subplots(1, 1, subplot_kw={'projection': image.image_acc.wcs.sub([1, 2]), 'frameon': False})
        ax.set_xlabel(image.image_acc.wcs.wcs.ctype[0])
        ax.set_ylabel(image.image_acc.wcs.wcs.ctype[1])
        # vlim = -arr.min() if symm else 0.
        # mask_comp = np.ma.masked_array(arr, arr < vlim, fill_value=vlim)
        aximc = ax.imshow(arr, origin="lower", cmap='hot', interpolation='none', alpha=alpha,
                          norm=mplc.PowerNorm(gamma=0.5, vmin=0., vmax=None))
        axinsc = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
        cbc = fig.colorbar(aximc, cax=axinsc, orientation="vertical")  # , ticks=[round(0) + 1, 500, 1000, 2000, 3000, 4000])
        fig.suptitle(title)
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.0, right=0.93, hspace=0.15, wspace=0.15)
        fig.show()

    # plot_1_image(test_sky_im)
    # import os
    # folder_path = os.path.join("/home/jarret/PycharmProjects/polyclean/scripts/simulations_ps", 'source')
    # # create a folder if it does not exist
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    # plt.savefig(os.path.join(folder_path, 'source.png'))