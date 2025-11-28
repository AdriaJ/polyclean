import os
import pickle
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
from ska_sdp_func_python.imaging import invert_visibility, create_image_from_visibility
from ska_sdp_func_python.image.deconvolution import restore_cube, fit_psf
from rascil.processing_components.visibility.base import export_visibility_to_ms


import pyxu.util.complex as pxc

import polyclean.reconstructions as reco
import polyclean.image_utils as ut
import polyclean.polyclean as pc
import polyclean.ra_utils as pcrau


# matplotlib.use("Qt5Agg")

srf = 10

seed = 64  # np.random.randint(0, 1000)  # np.random.randint(0, 1000)  # 492
rmax = 2000.  # 2000.
times = (np.arange(7) - 3) * np.pi / 9  # 7 angles from -pi/3 to pi/3
fov_deg = 3
npoints = 100
nufft_eps = 1e-3
chunked = False
psnrdb = 20

lambda_factor = .02

eps = 1e-4
tmax = 600.
min_iter = 5
ms_threshold = 0.8
init_correction_prec = 5e-2
final_correction_prec = min(1e-4, eps)
remove = True
min_correction_steps = 3
max_correction_steps = 1000
diagnostics = True
log_diagnostics = False

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(1000)
    print("Seed: {}".format(seed))
    rng = np.random.default_rng(seed)

    ### Loading of the configuration

    frequency = np.array([1e8])
    channel_bandwidth = np.array([1e6])
    phasecentre = SkyCoord(
        ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
    )

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

    ### Resolutions
    B = np.linalg.norm(uvwlambda, axis=1).max()
    nom_res = 1 / (3 * B)
    npix_low = np.ceil((fov_deg * np.pi / 180) / nom_res).astype(int)

    npix_high = pcrau.get_npixels(vt, fov_deg, phasecentre, 1e-3, srf=srf)
    high_res = fov_deg / npix_high * np.pi / 180

    print(f"Nominal resolution: {nom_res:.3e} rad, {npix_low} pixels")
    print(f"High resolution: {high_res:.3e} rad, {npix_high} pixels")
    from ska_sdp_func_python.imaging import advise_wide_field
    advice = advise_wide_field(vt, guard_band_image=6.0, delA=0.1, oversampling_synthesised_beam=3.0)
    print(f"Rascil advice: {advice['cellsize']:.3e} rad")

    ### Simulation of the source
    sky_ims = []
    d_cosines = []
    for npix in [npix_low, npix_high]:
        print(f"Number of pixels: {npix}")
        sky_im, sc = ut.generate_point_sources(npoints,
                                               fov_deg,
                                               npix,
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
        d_cosines.append(direction_cosines)
        sky_ims.append(sky_im)

    import polyclean.image_utils as ut
    ut.plot_image(sky_ims[0], title="Low resolution")
    ut.plot_image(sky_ims[1], title="High resolution")

    images_path = os.path.join(os.getcwd(), 'images')
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    sky_ims[0].image_acc.export_to_fits(os.path.join(images_path, 'source_low_res.fits'))
    sky_ims[1].image_acc.export_to_fits(os.path.join(images_path, 'source_high_res.fits'))

    ### Simulation of the measurements
    lowresOp = pc.generatorVisOp(direction_cosines=d_cosines[0],
                                  vlambda=flagged_uvwlambda,
                                  nufft_eps=nufft_eps,
                                  chunked=chunked)
    highresOp = pc.generatorVisOp(direction_cosines=d_cosines[1],
                                  vlambda=flagged_uvwlambda,
                                  nufft_eps=nufft_eps,
                                  chunked=chunked)
    start = time.time()
    lips_low = lowresOp.estimate_lipschitz(method='svd', tol=10.)
    lipschitz_time = time.time() - start
    print("Computation of the Lipschitz constant of the low resolution operator in: {:.3f} (s)".format(lipschitz_time))
    start = time.time()
    lips_high = highresOp.estimate_lipschitz(method='svd', tol=10.)
    lipschitz_time = time.time() - start
    print("Computation of the Lipschitz constant of the high resolution operator in: {:.3f} (s)".format(lipschitz_time))
    print(f"Lipschitz constants: \n\tLow resolution = {lips_low:.3e}\n\tHigh resolution = {lips_high:.3e}")

    noiseless_measurements = highresOp(sky_im.pixels.data.reshape(-1))
    noise_scale = np.abs(noiseless_measurements).max() * 10 ** (-psnrdb / 20) / np.sqrt(2)
    noise = np.random.normal(0, noise_scale, noiseless_measurements.shape)
    measurements = noiseless_measurements + noise

    dirty_lr = lowresOp.adjoint(measurements)
    dirty_hr = highresOp.adjoint(measurements)
    model_lr = create_image_from_visibility(vt, npixel=npix_low, cellsize=nom_res, override_cellsize=False)
    model_hr = create_image_from_visibility(vt, npixel=npix_high, cellsize=high_res, override_cellsize=False)

    dirty_im_lr = model_lr.copy(deep=True)
    dirty_im_lr.pixels.data[0, 0] = dirty_lr.reshape(dirty_im_lr.pixels.data.shape)
    dirty_im_hr = model_hr.copy(deep=True)
    dirty_im_hr.pixels.data[0, 0] = dirty_hr.reshape(dirty_im_hr.pixels.data.shape)

    ut.plot_image(dirty_im_lr, title="Low resolution dirty image")
    ut.plot_image(dirty_im_hr, title="High resolution dirty image")

    dirty_im_lr.image_acc.export_to_fits(os.path.join(images_path, 'dirty_low_res.fits'))
    dirty_im_hr.image_acc.export_to_fits(os.path.join(images_path, 'dirty_high_res.fits'))


    ### Save the visibilities
    visibility = vt.copy(deep=True)
    flag_shape = visibility.visibility_acc.flagged_weight.shape
    visibility.vis.data[flags_bool.reshape(flag_shape)] = pxc.view_as_complex(measurements)

    wsdir = os.path.join(os.getcwd(), 'wsclean-dir')
    if not os.path.exists(wsdir):
        os.makedirs(wsdir)
    filename = "ssms.ms"
    export_visibility_to_ms(os.path.join(wsdir, filename), [visibility], )

    with open(os.path.join(wsdir, 'vis.pkl'), 'wb') as handle:
        pickle.dump(visibility, handle)

    ### WSClean
    niter = 10_000
    thresh = 1
    print("WSClean low res: Solving...")
    start = time.time()
    os.system(
        f"wsclean -auto-threshold {thresh} -size {npix_low:d} {npix_low:d} -scale {fov_deg / npix_low:.6f} -mgain 0.7 "
        f"-niter {niter:d} -name ws-lr -weight natural -quiet wsclean-dir/ssms.ms")  # -no-dirty
    print("\tRun in {:.3f}s".format(dt_wsclean_lr := time.time() - start))
    os.system(f"mv ws-lr* wsclean-dir/")

    print("WSClean high res: Solving...")
    start = time.time()
    os.system(
        f"wsclean -auto-threshold {thresh} -size {npix_high:d} {npix_high:d} -scale {fov_deg / npix_high:.6f} -mgain 0.7 "
        f"-niter {niter:d} -name ws-hr -weight natural -quiet wsclean-dir/ssms.ms")  # -no-dirty
    print("\tRun in {:.3f}s".format(dt_wsclean_hr := time.time() - start))
    os.system(f"mv ws-hr* wsclean-dir/")

    ### PolyCLEAN
    # Parameters
    stop_crit = reco.stop_crit(tmax, min_iter, eps, dcv=1.2)  # todo : try with dual certificate for stop crit
    pclean_parameters = {
        "ms_threshold": ms_threshold,
        "init_correction_prec": init_correction_prec,
        "final_correction_prec": final_correction_prec,
        "min_correction_steps": min_correction_steps,
        "max_correction_steps": max_correction_steps,
        "remove_positions": remove,
        "show_progress": False,
    }
    print("PolyCLEAN low res: Setting up...")
    lambda_lr = lambda_factor * np.abs(dirty_lr).max()
    fit_parameters = {
        "stop_crit": stop_crit,
        "positivity_constraint": True,
        "diff_lipschitz": lips_low ** 2,
        "precision_rule": lambda k: 10 ** (-k / 10),
    }

    # Computations
    pclean = pc.PolyCLEAN(
        data=measurements,
        uvwlambda=flagged_uvwlambda,
        direction_cosines=d_cosines[0],
        lambda_=lambda_lr,
        chunked=chunked,
        nufft_eps=nufft_eps,
        **pclean_parameters
    )
    print("\tSolving...")
    start = time.time()
    pclean.fit(**fit_parameters)
    print("\tSolved in {:.3f} seconds".format(dt_pcleanlr := time.time() - start))
    if diagnostics:
        pclean.diagnostics(log=log_diagnostics)
    datalr, histlr = pclean.stats()
    residual_lr = lowresOp.adjoint(measurements - lowresOp(datalr["x"]))
    model_pclean_lr = model_lr.copy(deep=True)
    model_pclean_lr.pixels.data[0, 0] = datalr["x"].reshape(model_pclean_lr.pixels.data.shape)
    model_pclean_lr.image_acc.export_to_fits(os.path.join(images_path, 'pc-lr-model.fits'))


    print("PolyCLEAN high res: Setting up...")
    lambda_hr = lambda_factor * np.abs(dirty_hr).max()
    fit_parameters = {
        "stop_crit": stop_crit,
        "positivity_constraint": True,
        "diff_lipschitz": lips_high ** 2,
        "precision_rule": lambda k: 10 ** (-k / 10),
    }

    # Computations
    pclean = pc.PolyCLEAN(
        data=measurements,
        uvwlambda=flagged_uvwlambda,
        direction_cosines=d_cosines[1],
        lambda_=lambda_hr,
        chunked=chunked,
        nufft_eps=nufft_eps,
        **pclean_parameters
    )
    print("\tSolving...")
    start = time.time()
    pclean.fit(**fit_parameters)
    print("\tSolved in {:.3f} seconds".format(dt_pcleanhr := time.time() - start))
    if diagnostics:
        pclean.diagnostics(log=log_diagnostics)
    datahr, histhr = pclean.stats()
    residual_hr = highresOp.adjoint(measurements - highresOp(datahr["x"]))
    model_pclean_hr = model_hr.copy(deep=True)
    model_pclean_hr.pixels.data[0, 0] = datahr["x"].reshape(model_pclean_hr.pixels.data.shape)
    model_pclean_hr.image_acc.export_to_fits(os.path.join(images_path, 'pc-hr-model.fits'))

    ### Results
    # Show the two polyclean reconstructions
    ut.plot_image(model_pclean_lr, title="PolyCLEAN low resolution model")
    ut.plot_image(model_pclean_hr, title="PolyCLEAN high resolution model")

    #todo check convergence metrics and verify everything has been saved correctly
    print("PolyCLEAN low res final DCV: {:.3f}".format(datalr["dcv"]))
    print("polyCLEAN high res final DCV: {:.3f}".format(datahr["dcv"]))

    keys = ["dt_wsclean_lr", "dt_wsclean_hr", "dt_pcleanlr", "dt_pcleanhr", "rmax", "lambda_factor", "thresh", "nom_res", "high_res"]
    toexport = {k: float(v) for k, v in locals().items() if k in keys}
    toexport["dcvlr"] = float(datalr["dcv"].astype(float))
    toexport["dcvhr"] = float(datahr["dcv"].astype(float))

    ### save the dictionary toexport to a yaml file
    import yaml
    with open(os.path.join(images_path, 'variables.yaml'), 'w') as file:
        yaml.dump(toexport, file, default_flow_style=False)


    # ### Results
    # print("PolyCLEAN final DCV: {:.3f}".format(data["dcv"]))
    # print(f"Final value of the objective function: {hist['Memorize[objective_func]'][-1]:.3e}")
    # print("Iterations: {}".format(int(hist['N_iter'][-1])))
    # print("Final sparsity of the components: {}".format(np.count_nonzero(data["x"])))
    #
    # pclean_comp = sky_im.copy(deep=True)
    # pclean_comp.pixels.data[0, 0] = data["x"].reshape((npixel,) * 2)
    # psf, sumwt = invert_visibility(vt, sky_im, context="ng", dopsf=True)
    # clean_beam = fit_psf(psf)
    # pclean_restored = restore_cube(pclean_comp, None, None, clean_beam)
    # sky_im_restored = restore_cube(sky_im, None, None, clean_beam)
    # pclean_residual_im = sky_im.copy(deep=True)
    # pclean_residual_im.pixels.data = pclean_residual.reshape(pclean_residual_im.pixels.data.shape) / (
    #             measurements.shape[0] // 2)
    #
    # ut.plot_source_reco_diff(sky_im_restored, pclean_restored, title="PolyCLEAN Convolved", suptitle="Comparison",
    #                          sc=sc)
    #
    # # ut.compare_3_images(sky_im_restored, pclean_comp, pclean_restored, titles=["components", "convolution"], sc=sc)
    #
    # from ska_sdp_func_python.imaging import predict_visibility
    #
    # predicted_visi = predict_visibility(vt, sky_im, context="ng")
    # dirty_rascil, _ = invert_visibility(predicted_visi, sky_im, context="ng", dopsf=False, normalise=True)
    #
    # print(
    #     "CLEAN beam (MSE/MAD):\n\tDirty image: {:.2e}/{:.2e}\n\tComponents convolved: {:.2e}/{:.2e}\n\tRaw components: {:.2e}/{:.2e}".format(
    #         ut.MSE(dirty_rascil, sky_im_restored), ut.MAD(dirty_rascil, sky_im_restored),
    #         ut.MSE(pclean_restored, sky_im_restored), ut.MAD(pclean_restored, sky_im_restored),
    #         ut.MSE(sky_im, pclean_comp), ut.MAD(sky_im, pclean_comp)
    #     )
    # )
    #
    # sharp_beam = clean_beam.copy()
    # sharp_beam["bmin"] = clean_beam["bmin"] / 2
    # sharp_beam["bmaj"] = clean_beam["bmaj"] / 2
    # pclean_comp_sharp = restore_cube(pclean_comp, None, None, sharp_beam)
    # sky_im_sharp = restore_cube(sky_im, None, None, sharp_beam)
    #
    # print("Sharp beam (MSE/MAD):\n\tDirty image: {:.2e}/{:.2e}\n\tComponents convolved: {:.2e}/{:.2e}".format(
    #     ut.MSE(dirty_rascil, sky_im_sharp), ut.MAD(dirty_rascil, sky_im_sharp),
    #     ut.MSE(pclean_comp_sharp, sky_im_sharp), ut.MAD(pclean_comp_sharp, sky_im_sharp),
    # )
    # )
    #
    # # print(np.allclose(dirty_image/dirty_image.max(), dirty_rascil.pixels.data.flatten()/dirty_image.max()))
    #
    # # ut.plot_image(dirty_rascil, title="Rascil")
    # # dirty_copy = dirty_rascil.copy(deep=True)
    # # dirty_copy.pixels.data = dirty_image.reshape((1, 1, npixel, npixel))
    # # ut.plot_image(dirty_copy, title="Nufft")
    # # diff = dirty_rascil.copy(deep=True)
    # # diff.pixels.data = (dirty_image.reshape((1, 1, npixel, npixel)) - dirty_rascil.pixels.data)/dirty_image.max()
    # # ut.plot_image(diff, title="Diff", cmap="bwr")
    #
    # # import matplotlib.pyplot as plt
    # # fluxs = np.array([comp.flux[0,0] for comp in sc])
    # # fluxs /= fluxs.max()
    # # plt.figure()
    # # plt.hist(fluxs, bins=50)
    # # plt.show()
    #
    # from scripts.observations.pclean import plot_1_image
    # sharp_beam = clean_beam.copy()
    # sharp_beam["bmin"] = clean_beam["bmin"] / 10
    # sharp_beam["bmaj"] = clean_beam["bmaj"] / 10
    # test_sky_im = restore_cube(sky_im, None, None, sharp_beam)
    # # test_sky_im['pixels'].data = 1000 * test_sky_im['pixels'].data
    #
    # def plot_1_image(image, title="", cmaps=['hot', 'Greys'], alpha=.95, offset_cm=0., symm=True):
    #     from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    #     import matplotlib.colors as mplc
    #     arr = image.pixels.data[0, 0]
    #
    #     fig = plt.figure(figsize=(12, 10))
    #     ax = fig.subplots(1, 1, subplot_kw={'projection': image.image_acc.wcs.sub([1, 2]), 'frameon': False})
    #     ax.set_xlabel(image.image_acc.wcs.wcs.ctype[0])
    #     ax.set_ylabel(image.image_acc.wcs.wcs.ctype[1])
    #     # vlim = -arr.min() if symm else 0.
    #     # mask_comp = np.ma.masked_array(arr, arr < vlim, fill_value=vlim)
    #     aximc = ax.imshow(arr, origin="lower", cmap='hot', interpolation='none', alpha=alpha,
    #                       norm=mplc.PowerNorm(gamma=0.5, vmin=0., vmax=None))
    #     axinsc = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
    #     cbc = fig.colorbar(aximc, cax=axinsc, orientation="vertical")  # , ticks=[round(0) + 1, 500, 1000, 2000, 3000, 4000])
    #     fig.suptitle(title)
    #     plt.subplots_adjust(top=0.92, bottom=0.08, left=0.0, right=0.93, hspace=0.15, wspace=0.15)
    #     fig.show()
    #
    # # plot_1_image(test_sky_im)
    # # import os
    # # folder_path = os.path.join("/home/jarret/PycharmProjects/polyclean/scripts/simulations_ps", 'source')
    # # # create a folder if it does not exist
    # # if not os.path.exists(folder_path):
    # #     os.makedirs(folder_path)
    # # plt.savefig(os.path.join(folder_path, 'source.png'))
    #
    # ### Compare cellsizes
    # from ska_sdp_func_python.imaging import advise_wide_field
    # import polyclean.ra_utils as pcrau
    #
    # srf = 2.5
    #
    # advice = advise_wide_field(
    #     vt, guard_band_image=6.0, delA=0.1, oversampling_synthesised_beam=3.0
    #     )
    # rascil_cs = advice["cellsize"]  # radians
    # npix = pcrau.get_npixels(vt, fov_deg, phasecentre, 1e-3, srf=srf)
    # hvox_cs = np.pi * fov_deg / (180 * npix) # radians
    # print(f"Rascil cellsize: {rascil_cs:.3e} rad")
    # print(f"HVox cellsize: {hvox_cs:.3e} rad")
    # print(f"Current cellsize: {fov_deg/npixel * np.pi/180:.3e} rad")
    #
    # import polyclean.image_utils as ut
    #
    # ut.plot_image(pclean_comp, sc=sc)
