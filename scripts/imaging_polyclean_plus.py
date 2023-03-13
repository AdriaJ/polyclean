import numpy as np
import time
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.processing_components.util import skycoord_to_lmn
from rascil.processing_components import create_named_configuration, create_visibility
from rascil.data_models import PolarisationFrame

import polyclean.reconstructions as reco
import polyclean.image_utils as ut
import polyclean.polyclean as pc

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


seed = np.random.randint(0, 1000)  # np.random.randint(0, 1000)  # 492
rmax = 300.  # 2000.
times = np.zeros([1])
fov_deg = 6.5
npixel = 256  # 512  # 384 #  128 * 2
npoints = 100
nufft_eps = 1e-3

lambda_factor = .12

eps = 1e-3
tmax = 240.
min_iter = 20
ms_threshold = 0.75
init_correction_prec = 1e-1
final_correction_prec = 1e-4
remove = True
min_correction_steps = 3

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
    uvwlambda = vt.uvw_lambda.data.reshape(-1, 3)
    flags_bool = np.any(uvwlambda != 0., axis=-1)
    flagged_uvwlambda = uvwlambda[flags_bool]

    ### Simulation of the measurements
    forwardOp = pc.generatorVisOp(direction_cosines=direction_cosines,
                                  vlambda=flagged_uvwlambda,
                                  nufft_eps=nufft_eps)
    start = time.time()
    fOp_lipschitz = forwardOp.lipschitz(tol=1.)
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
    }
    fit_parameters = {
        "stop_crit": stop_crit,
        "positivity_constraint": True,
        "diff_lipschitz": fOp_lipschitz ** 2
    }

    # Computations
    data, hist = reco.reco_pclean_plus(flagged_uvwlambda, direction_cosines, measurements, lambda_, pcleanp_parameters, fit_parameters)

    ### Results
    print("PolyCLEAN final DCV (before post processing): {:.3f}".format(data["dcv"]))
    print("Iterations: {}".format(int(hist['N_iter'][-1])))
    print("Final sparsity: {}".format(np.count_nonzero(data["lsr_x"])))

    # Visualization
    from rascil.processing_components import invert_visibility, fit_psf, restore_cube
    from astropy.wcs.utils import skycoord_to_pixel
    from matplotlib import colors

    pclean_comp = sky_im.copy(deep=True)
    pclean_comp.pixels.data[0, 0] = data["lsr_x"].reshape((npixel, )*2)
    psf, sumwt = invert_visibility(vt, sky_im, context="ng", dopsf=True)
    clean_beam = fit_psf(psf)
    pclean_restored = restore_cube(pclean_comp, None, None, clean_beam)
    sky_im_restored = restore_cube(sky_im, None, None, clean_beam)

    # ut.compare_3_images(sky_im_restored, pclean_comp, pclean_restored, titles=["components", "convolution"], sc=sc)

    source = sky_im_restored
    im1 = pclean_restored
    normalize = False
    chan, pol = 0, 0

    if normalize:
        vmax, vmin = source['pixels'].data.max(), 0.
    else:
        vmax, vmin = None, None
    cm = "Greys"
    fig = plt.figure(figsize=(15, 6))
    axes = fig.subplots(1, 3, sharex=True, sharey=True, subplot_kw={'projection': source.image_acc.wcs.sub([1, 2]),
                                                                    'frameon': False})

    ax = axes[0]
    source_array = np.real(source["pixels"].data[chan, pol, :, :])
    im = ax.imshow(source_array, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_ylabel(source.image_acc.wcs.wcs.ctype[1])
    ax.set_xlabel(source.image_acc.wcs.wcs.ctype[0])
    ax.set_title("Source image")
    if sc is not None:
        for component in sc:
            x, y = skycoord_to_pixel(component.direction, source.image_acc.wcs, 0, "wcs")
            ax.scatter(x, y, marker=".", color="red", s=1, alpha=.9)
    fig.colorbar(im, orientation="vertical", shrink=0.8, ax=ax)

    ax = axes[1]
    ax.coords[1].set_ticklabel_visible(False)
    ax.coords[1].set_axislabel('')
    ax.set_xlabel(source.image_acc.wcs.wcs.ctype[0])
    im_array = np.real(im1["pixels"].data[chan, pol, :, :])
    ims = ax.imshow(im_array, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_title("PolyCLEAN Convolved")
    fig.colorbar(ims, ax=ax, orientation="vertical", shrink=0.8)
    if sc is not None:
        for component in sc:
            x, y = skycoord_to_pixel(component.direction, im1.image_acc.wcs, 0, "wcs")
            ax.scatter(x, y, marker=".", color="red", s=1, alpha=.9)

    ax = axes[2]
    ax.coords[1].set_ticklabel_visible(False)
    ax.coords[1].set_axislabel('')
    ax.set_xlabel(source.image_acc.wcs.wcs.ctype[0])
    diff_array = im_array - np.real(source['pixels'].data[chan, pol])
    vmax_diff = diff_array.max()
    vmin_diff = diff_array.min()
    vext = max(vmax_diff, -vmin_diff)
    ims = ax.imshow(diff_array, norm=colors.CenteredNorm(halfrange=vext), origin="lower", cmap="bwr")
    ax.set_title("Difference")
    fig.colorbar(ims, ax=ax, orientation="vertical", shrink=0.8)
    if sc is not None:
        for component in sc:
            x, y = skycoord_to_pixel(component.direction, im1.image_acc.wcs, 0, "wcs")
            ax.scatter(x, y, marker=".", color="red", s=1, alpha=.9)

    fig.suptitle("Comparison of the reconstructions")
    plt.show()
