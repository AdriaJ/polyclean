import numpy as np
import time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel

from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_func_python.imaging import (
    predict_visibility,
    invert_visibility,
    create_image_from_visibility,
    advise_wide_field
)
from ska_sdp_func_python.image import (
    deconvolve_cube,
    restore_cube,
    fit_psf,
)
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame

import polyclean.image_utils as ut
from polyclean.clean_utils import mjCLEAN

import pyxu.util.complex as pxc


# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

plt.rc('text', usetex=False)
# plt.rc('text.latex', unicode=False)
plt.rc('svg', fonttype='none')

# import logging
# import sys
#
# log = logging.getLogger("rascil-logger")
# log.setLevel(logging.DEBUG)
# log.addHandler(logging.StreamHandler(sys.stdout))

seed = 64  # np.random.randint(0, 1000)  # np.random.randint(0, 1000)  # 492
fov_deg = 5
npoints = 200
times = (np.arange(7)-3) * np.pi/9  # 7 angles from -pi/3 to pi/3

rmax = 300.  # 2000.
npixel = 720  # 512  # 384 #  128 * 2
psnrdb = 20

niter = 10_000
nmajor = 5
gain = 0.1
context = "ng"
algorithm = "hogbom"

if __name__ == "__main__":
    if seed is None:
        seed = np.random.randint(1000)
    print("Seed: {}".format(seed))
    rng = np.random.default_rng(seed)

    ### Simulation of the source

    frequency = np.array([1.e+8])
    channel_bandwidth = np.array([1.e+6])
    phasecentre = SkyCoord(
        ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
    )

    sky_im, sc = ut.generate_point_sources(npoints,
                                           fov_deg,
                                           npixel,
                                           flux_sigma=.8,
                                           radius_rate=.9,
                                           phasecentre=phasecentre,
                                           frequency=frequency[0],
                                           channel_bandwidth=channel_bandwidth[0],
                                           seed=seed)
    # parametrisation of the image
    directions = SkyCoord(
        ra=sky_im.ra_grid.data.ravel() * u.rad,
        dec=sky_im.dec_grid.data.ravel() * u.rad,
        frame="icrs",
        equinox="J2000",
    )

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

    advice = advise_wide_field(
        vt, guard_band_image=3.0, delA=0.1, oversampling_synthesised_beam=4.0
    )
    cellsize_advice = advice["cellsize"]  # radians
    fov_deg_advice = 180. * npixel * cellsize_advice / np.pi
    print(f"Advise wide field recommends a FOV of {fov_deg_advice:.2f} deg for a good clean reconstruction.")

    ### Image reconstruction with CLEAN
    cellsize = abs(sky_im.coords["x"].data[1] - sky_im.coords["x"].data[0])  # radians
    npixel = sky_im.dims["x"]

    predicted_visi = predict_visibility(vt, sky_im, context=context)
    # clean_model = create_image_from_visibility(predicted_visi, cellsize=cellsize, npixel=2 * npixel)
    # noiseless_dirty, sumwt_dirty = invert_visibility(predicted_visi, clean_model, context=context)
    # ut.plot_image(noiseless_dirty)
    real_visi = pxc.view_as_real(predicted_visi.vis.data[:, :, 0, 0])
    noise_scale = np.abs(real_visi).max() * 10 ** (-psnrdb / 20) / np.sqrt(2)
    noise = np.random.normal(0, noise_scale, real_visi.shape)
    predicted_visi.vis.data += pxc.view_as_complex(noise)[:, :, None, None]

    image_model = create_image_from_visibility(predicted_visi, cellsize=cellsize, npixel=npixel)

    clean_model = create_image_from_visibility(predicted_visi, cellsize=cellsize, npixel=2 * npixel)
    dirty, sumwt_dirty = invert_visibility(predicted_visi, clean_model, context=context)
    psf, sumwt = invert_visibility(predicted_visi, image_model, context=context, dopsf=True)
    print("CLEAN: Solving...")
    start = time.time()
    # tmp_clean_comp, tmp_clean_residual = deconvolve_cube(
    #     dirty,
    #     psf,
    #     niter=niter,
    #     threshold=0.001,
    #     fractional_threshold=0.001,
    #     window_shape="quarter",
    #     gain=gain,
    #     algorithm='hogbom',
    # )

    tmp_clean_comp, tmp_clean_residual = mjCLEAN(
        dirty,
        psf,
        n_major=nmajor,
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
    print(f"\tFinal sparsity of the components: {np.count_nonzero(clean_comp.pixels.data):d}")

    cropped_dirty = image_model.copy(deep=True)
    cropped_dirty['pixels'].data[0, 0, ...] = \
        dirty['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]

    clean_beam = fit_psf(psf)
    # clean_beam["bmaj"] /= np.sqrt(2)  # super resolution
    # clean_beam["bmin"] /= np.sqrt(2)  # super resolution
    clean_restored = restore_cube(clean_comp, None, None, clean_beam)
    sky_im_restored = restore_cube(sky_im, None, None, clean_beam)

    # ut.plot_image(sky_im_restored, sc=sc, title="Sky Image")
    # ut.plot_image(cropped_dirty, sc=sc, title="Dirty Image")
    # ut.plot_analysis_reconstruction(sky_im_restored, clean_restored, normalize=True,
    #                                 title="CLEAN - niter {}".format(niter), sc=sc,
    #                                 suptitle=f"Reconstruction - rmax {rmax} - niter {niter}")
    ut.plot_source_reco_diff(sky_im_restored, clean_restored, title="CLEAN Convolved",
                             suptitle=f"Reconstruction - rmax {rmax} - niter {niter}", sc=sc)
    ut.plot_image(dirty)
    ut.plot_image(clean_residual)


    print("CLEAN beam (MSE/MAD):\n\tDirty image: {:.2e}/{:.2e}\n\tComponents convolved: {:.2e}/{:.2e}\n\tCLEAN components: {:.2e}/{:.2e}".format(
        ut.MSE(cropped_dirty, sky_im_restored), ut.MAD(cropped_dirty, sky_im_restored),
        ut.MSE(clean_restored, sky_im_restored), ut.MAD(clean_restored, sky_im_restored),
        ut.MSE(sky_im, clean_comp), ut.MAD(sky_im, clean_comp)
        )
    )

    sharp_beam = clean_beam.copy()
    sharp_beam["bmin"] = clean_beam["bmin"] / 2
    sharp_beam["bmaj"] = clean_beam["bmaj"] / 2
    clean_comp_sharp = restore_cube(clean_comp, None, None, sharp_beam)
    sky_im_sharp = restore_cube(sky_im, None, None, sharp_beam)

    print("Sharp beam (MSE/MAD):\n\tDirty image: {:.2e}/{:.2e}\n\tComponents convolved: {:.2e}/{:.2e}".format(
        ut.MSE(cropped_dirty, sky_im_sharp), ut.MAD(cropped_dirty, sky_im_sharp),
        ut.MSE(clean_comp_sharp, sky_im_sharp), ut.MAD(clean_comp_sharp, sky_im_sharp),
        )
    )

    # ### Sharp sky image
    # clean_beam["bmaj"] /= 2.  # super resolution
    # clean_beam["bmin"] /= 2.  # super resolution
    # sky_im_sharp = restore_cube(sky_im, None, None, clean_beam)
    # ut.plot_image(sky_im_sharp, sc=sc, title="Sky Image")
    # ut.plot_image(clean_comp, sc=sc, title="CLEAN Components")

    ## Export the simulated visibilities as a MS file
    folder = "/home/jarret/PycharmProjects/polyclean/scripts/simulations_ps/simu_ms/rmax800" + "/"
    from rascil.processing_components.visibility.base import export_visibility_to_ms
    export_visibility_to_ms(folder + "rmax800.ms", [predicted_visi],)

    images = [psf, cropped_dirty, clean_comp, clean_residual, clean_restored]
    names = ['psf', 'dirty', 'model', 'residual', 'image']

    for im, n in zip(images, names):
        filename = 'rsclean-' + n + '.fits'
        im.image_acc.export_to_fits(folder + filename)
    sky_im.image_acc.export_to_fits(folder + 'sky-image.fits')

    ut.myplot_uvcoverage(predicted_visi, title="UV coverage")

    import polyclean.ra_utils as pcrau
    npix = pcrau.get_npixels(vt, fov_deg, phasecentre, 1e-3)
    pcrau.get_nmodes(vt, 1e-3, phasecentre=phasecentre, fov=fov_deg, upsampfac=1.25)
