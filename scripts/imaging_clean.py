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
)
from ska_sdp_func_python.image import (
    deconvolve_cube,
    restore_cube,
    fit_psf,
)
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame

import polyclean.image_utils as ut

import matplotlib

matplotlib.use("Qt5Agg")
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

seed = 195  # np.random.randint(0, 1000)  # np.random.randint(0, 1000)  # 492
rmax = 500.  # 2000.
times = np.zeros([1])
fov_deg = 5
npixel = 1024  # 512  # 384 #  128 * 2
npoints = 200
niter = 500

context = "ng"

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

    ### Image reconstruction with CLEAN
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
        niter=niter,
        threshold=0.001,
        fractional_threshold=0.001,
        window_shape="quarter",
        gain=0.7,
        algorithm='hogbom',
    )

    clean_comp = image_model.copy(deep=True)
    clean_comp['pixels'].data[0, 0, ...] = \
        tmp_clean_comp['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]
    clean_residual = image_model.copy(deep=True)
    clean_residual['pixels'].data[0, 0, ...] = \
        tmp_clean_residual['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]
    dt = time.time() - start
    print("\tSolved in {:.3f} seconds".format(dt))

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
    ut.plot_analysis_reconstruction(sky_im_restored, clean_restored, normalize=True,
                                    title="CLEAN - niter {}".format(niter), sc=sc,
                                    suptitle=f"Reconstruciton - rmax{rmax}")

    print("Errors:\n\tDirty image: {:.2e}\n\tCLEAN image: {:.2e}\n\tCLEAN components: {:.2e}".format(
        ut.MSE(cropped_dirty, sky_im_restored)[0][0],
        ut.MSE(clean_restored, sky_im_restored)[0][0],
        ut.MSE(sky_im, clean_comp)[0][0]))

    # ### Sharp sky image
    # clean_beam["bmaj"] /= 2.  # super resolution
    # clean_beam["bmin"] /= 2.  # super resolution
    # sky_im_sharp = restore_cube(sky_im, None, None, clean_beam)
    # ut.plot_image(sky_im_sharp, sc=sc, title="Sky Image")
    # ut.plot_image(clean_comp, sc=sc, title="CLEAN Components")
