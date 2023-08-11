import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_func_python.imaging import (
    predict_visibility,
    invert_visibility,
    create_image_from_visibility,
)

import polyclean.image_utils as ut
import polyclean.polyclean as pc

import pycsou.util.complex as pycuc


seed = None  # np.random.randint(0, 1000)  # np.random.randint(0, 1000)  # 492
rmax = 300.
times = np.zeros([1])
fov_deg = 10
npixel = 256  # 512  # 384 #  128 * 2
npoints = 100
nufft_eps = 0.
context = "ng"

## ----------------------------------

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
uvwlambda = vt.visibility_acc.uvw_lambda.reshape(-1, 3)
flags_bool = np.any(uvwlambda != 0., axis=-1)
flagged_uvwlambda = uvwlambda[flags_bool]

### Simulation of the measurements
forwardOp = pc.generatorVisOp(direction_cosines=direction_cosines,
                              # vlambda=uvwlambda,
                              vlambda=flagged_uvwlambda,
                              nufft_eps=nufft_eps)


def test_forward():
    """
    Test consistency with RASCIL
    """
    measurements = forwardOp.apply(sky_im.pixels.data.flatten())
    meas = pycuc.view_as_complex(measurements)
    predicted_visi = predict_visibility(vt, sky_im, context=context)
    pred_visi_flat = predicted_visi.vis.data.flatten()[flags_bool]

    assert meas.size == pred_visi_flat.size
    print(np.linalg.norm(meas - pred_visi_flat))
    print(np.abs(meas - pred_visi_flat).max(), meas.max())
    assert np.allclose(meas, pred_visi_flat)

def test_adjoint():
    """
    Test consistency of the computed dirty image with RASCIL
    """
    dirty_hvox = forwardOp.adjoint(forwardOp.apply(sky_im.pixels.data.flatten()))
    maxi = dirty_hvox.max()

    ### Image reconstruction with CLEAN
    cellsize = abs(sky_im.coords["x"].data[1] - sky_im.coords["x"].data[0])
    npixel = sky_im.dims["x"]
    predicted_visi = predict_visibility(vt, sky_im, context=context)
    image_model = create_image_from_visibility(predicted_visi, cellsize=cellsize, npixel=npixel)
    dirty_rascil, _ = invert_visibility(predicted_visi, image_model, context=context, normalise=False)

    assert dirty_hvox.size == dirty_rascil.pixels.data.size
    assert np.allclose(dirty_hvox/maxi, dirty_rascil.pixels.data.flatten()/maxi, rtol=1e-5)
    assert np.allclose(dirty_hvox, dirty_rascil.pixels.data.flatten(), rtol=1e-5)



