# This script makes a fake data set and then deconvolves it.


import logging
import sys

import numpy
import time as time
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_func_python.image import (
    deconvolve_cube,
    restore_cube,
)
from ska_sdp_func_python.imaging import (
    predict_visibility,
    invert_visibility,
    create_image_from_visibility,
    advise_wide_field,
)

from rascil.processing_components import create_test_image


# matplotlib.use("Qt5Agg")

log = logging.getLogger("rascil-logger")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

results_dir = "./"

context = "ng"

# Construct LOW core configuration
lowr3 = create_named_configuration("LOWBD2", rmax=1000.0)

# We create the visibility. This just makes the uvw, time, antenna1, antenna2,
# weight columns in a table. We subsequently fill the visibility value in by
# a predict step.

times = numpy.zeros([1])
frequency = numpy.array([1e8])
channel_bandwidth = numpy.array([1e6])
phasecentre = SkyCoord(
    ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
)
vt = create_visibility(
    lowr3,
    times,
    frequency,
    channel_bandwidth=channel_bandwidth,
    weight=1.0,
    phasecentre=phasecentre,
    polarisation_frame=PolarisationFrame("stokesI"),
)

# Find the recommended imaging parameters
advice = advise_wide_field(
    vt, guard_band_image=3.0, delA=0.1, oversampling_synthesised_beam=4.0
)
cellsize = advice["cellsize"]  # radians


# Read the venerable test image, constructing a RASCIL Image
m31image = create_test_image(
    cellsize=cellsize, frequency=frequency, phasecentre=vt.phasecentre
)
npixel = m31image.pixels.shape[-1]
fov_deg = 180. * npixel * cellsize / numpy.pi
print(f"Field of view size: {fov_deg:.2f} (deg)")

# Predict the visibility for the Image
vt = predict_visibility(vt, m31image, context=context)

# Make the dirty image and point spread function
model = create_image_from_visibility(vt, cellsize=cellsize, npixel=512)
dirty, sumwt = invert_visibility(vt, model, context=context)
psf, sumwt = invert_visibility(vt, model, context=context, dopsf=True)

print(
    "Max, min in dirty image = %.6f, %.6f, sumwt = %f"
    % (dirty["pixels"].data.max(), dirty["pixels"].data.min(), sumwt)
)
print(
    "Max, min in PSF         = %.6f, %.6f, sumwt = %f"
    % (psf["pixels"].data.max(), psf["pixels"].data.min(), sumwt)
)

dirty.image_acc.export_to_fits("%s/imaging_dirty.fits" % (results_dir))
psf.image_acc.export_to_fits("%s/imaging_psf.fits" % (results_dir))

start = time.time()
# Deconvolve using clean
comp, residual = deconvolve_cube(
    dirty,
    psf,
    niter=10000,
    threshold=0.001,
    fractional_threshold=0.001,
    window_shape="quarter",
    gain=0.7,
    scales=[0, 3, 10, 30],
)

restored = restore_cube(comp, psf, None)  # residual)
duration = time.time() - start
print(
    "Max, min in restored image = %.6f, %.6f, sumwt = %f"
    % (restored["pixels"].data.max(), restored["pixels"].data.min(), sumwt)
)
print(f"Reconstruction with CLEAN in {duration:.3f} s.")
# restored.image_acc.export_to_fits("%s/imaging_restored.fits" % (results_dir))

import polyclean.image_utils as ut
ut.plot_image(m31image, title="Source")
ut.plot_image(comp, title="Components")
ut.plot_image(restored, title="Restored")
