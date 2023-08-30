import numpy as np

from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_func_python.sky_component import insert_skycomponent

import pycsou.operator.linop as pycl

__all__ = [
    "generate_point_sources",
    "get_npixels",
    "image_add_ra_dec_grid"
]

DEFAULT_PHASECENTER = SkyCoord(
    ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
)

def get_npixels(vt, fov, phasecentre, epsilon):
    nmodes = get_nmodes(vt, epsilon, phasecentre=phasecentre, fov=fov, upsampfac=2)[0]
    return max(nmodes[:2]) * 10


def get_nmodes(vt, epsilon, phasecentre=None, fov=None, direction_cosines=None, upsampfac=2):
    """

    Parameters
    ----------
    vt
        Rascil visibility object
    epsilon
        desired accuracy for NUFFT (e.g. 1e-4)
    phasecentre
        center of your observation as SkyCoord
    fov
        field of view in degrees
    direction_cosines
    upsampfac

    Returns
    -------

    """

    if (fov is not None) and (phasecentre is not None) and (direction_cosines is None):
        llc = SkyCoord(
            ra=phasecentre.ra - fov / 2 * u.deg,
            dec=phasecentre.dec - fov / 2 * u.deg,
            frame="icrs",
            equinox="J2000",
        )
        urc = SkyCoord(
            ra=phasecentre.ra + fov / 2 * u.deg,
            dec=phasecentre.dec + fov / 2 * u.deg,
            frame="icrs",
            equinox="J2000",
        )
        lmn = (
            skycoord_to_lmn(llc, phasecentre),
            skycoord_to_lmn(urc, phasecentre),
            skycoord_to_lmn(phasecentre, phasecentre),
        )
        nufft = pycl.NUFFT.type3(
            x=np.array(lmn),
            z=2 * np.pi * vt.visibility_acc.uvw_lambda.reshape(-1, 3),
            real=True,
            isign=-1,
            eps=epsilon,
            plan_fw=False,
            plan_bw=False,
            upsampfac=upsampfac,
        )
        return nufft._fft_shape(), None, None

    elif (fov is None) and (direction_cosines is not None):
        nufft = pycl.NUFFT.type3(
            x=direction_cosines,
            z=2 * np.pi * vt.visibility_acc.uvw_lambda.reshape(-1, 3),
            real=True,
            isign=-1,
            eps=epsilon,
            plan_fw=False,
            plan_bw=False,
            upsampfac=upsampfac,
        )
        return (
            nufft._fft_shape(),
            len(direction_cosines),
            len(vt.visibility_acc.uvw_lambda.reshape(-1, 3)),
        )
    else:
        raise NotImplementedError

def image_add_ra_dec_grid(im):
    """Add ra, dec coordinates"""
    _, _, ny, nx = im["pixels"].shape
    lmesh, mmesh = np.meshgrid(np.arange(ny), np.arange(nx))
    ra_grid, dec_grid = image_wcs(im).sub([1, 2]).wcs_pix2world(lmesh, mmesh, 0)
    ra_grid = np.deg2rad(ra_grid)
    dec_grid = np.deg2rad(dec_grid)
    im = im.assign_coords(
        ra_grid=(("x", "y"), ra_grid), dec_grid=(("x", "y"), dec_grid)
    )
    return im

def image_wcs(ds):
    """

    :param ds:
    :return:
    """
    # assert ds.rascil_data_model == "Image", ds.rascil_data_model

    w = WCS(naxis=4)
    nchan, npol, ny, nx = ds["pixels"].shape
    l = np.rad2deg(ds["x"].data[nx // 2])
    m = np.rad2deg(ds["y"].data[ny // 2])
    cellsize_l = np.rad2deg((ds["x"].data[-1] - ds["x"].data[0]) / (nx - 1))
    cellsize_m = np.rad2deg((ds["y"].data[-1] - ds["y"].data[0]) / (ny - 1))
    freq = ds["frequency"].data[0]
    pol = PolarisationFrame.fits_codes[ds.attrs["_polarisation_frame"]]
    if npol > 1:
        dpol = pol[1] - pol[0]
    else:
        dpol = 1.0
    if nchan > 1:
        channel_bandwidth = (ds["frequency"].data[-1] - ds["frequency"].data[0]) / (
                nchan - 1
        )
    else:
        channel_bandwidth = freq

    projection = ds._projection
    # The negation in the longitude is needed by definition of RA, DEC
    if ds.spectral_type == "MOMENT":
        w.wcs.crpix = ds.attrs["refpixel"]
        w.wcs.ctype = [projection[0], projection[1], "STOKES", ds.spectral_type]
        w.wcs.crval = [l, m, pol[0], 0.0]
        w.wcs.cdelt = [-cellsize_l, cellsize_m, dpol, 1]
        w.wcs.radesys = "ICRS"
        w.wcs.equinox = 2000.0
    else:
        w.wcs.crpix = ds.attrs["refpixel"]
        w.wcs.ctype = [projection[0], projection[1], "STOKES", ds.spectral_type]
        w.wcs.crval = [l, m, pol[0], freq]
        w.wcs.cdelt = [-cellsize_l, cellsize_m, dpol, channel_bandwidth]
        w.wcs.radesys = "ICRS"
        w.wcs.equinox = 2000.0

    return w

def generate_point_sources(nsources: int,
                           fov_deg: float,
                           npixel: int,
                           flux_sigma: float = .4,
                           radius_rate: float = .9,
                           phasecentre=DEFAULT_PHASECENTER,
                           frequency=np.array([1e8]),
                           channel_bandwidth=np.array([1e6]),
                           seed: int = None,
                           ):
    fov = fov_deg * np.pi / 180
    cellsize = fov / npixel
    radius = radius_rate * fov
    rng = np.random.default_rng(seed)

    sc = []
    maxi = 0.
    for i in range(nsources):
        sc_flux = rng.lognormal(sigma=flux_sigma)
        if sc_flux > maxi:
            maxi = sc_flux
        sc_ra = radius * rng.random() - radius / 2 + phasecentre.ra.rad
        sc_dec = radius * rng.random() - radius / 2 + phasecentre.dec.rad
        sc_coord = SkyCoord(ra=sc_ra * u.rad, dec=sc_dec * u.rad, frame="icrs", equinox="J2000")
        sc.append(
            SkyComponent(sc_coord, flux=np.r_[sc_flux].reshape(1, 1), frequency=np.r_[frequency],
                         polarisation_frame=PolarisationFrame("stokesI")
                         ))
    for s in sc:
        s.flux /= maxi
    sky_im = create_image(
        npixel=npixel,
        cellsize=cellsize,
        polarisation_frame=PolarisationFrame("stokesI"),
        frequency=frequency,
        nchan=frequency.size,
        channel_bandwidth=channel_bandwidth,
        phasecentre=phasecentre)

    sky_im = image_add_ra_dec_grid(sky_im)
    insert_skycomponent(sky_im, sc, insert_method='Nearest')
    sky_im.pixels.data /= sky_im.pixels.data.max()
    return sky_im, sc