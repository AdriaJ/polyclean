import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import pycsou.runtime as pycrt

from ska_sdp_datamodels.image import Image
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_func_python.sky_component import insert_skycomponent
from ska_sdp_datamodels.image.image_create import create_image

# from rascil.processing_components import image_add_ra_dec_grid
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy import units as u

__all__ = [
    "generate_point_sources",
    "display_image_list",
    "MSE",
]

DEFAULT_PHASECENTER = SkyCoord(
    ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
)


def generate_point_sources(npoints: int,
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
    for i in range(npoints):
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
    # w = WCS(naxis=4)
    # pol = PolarisationFrame.fits_codes[polarisation_frame.type]
    # if npol > 1:
    #     dpol = pol[1] - pol[0]
    # else:
    #     dpol = 1.0
    #
    # # The negation in the longitude is needed by definition of RA, DEC
    # w.wcs.cdelt = [
    #     -cellsize * 180.0 / numpy.pi,
    #     cellsize * 180.0 / numpy.pi,
    #     dpol,
    #     channel_bandwidth[0],
    # ]
    # w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, pol[0], 1.0]
    # w.wcs.ctype = ["RA---SIN", "DEC--SIN", "STOKES", "FREQ"]
    # w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, frequency[0]]
    # w.naxis = 4
    # w.wcs.radesys = "ICRS"
    # w.wcs.equinox = 2000.0
    #
    # sky_im = Image.constructor(
    #     data=np.zeros((1, 1, npixel, npixel,), dtype=pycrt.getPrecision().value),
    #     polarisation_frame=PolarisationFrame("stokesI"),
    #     wcs=w,
    # )

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


def display_image_list(
        dirty: Image,
        restored: Image,
        restored_list,
        methods,
        sc=None,
        suptitle="",
        normalize=False,
        absolute=False,
):
    chan, pol = 0, 0
    if normalize:
        vmax, vmin = restored_list[0]['pixels'].data.max(), restored_list[0]['pixels'].data.min()
        if absolute:
            vmin = 0.
    else:
        vmax, vmin = None, None
    cm = "Greys"

    fig = plt.figure(figsize=(15, 10))
    axes = fig.subplots(2, 3, sharex=True, sharey=True, subplot_kw={'projection': dirty.image_acc.wcs.sub([1, 2]),
                                                                    'frameon': False})

    dirty_array = np.real(dirty["pixels"].data[chan, pol, :, :])
    ax = axes[0, 0]
    ax.coords[0].set_ticklabel_visible(False)
    ax.coords[0].set_axislabel('')
    if absolute:
        dirty_array = abs(dirty_array)
    im = ax.imshow(dirty_array, origin="lower", cmap=cm, vmax=None, vmin=None)
    ax.set_ylabel(dirty.image_acc.wcs.wcs.ctype[1])
    ax.set_title("Dirty image")
    fig.colorbar(im, orientation="vertical", shrink=0.8, ax=ax)

    clean_array = np.real(restored["pixels"].data[chan, pol, :, :])
    if absolute:
        clean_array = abs(clean_array)
    ax = axes[1, 0]
    im = ax.imshow(clean_array, origin="lower", cmap=cm, vmax=vmax, vmin=vmin)
    ax.set_xlabel(restored.image_acc.wcs.wcs.ctype[0])
    ax.set_ylabel(restored.image_acc.wcs.wcs.ctype[1])
    ax.set_title("CLEAN image")
    fig.colorbar(im, orientation="vertical", shrink=0.8, ax=ax)

    to_fill = (0, 1), (0, 2), (1, 1), (1, 2)

    for idx, im in enumerate(restored_list):
        ax = axes[to_fill[idx]]
        if idx < 2:
            ax.coords[0].set_ticklabel_visible(False)
            ax.coords[0].set_axislabel('')
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')
        else:
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')

        im_array = np.real(im["pixels"].data[chan, pol, :, :])
        if absolute:
            im_array = abs(im_array)
        ims = ax.imshow(im_array, origin="lower", cmap=cm, vmax=vmax, vmin=vmin)
        if idx > 1:
            ax.set_xlabel(im.image_acc.wcs.wcs.ctype[0])
        ax.set_title(methods[idx])
        fig.colorbar(ims, ax=ax, orientation="vertical", shrink=0.8)

    if sc is not None:
        for line in axes:
            for ax in line:
                for component in sc:
                    x, y = skycoord_to_pixel(component.direction, im.image_acc.wcs, 0, "wcs")
                    ax.scatter(x, y, marker="+", color="red", s=1, alpha=.6)

    fig.suptitle(suptitle)
    plt.show()


def MSE(im1: Image, im2: Image):
    assert im1['pixels'].data.shape == im2['pixels'].data.shape, f"Got image 1 of shape {im1['pixels'].data.shape} " \
                                                                 f"and image 2 {im2['pixels'].data.shape}"
    n = im1['pixels'].data.shape[-1] * im1['pixels'].data.shape[-2]
    diff = im1['pixels'].data[0, 0] - im2['pixels'].data[0, 0]
    return np.sum(diff ** 2, axis=(-1, -2)) / n


def MAD(im1: Image, im2: Image):
    assert im1['pixels'].data.shape == im2['pixels'].data.shape, f"Got image 1 of shape {im1['pixels'].data.shape} " \
                                                                 f"and image 2 {im2['pixels'].data.shape}"
    n = im1['pixels'].data.shape[-1] * im1['pixels'].data.shape[-2]
    diff = im1['pixels'].data[0, 0] - im2['pixels'].data[0, 0]
    return np.sum(np.abs(diff)) / n


def RMSE(im1: Image, im2: Image):
    return np.sqrt(MSE(im1, im2))


def display_image_error(
        source: Image,
        dirty: Image,
        im1: Image,
        im2: Image,
        titles,
        sc=None,
        suptitle="",
        normalize=False,
        cm="Greys",
        sc_marker=".",
        sc_color="red",
        sc_size=20
):
    chan, pol = 0, 0
    if normalize:
        vmax, vmin = source['pixels'].data.max(), 0.
    else:
        vmax, vmin = None, None

    fig = plt.figure(figsize=(15, 10))
    axes = fig.subplots(2, 3, sharex=True, sharey=True, subplot_kw={'projection': source.image_acc.wcs.sub([1, 2]),
                                                                    'frameon': False})

    source_array = np.real(source["pixels"].data[chan, pol, :, :])
    ax = axes[0, 0]
    im = ax.imshow(source_array, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_ylabel(source.image_acc.wcs.wcs.ctype[1])
    ax.coords[0].set_ticklabel_visible(False)
    ax.coords[0].set_axislabel('')
    # ax.set_xlabel(source.image_acc.wcs.wcs.ctype[0])
    ax.set_title("Source image")
    if sc is not None:
        for component in sc:
            x, y = skycoord_to_pixel(component.direction, source.image_acc.wcs, 0, "wcs")
            ax.scatter(x, y, marker=sc_marker, color=sc_color, s=sc_size, alpha=.9)
    fig.colorbar(im, orientation="vertical", shrink=0.7, ax=ax)

    dirty_array = np.real(dirty["pixels"].data[chan, pol, :, :])
    ax = axes[1, 0]
    im = ax.imshow(dirty_array, origin="lower", cmap=cm)
    ax.set_ylabel(source.image_acc.wcs.wcs.ctype[1])
    ax.set_xlabel(source.image_acc.wcs.wcs.ctype[0])
    ax.set_title("Dirty image")
    if sc is not None:
        for component in sc:
            x, y = skycoord_to_pixel(component.direction, dirty.image_acc.wcs, 0, "wcs")
            ax.scatter(x, y, marker=sc_marker, color=sc_color, s=sc_size, alpha=.9)
    fig.colorbar(im, orientation="vertical", shrink=0.7, ax=ax)

    to_fill = [[(0, 1), (1, 1)],
               [(0, 2), (1, 2)]]
    for idx, im in enumerate([im1, im2]):
        ax = axes[to_fill[idx][0]]
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[0].set_axislabel('')
        ax.coords[1].set_ticklabel_visible(False)
        ax.coords[1].set_axislabel('')
        im_array = np.real(im["pixels"].data[chan, pol, :, :])
        ims = ax.imshow(im_array, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_title(titles[idx])
        fig.colorbar(ims, ax=ax, orientation="vertical", shrink=0.7)
        if sc is not None:
            for component in sc:
                x, y = skycoord_to_pixel(component.direction, im.image_acc.wcs, 0, "wcs")
                ax.scatter(x, y, marker=sc_marker, color=sc_color, s=sc_size, alpha=.9)

        ax = axes[to_fill[idx][1]]
        ax.coords[1].set_ticklabel_visible(False)
        ax.coords[1].set_axislabel('')
        ax.set_xlabel(im.image_acc.wcs.wcs.ctype[0])
        diff_array = im_array - np.real(
            source['pixels'].data[chan, pol])  # abs(np.real(source['pixels'].data[chan, pol]) - im_array)
        if idx == 0:
            vmax_diff = diff_array.max()
            vmin_diff = diff_array.min()
            vext = max(vmax_diff, -vmin_diff)
        ims = ax.imshow(diff_array, norm=colors.CenteredNorm(halfrange=vext), origin="lower", cmap="bwr")
        # ims = ax.imshow(diff_array, origin="lower", cmap="viridis", vmin=0, vmax=vmax_diff)
        ax.set_title("Difference between source and " + titles[idx])
        fig.colorbar(ims, ax=ax, orientation="vertical", shrink=0.7)
        if sc is not None:
            for component in sc:
                x, y = skycoord_to_pixel(component.direction, im.image_acc.wcs, 0, "wcs")
                ax.scatter(x, y, marker=sc_marker, color=sc_color, s=sc_size, alpha=.9)

    fig.suptitle(suptitle)
    plt.show()


def compare_4_images(
        source: Image,
        dirty: Image,
        im1: Image,
        im2: Image,
        im3: Image,
        im4: Image,
        titles,
        sc=None,
        suptitle="",
        normalize=False,
):
    chan, pol = 0, 0
    if normalize:
        vmax, vmin = source['pixels'].data.max(), 0.
    else:
        vmax, vmin = None, None
    cm = "Greys"

    fig = plt.figure(figsize=(15, 10))
    axes = fig.subplots(2, 3, sharex=True, sharey=True, subplot_kw={'projection': source.image_acc.wcs.sub([1, 2]),
                                                                    'frameon': False})

    source_array = np.real(source["pixels"].data[chan, pol, :, :])
    ax = axes[0, 0]
    im = ax.imshow(source_array, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_ylabel(source.image_acc.wcs.wcs.ctype[1])
    ax.coords[0].set_ticklabel_visible(False)
    ax.coords[0].set_axislabel('')
    # ax.set_xlabel(source.image_acc.wcs.wcs.ctype[0])
    ax.set_title("Source image")
    if sc is not None:
        for component in sc:
            x, y = skycoord_to_pixel(component.direction, source.image_acc.wcs, 0, "wcs")
            ax.scatter(x, y, marker=".", color="red", s=1, alpha=.9)
    fig.colorbar(im, orientation="vertical", shrink=0.8, ax=ax)

    dirty_array = np.real(dirty["pixels"].data[chan, pol, :, :])
    ax = axes[1, 0]
    im = ax.imshow(dirty_array, origin="lower", cmap=cm)
    ax.set_ylabel(source.image_acc.wcs.wcs.ctype[1])
    ax.set_xlabel(source.image_acc.wcs.wcs.ctype[0])
    ax.set_title("Dirty image")
    if sc is not None:
        for component in sc:
            x, y = skycoord_to_pixel(component.direction, dirty.image_acc.wcs, 0, "wcs")
            ax.scatter(x, y, marker=".", color="red", s=1, alpha=.9)
    fig.colorbar(im, orientation="vertical", shrink=0.8, ax=ax)

    to_fill = [(0, 1), (0, 2), (1, 1), (1, 2)]
    for idx, im in enumerate([im1, im2, im3, im4]):
        ax = axes[to_fill[idx]]
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[0].set_axislabel('')
        ax.coords[1].set_ticklabel_visible(False)
        ax.coords[1].set_axislabel('')
        if idx > 1:
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')
            ax.set_xlabel(im.image_acc.wcs.wcs.ctype[0])
        im_array = np.real(im["pixels"].data[chan, pol, :, :])
        ims = ax.imshow(im_array, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_title(titles[idx])
        fig.colorbar(ims, ax=ax, orientation="vertical", shrink=0.8)
        if sc is not None:
            for component in sc:
                x, y = skycoord_to_pixel(component.direction, im.image_acc.wcs, 0, "wcs")
                ax.scatter(x, y, marker=".", color="red", s=1, alpha=.9)

    fig.suptitle(suptitle)
    plt.show()


def compare_3_images(
        source: Image,
        im1: Image,
        im2: Image,
        titles,
        sc=None,
        suptitle="",
        normalize=False,
):
    chan, pol = 0, 0
    if normalize:
        vmax, vmin = source['pixels'].data.max(), 0.
    else:
        vmax, vmin = None, None
    cm = "Greys"

    fig = plt.figure(figsize=(15, 6))
    axes = fig.subplots(1, 3, sharex=True, sharey=True, subplot_kw={'projection': source.image_acc.wcs.sub([1, 2]),
                                                                    'frameon': False})

    source_array = np.real(source["pixels"].data[chan, pol, :, :])
    ax = axes[0]
    im = ax.imshow(source_array, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_ylabel(source.image_acc.wcs.wcs.ctype[1])
    ax.set_xlabel(source.image_acc.wcs.wcs.ctype[0])
    # ax.coords[0].set_ticklabel_visible(False)
    # ax.coords[0].set_axislabel('')
    # ax.set_xlabel(source.image_acc.wcs.wcs.ctype[0])
    ax.set_title("Source image")
    if sc is not None:
        for component in sc:
            x, y = skycoord_to_pixel(component.direction, source.image_acc.wcs, 0, "wcs")
            ax.scatter(x, y, marker=".", color="red", s=1, alpha=.9)
    fig.colorbar(im, orientation="vertical", shrink=0.8, ax=ax)

    for idx, im in enumerate([im1, im2]):
        ax = axes[idx + 1]
        ax.coords[1].set_ticklabel_visible(False)
        ax.coords[1].set_axislabel('')
        ax.set_xlabel(im.image_acc.wcs.wcs.ctype[0])
        im_array = np.real(im["pixels"].data[chan, pol, :, :])
        ims = ax.imshow(im_array, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_title(titles[idx])
        fig.colorbar(ims, ax=ax, orientation="vertical", shrink=0.8)
        if sc is not None:
            for component in sc:
                x, y = skycoord_to_pixel(component.direction, im.image_acc.wcs, 0, "wcs")
                ax.scatter(x, y, marker=".", color="red", s=1, alpha=.9)

    fig.suptitle(suptitle)
    plt.show()


def qq_plot_point_sources(sky_im, clean_comp, pclean_comp, titles):
    support_solution = np.nonzero(sky_im.pixels.data[0, 0])
    source_values = sky_im.pixels.data[0, 0][support_solution]
    clean_values0 = clean_comp.pixels.data[0, 0][support_solution]
    pclean_values0 = pclean_comp.pixels.data[0, 0][support_solution]

    pclean_values = np.zeros_like(pclean_values0)
    clean_values = np.zeros_like(clean_values0)

    for i in range(-1, 2):
        for j in range(-1, 2):
            shift = np.array([i, j]).reshape((2, 1))
            shifted_support = np.array(support_solution) + shift
            pclean_values += pclean_comp.pixels.data[0, 0][tuple(shifted_support)]
            clean_values += clean_comp.pixels.data[0, 0][tuple(shifted_support)]

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.title(titles[0])
    plt.scatter(clean_values0, source_values, marker=".")
    plt.scatter(clean_values, source_values, marker=".")
    plt.plot([0., 1.], [0., 1.], alpha=.5, ls="--", c="r")
    plt.ylabel("Source intensity")
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])

    plt.subplot(122)
    plt.title(titles[1])
    plt.scatter(pclean_values0, source_values, marker=".")
    plt.scatter(pclean_values, source_values, marker=".")
    plt.plot([0., 1.], [0., 1.], alpha=.5, ls="--", c="r")
    plt.ylabel("Source intensity")
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.show()


def qq_plot_point_sources_stacked(sky_im, clean_comp, pclean_comp, reweighted_comp, title=""):
    support_solution = np.nonzero(sky_im.pixels.data[0, 0])
    source_values = sky_im.pixels.data[0, 0][support_solution]
    clean_values0 = clean_comp.pixels.data[0, 0][support_solution]
    pclean_values0 = pclean_comp.pixels.data[0, 0][support_solution]

    pclean_values = np.zeros_like(pclean_values0)
    clean_values = np.zeros_like(clean_values0)
    reweighted_values = np.zeros_like(clean_values0)

    for i in range(-1, 2):
        for j in range(-1, 2):
            shift = np.array([i, j]).reshape((2, 1))
            shifted_support = np.array(support_solution) + shift
            pclean_values += pclean_comp.pixels.data[0, 0][tuple(shifted_support)]
            clean_values += clean_comp.pixels.data[0, 0][tuple(shifted_support)]
            reweighted_values += reweighted_comp.pixels.data[0, 0][tuple(shifted_support)]

    plt.figure(figsize=(6, 6), dpi=200)
    plt.scatter(clean_values, source_values, marker=".", label="CLEAN", lw=1)
    plt.scatter(pclean_values, source_values, marker="+", label="PolyCLEAN", lw=1)
    plt.scatter(reweighted_values, source_values, marker="x", label="PolyCLEAN+", lw=1)
    plt.plot([0., 1.06], [0., 1.06], alpha=.5, ls="--", c="r", lw=1)
    plt.ylabel("Source intensity")
    plt.xlim([0., 1.05])
    plt.ylim([0., 1.05])
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()


def myplot_uvcoverage(vis, title="UV coverage"):
    """Standard plot of uv coverage

    :param vis_list:
    :param plot_file:
    :param kwargs:
    :return:
    """
    gvis = vis.where(vis["flags"] == 0)
    bvis = vis.where(vis["flags"] > 0)
    u = np.array(gvis.visibility_acc.uvw_lambda.reshape((-1, 3))[..., 0].flat)
    v = np.array(gvis.visibility_acc.uvw_lambda.reshape((-1, 3))[..., 1].flat)
    plt.plot(u, v, "o", color="b", markersize=0.5, label="Valid")
    plt.plot(-u, -v, "o", color="b", markersize=0.5)

    u = np.array(bvis.visibility_acc.uvw_lambda.reshape((-1, 3))[..., 0].flat)
    v = np.array(bvis.visibility_acc.uvw_lambda.reshape((-1, 3))[..., 1].flat)
    plt.plot(u, v, "o", color="r", markersize=0.5, label="Non-valid")
    plt.plot(-u, -v, "o", color="r", markersize=0.5)
    plt.xlabel("U (wavelengths)")
    plt.ylabel("V (wavelengths)")
    plt.legend()
    plt.title(title)
    plt.show(block=False)


def plot_analysis_reconstruction(
        source,
        reconstruction,
        normalize=False,
        title="",
        sc=None,
        cm="cubehelix_r",
        suptitle=""
):
    chan, pol = 0, 0

    if normalize:
        vmax, vmin = source['pixels'].data.max(), 0.
    else:
        vmax, vmin = None, None
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
    im_array = np.real(reconstruction["pixels"].data[chan, pol, :, :])
    ims = ax.imshow(im_array, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(ims, ax=ax, orientation="vertical", shrink=0.8)
    if sc is not None:
        for component in sc:
            x, y = skycoord_to_pixel(component.direction, reconstruction.image_acc.wcs, 0, "wcs")
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
            x, y = skycoord_to_pixel(component.direction, reconstruction.image_acc.wcs, 0, "wcs")
            ax.scatter(x, y, marker=".", color="red", s=1, alpha=.9)

    fig.suptitle(suptitle)
    plt.show()


def display_image(image: Image, sc=None, title="", vmax=None, cmap="Greys"):
    chan, pol = 0, 0

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection=image.image_acc.wcs.sub([1, 2]))
    im_array = np.real(image["pixels"].data[chan, pol, :, :])
    im = ax.imshow(im_array, origin="lower", cmap=cmap, vmax=vmax, interpolation="none")
    ax.set_ylabel(image.image_acc.wcs.wcs.ctype[1])
    ax.set_xlabel(image.image_acc.wcs.wcs.ctype[0])
    ax.set_title(title)
    if sc is not None:
        for component in sc:
            x, y = skycoord_to_pixel(component.direction, image.image_acc.wcs, 0, "wcs")
            ax.scatter(x, y, marker=".", color="red", s=1, alpha=.6)
    fig.colorbar(im, orientation="vertical", shrink=0.8, ax=ax)

    plt.show()


def plot_image(
        image,
        cmap="cubehelix_r",
        sc=None,
        title="",
        vmin=None,
        integer_coordinate=True,
        log=False,
):
    chan, pol = 0, 0
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection=image.image_acc.wcs.sub([1, 2]))
    im_array = np.real(image["pixels"].data[chan, pol, :, :])
    im = ax.imshow(im_array, origin="lower", cmap=cmap, interpolation="none", vmin=vmin)
    ax.set_ylabel(image.image_acc.wcs.wcs.ctype[1])
    ax.set_xlabel(image.image_acc.wcs.wcs.ctype[0])
    ax.set_title(title)
    if integer_coordinate:
        def format_coord(x, y):
            col = round(x)
            row = round(y)
            # nrows, ncols = im_array.shape
            # if 0 <= col < ncols and 0 <= row < nrows:
            #     z = im_array[row, col]
            #     return f'x={x:1.4f}, y={y:1.4f}, z={z:1.4f}'
            # else:
            #     return f'x={x:1.4f}, y={y:1.4f}'
            return f'x={x:1.4f}, y={y:1.4f}'

        ax.format_coord = format_coord
    if sc is not None:
        for component in sc:
            x, y = skycoord_to_pixel(component.direction, image.image_acc.wcs, 0, "wcs")
            ax.scatter(x, y, marker=".", color="red", s=1, alpha=.6)
    fig.colorbar(im, orientation="vertical", shrink=0.8, ax=ax)
    plt.show()


def plot_source_reco_diff(source, im1, title="", suptitle="", normalize=False, cmap="cubehelix_r", sc=None):
    chan, pol = 0, 0

    if normalize:
        vmax, vmin = source['pixels'].data.max(), 0.
    else:
        vmax, vmin = None, None
    fig = plt.figure(figsize=(15, 6))
    axes = fig.subplots(1, 3, sharex=True, sharey=True, subplot_kw={'projection': source.image_acc.wcs.sub([1, 2]),
                                                                    'frameon': False})

    ax = axes[0]
    source_array = np.real(source["pixels"].data[chan, pol, :, :])
    im = ax.imshow(source_array, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
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
    ims = ax.imshow(im_array, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
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

    fig.suptitle(suptitle)
    plt.show()


def plot_certificate(
        image,
        cmap="cubehelix_r",
        sc=None,
        title="",
        level=1.,
):
    chan, pol = 0, 0
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection=image.image_acc.wcs.sub([1, 2]))
    im_array = np.real(image["pixels"].data[chan, pol, :, :])
    im = ax.imshow(im_array, origin="lower", cmap=cmap, interpolation="none")
    ax.set_ylabel(image.image_acc.wcs.wcs.ctype[1])
    ax.set_xlabel(image.image_acc.wcs.wcs.ctype[0])
    ax.contour(im_array, levels=[level], colors="c")
    ax.set_title(title)
    if sc is not None:
        for component in sc:
            x, y = skycoord_to_pixel(component.direction, image.image_acc.wcs, 0, "wcs")
            ax.scatter(x, y, marker=".", color="red", s=1, alpha=.6)
    fig.colorbar(im, orientation="vertical", shrink=0.8, ax=ax)
    plt.show()


def plot_4_images(
        im_list,
        title_list,
        suptitle="",
        normalize=False,
        cm='cubehelix_r'
):
    chan, pol = 0, 0
    if normalize:
        vmax, vmin = im_list[0]['pixels'].data.max(), 0.
    else:
        vmax, vmin = None, None

    fig = plt.figure(figsize=(15, 6))
    axes = fig.subplots(1, 4, sharex=True, sharey=True, subplot_kw={'projection': im_list[0].image_acc.wcs.sub([1, 2]),
                                                                    'frameon': False})
    for i in range(4):
        ax = axes[i]
        arr = np.real(im_list[i]["pixels"].data[chan, pol, :, :])
        if i == 3:
            ims = ax.imshow(arr, origin="lower", cmap=cm, vmin=None, vmax=None)
        else:
            ims = ax.imshow(arr, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_ylabel(im_list[i].image_acc.wcs.wcs.ctype[1])
        else:
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')
        ax.set_xlabel(im_list[i].image_acc.wcs.wcs.ctype[0])
        ax.set_title(title_list[i])
        fig.colorbar(ims, orientation="vertical", shrink=0.5, ax=ax)

    fig.suptitle(suptitle)
    plt.show()
