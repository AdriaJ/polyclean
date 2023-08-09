import logging
import sys
import pickle
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import time

from astropy.wcs.utils import skycoord_to_pixel

from ska_sdp_func_python.imaging import (
    invert_visibility,
    create_image_from_visibility,
)
from ska_sdp_func_python.image import (
    restore_cube,
    fit_psf,
)
from ska_sdp_datamodels.visibility.vis_utils import generate_baselines

from polyclean.clean_utils import mjCLEAN

matplotlib.use("Qt5Agg")

nantennas = 28
ntimes = 50
npixel = 1024
fov_deg = 6.
context = "ng"

algorithm = 'hogbom'
niter = 6000
gain = 0.1
n_major = 10


def plot_2_images(
        im_list,
        title_list,
        suptitle="",
        normalize=False,
        vmin=1e-3,
        norm='log',
        cm='cubehelix_r',
        cmn="CMRmap_r",
        sc=None,
):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    chan, pol = 0, 0
    vmax = im_list[0]['pixels'].data.max()
    fig = plt.figure(figsize=(13, 6))
    axes = fig.subplots(1, 2, sharex=True, sharey=True,
                        subplot_kw={'projection': im_list[0].image_acc.wcs.sub([1, 2]),
                                    'frameon': False})
    if norm == 'sqrt':
        n = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    for i, ax in enumerate(axes.flat):
        # ax = axes[i]
        arr = np.real(im_list[i]["pixels"].data[chan, pol, :, :])
        im_pos = np.ma.masked_array(arr, arr < vmin, fill_value=0.)
        if neg := (arr < -vmin).any():
            im_neg = np.ma.masked_array(arr, arr > -vmin, fill_value=0.)
            if norm == 'sqrt':
                aximn = ax.imshow(-im_neg, origin="lower", cmap=cmn, interpolation='none', alpha=0.8, norm=n)
            else:
                aximn = ax.imshow(-im_neg, origin="lower", cmap=cmn, interpolation='none', alpha=0.8, norm=norm,
                              vmin=vmin, vmax=vmax)
        if norm == 'sqrt':
            aximp = ax.imshow(im_pos, origin="lower", cmap=cm, interpolation='none', norm=n)
        else:
            aximp = ax.imshow(im_pos, origin="lower", cmap=cm, interpolation='none', norm=norm, vmax=vmax, vmin=vmin)
        # ims = ax.imshow(arr, origin="lower", cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        if i == 0:
            ax.set_ylabel(im_list[i].image_acc.wcs.wcs.ctype[1])
        else:
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')
        if sc is not None:
            for component in sc:
                x, y = skycoord_to_pixel(component.direction, im_list[0].image_acc.wcs, 0, "wcs")
                ax.scatter(x, y, marker=".", color="red", s=3, alpha=.9)
        ax.set_xlabel(im_list[i].image_acc.wcs.wcs.ctype[0])
        ax.set_title(title_list[i])
        # fig.colorbar(ims, orientation="vertical", shrink=0.5, ax=ax)
        if i == 1:
            axinsp = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
            cbp = fig.colorbar(aximp, cax=axinsp, orientation="vertical")
            if neg:
                axinsn = inset_axes(axinsp, width="100%", height="100%", loc='center right', borderpad=-4)
                cbn = fig.colorbar(aximn, cax=axinsn, orientation="vertical")
                cbn.ax.invert_yaxis()
    fig.suptitle(suptitle)
    plt.subplots_adjust(top=0.88, bottom=0.11, left=0.06, right=0.9, hspace=0.17, wspace=0.05)
    plt.show()

if __name__ == "__main__":
    log = logging.getLogger("rascil-logger")
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))

    path = "/home/jarret/Documents/EPFL/PhD/ra_data/"
    pklname = path + "bootes.pkl"
    with open(pklname, 'rb') as handle:
        total_vis = pickle.load(handle)

    print(total_vis.dims)

    vis = total_vis.isel({
        "time": slice(0, total_vis.dims["time"], ntimes),
        # "time": slice(total_vis.dims["time"]//10, total_vis.dims["time"]//10 + 1),
        # "frequency": slice(vis.dims["frequency"] // 2, vis.dims["frequency"] // 2 + 1),
    })
    vis = vis.sel({"baselines": list(generate_baselines(nantennas)), })
    print("Selected vis: ", vis.dims)
    # Broken antennas: 12, 13, 16, 17, 47
    # Unusable baseline: (22, 23)
    # ut.myplot_uvcoverage(vis, title="Subsampled UV coverage")

    phasecentre = vis.phasecentre
    fov = fov_deg * np.pi / 180.
    cellsize = fov / npixel
    image_model = create_image_from_visibility(vis, npixel=npixel, cellsize=cellsize, override_cellsize=False)

    print("Field of view in degrees: {:.3f}".format(fov_deg))

    ## CLEAN reconstruction

    # For invert_visibility of Rascil, we need to fill the Nan values: apparently it is ok now

    clean_model = create_image_from_visibility(vis, npixel=2 * npixel, cellsize=cellsize, override_cellsize=False)
    large_dirty, sumwt_dirty = invert_visibility(vis, clean_model, context=context)
    psf, sumwt = invert_visibility(vis, image_model, context=context, dopsf=True)
    clean_beam = fit_psf(psf)
    # ut.plot_image(large_dirty, title="Dirty image")
    dirty = image_model.copy(deep=True)
    dirty['pixels'].data[0, 0, ...] = \
        large_dirty['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]

    print("CLEAN with Major cycles: ...")
    start = time.time()
    tmp_clean_comp_mj, tmp_clean_residual_mj = mjCLEAN(
        large_dirty,
        psf,
        n_major=n_major,
        n_minor=niter,
        vt=vis,
        threshold=0.001,
        fractional_threshold=0.001,
        window_shape="quarter",
        gain=gain,
        algorithm=algorithm,
    )
    clean_comp_mj = image_model.copy(deep=True)
    clean_comp_mj['pixels'].data[0, 0, ...] = \
        tmp_clean_comp_mj['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]
    clean_residual_mj = image_model.copy(deep=True)
    clean_residual_mj['pixels'].data[0, 0, ...] = \
        tmp_clean_residual_mj['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]
    dt_clean_mj = time.time() - start
    print("\nSolved in {:.3f}".format(dt_clean_mj))

    clean_comp_restored_mj = restore_cube(clean_comp_mj, None, None, clean_beam=clean_beam)
    clean_restored_mj = restore_cube(clean_comp_mj, None, clean_residual_mj, clean_beam=clean_beam)

    print("Runtime mjCLEAN: {:.2f}".format(dt_clean_mj))
    print("Sparsity on the components: {:d}".format(np.count_nonzero(clean_comp_mj.pixels.data)))

    folder_path = "/home/jarret/PycharmProjects/polyclean/examples/figures/lofar_ps/clean/" + str(niter)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plot_2_images([clean_comp_restored_mj, clean_restored_mj],
                  ["Components convolved", "Components + residual"],
                  suptitle="mjCLEAN {:d} iterations - {:.2f}s".format(niter, dt_clean_mj),
                  sc=None, vmin=1., norm='log')
    plt.savefig(folder_path + "/log.png")
    plot_2_images([clean_comp_restored_mj, clean_restored_mj],
                  ["Components convolved", "Components + residual"],
                  suptitle="mjCLEAN {:d} iterations - {:.2f}s".format(niter, dt_clean_mj),
                  sc=None, vmin=1., norm='sqrt')
    plt.savefig(folder_path + "/sqrt.png")
    plot_2_images([clean_comp_restored_mj, clean_restored_mj],
                  ["Components convolved", "Components + residual"],
                  suptitle="mjCLEAN {:d} iterations - {:.2f}s".format(niter, dt_clean_mj),
                  sc=None, vmin=1., norm='linear')
    plt.savefig(folder_path + "/linear.png")

    clean_comp_restored_mj.image_acc.export_to_fits(folder_path + "/components.fits")
    clean_restored_mj.image_acc.export_to_fits(folder_path + "/restored.fits")

    do_sharp_beam = True
    if do_sharp_beam:
        sharp_beam = clean_beam.copy()
        sharp_beam["bmin"] = clean_beam["bmin"] / 2
        sharp_beam["bmaj"] = clean_beam["bmaj"] / 2

        clean_comp_sharp_mj = restore_cube(clean_comp_mj, None, None, clean_beam=sharp_beam)
        clean_sharp_mj = restore_cube(clean_comp_mj, None, clean_residual_mj, clean_beam=sharp_beam)
        folder_path += "/sharp_beam"
        os.makedirs(folder_path, exist_ok=True)
        plot_2_images([clean_comp_sharp_mj, clean_sharp_mj],
                      ["Components convolved", "Components + residual"],
                      suptitle="Sharp bean: mjCLEAN {:d} iterations - {:.2f}s".format(niter, dt_clean_mj),
                      sc=None, vmin=1., norm='log')
        plt.savefig(folder_path + "/log.png")
        plot_2_images([clean_comp_sharp_mj, clean_sharp_mj],
                      ["Components convolved", "Components + residual"],
                      suptitle="Sharp bean: mjCLEAN {:d} iterations - {:.2f}s".format(niter, dt_clean_mj),
                      sc=None, vmin=1., norm='sqrt')
        plt.savefig(folder_path + "/sqrt.png")
        plot_2_images([clean_comp_sharp_mj, clean_sharp_mj],
                      ["Components convolved", "Components + residual"],
                      suptitle="Sharp bean: mjCLEAN {:d} iterations - {:.2f}s".format(niter, dt_clean_mj),
                      sc=None, vmin=1., norm='linear')
        plt.savefig(folder_path + "/linear.png")

        clean_comp_sharp_mj.image_acc.export_to_fits(folder_path + "/components.fits")
        clean_sharp_mj.image_acc.export_to_fits(folder_path + "/restored.fits")

    save_dirty = False
    if save_dirty:
        folder = "/home/jarret/PycharmProjects/polyclean/examples/figures/lofar_ps"
        dirty.image_acc.export_to_fits(folder + "/dirty.fits")
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cm = 'cubehelix_r'
        for norm in ['log', 'linear']:
            chan, pol = 0, 0
            vmax = dirty['pixels'].data.max()
            vmin = 1.
            fig = plt.figure(figsize=(8, 6))
            ax = fig.subplots(1, 1, sharex=True, sharey=True,
                                subplot_kw={'projection': dirty.image_acc.wcs.sub([1, 2]),
                                            'frameon': False})
            arr = np.real(dirty["pixels"].data[chan, pol, :, :])
            im_pos = np.ma.masked_array(arr, arr < vmin, fill_value=0.)
            if neg := (arr < -vmin).any():
                im_neg = np.ma.masked_array(arr, arr > -vmin, fill_value=0.)
                aximn = ax.imshow(-im_neg, origin="lower", cmap="ocean_r", interpolation='none', alpha=0.8, norm=norm,
                                  vmin=vmin, vmax=vmax)
            aximp = ax.imshow(im_pos, origin="lower", cmap=cm, interpolation='none', norm=norm, vmax=vmax, vmin=vmin)
            ax.set_ylabel(dirty.image_acc.wcs.wcs.ctype[1])
            ax.set_xlabel(dirty.image_acc.wcs.wcs.ctype[0])
            axinsp = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
            cbp = fig.colorbar(aximp, cax=axinsp, orientation="vertical")
            if neg:
                axinsn = inset_axes(axinsp, width="100%", height="100%", loc='center right', borderpad=-4)
                cbn = fig.colorbar(aximn, cax=axinsn, orientation="vertical")
                cbn.ax.invert_yaxis()
            fig.suptitle("Dirty image")
            plt.subplots_adjust(top=0.88, bottom=0.11, left=0.065, right=0.9, hspace=0.2, wspace=0.2)
            plt.show()
            plt.savefig(folder + '/dirty_' + norm + '.png')
