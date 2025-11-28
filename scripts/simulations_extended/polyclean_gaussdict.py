import numpy as np
import time
import logging
import sys
import os
from astropy import units as u
from astropy.coordinates import SkyCoord

from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_func_python.imaging import predict_visibility, invert_visibility, create_image_from_visibility
from ska_sdp_func_python.image import restore_list, fit_psf
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame

import polyclean
from rascil.processing_components.simulation import create_test_image

import polyclean.reconstructions as reco
import polyclean.image_utils as ut
import polyclean.polyclean as pc
import polyclean.kernels as pck
from polyclean.clean_utils import mjCLEAN

import pyxu.operator as pxop
import pyxu.opt.solver as pxsol
import pyxu.info.ptype as pxt
import pyxu.util.complex as pxc

import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# matplotlib.use("Qt5Agg")



log = logging.getLogger("rascil-logger")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

do_apgd = False

# Simulation parameters
rmax = 1000.
fov_deg = 6.5
npixel = 256
ntimes = 11
times = (np.arange(ntimes) - (ntimes - 1) / 2) * np.pi / (3 * (ntimes - 1) / 2)
nufft_eps = 1e-3

psnrdb = 20
# psnrdb = 10 np.log10(max ** 2 / sigma **2) <=> sigma**2 = max**2 . 10 ** (-psnrdb / 10)

# LASSO parameters
scales = [0, 2, 5, 8]
r = 1.2
scale_bias = [r ** k for k in range(len(scales))]
n_supp = 2
norm_kernels = 1
lambda_factor = .02
tmax = 240.
eps = 1e-5

# PolyCLEAN parameters
min_iter = 5
ms_threshold = 0.8
init_correction_prec = 5e-2
final_correction_prec = min(1e-4, eps)
remove = True
min_correction_steps = 5
max_correction_steps = 150
positivity_constraint = True
diagnostics = False

# CLEAN parameters
niter = 10_000
n_major = 10
gain = .7
algorithm = 'msclean'
context = "ng"

scales_msclean = [0, 3, 10, 30]


class GaussPolyCLEAN(polyclean.PolyCLEAN):
    def __init__(self,
                 scales: list,
                 kernel_bias: list = None,
                 n_supp: int = 2,
                 norm_kernels: int = 2,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_bias = kernel_bias
        self.kernels = pck.stackedKernels((npixel,) * 2, scales,
                                          n_supp=n_supp,
                                          tight_lipschitz=False,
                                          verbose=True,
                                          norm=norm_kernels,
                                          bias_list=kernel_bias)
        self.measOp = self.forwardOp
        self.forwardOp = self.measOp * self.kernels

    def rs_forwardOp(self, support_indices: pxt.NDArray) -> pxt.OpT:
        if support_indices.size == 0:
            return pxop.NullOp(shape=(self.forwardOp.shape[0], 0))
        else:
            tmp = np.zeros(self.kernels.shape[1])
            tmp[support_indices] = 1.
            supp = np.where(self.kernels(tmp) != 0)[0]
            ss = pxop.SubSample(self.kernels.shape[0], supp)
            op = pc.generatorVisOp(self._direction_cosines[supp, :],
                                   self._uvw,
                                   self._nufft_eps,
                                   chunked=self._chunked,
                                   )
            injection = pxop.SubSample(self.kernels.shape[1], support_indices).T
            return op * ss * self.kernels * injection

def truncate_colormap(cmap, minval, maxval, n=100):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mplc.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_3_images(
        im_list,
        title_list,
        suptitle="",
        normalize=True,
        offset_cm=0.,
        vlim=None,
        alpha=.8,
        ):
    chan, pol = 0, 0
    set_vlim = vlim is None
    if normalize:
        vmax = max([np.abs(im.pixels.data).max() for im in im_list])
        if set_vlim:
            vlim = max(0, -min([im.pixels.data.min() for im in im_list]))
    else:
        vmax = None
    vmin = 0
    any_lim = False

    fig = plt.figure(figsize=(15, 5))
    axes = fig.subplots(1, 3, sharex=True, sharey=True,
                        subplot_kw={'projection': im_list[0].image_acc.wcs.sub([1, 2]), 'frameon': False})
    for i in range(3):
        ax = axes[i]
        arr = np.real(im_list[i]["pixels"].data[chan, pol, :, :])
        if not normalize:
            if i == 0:
                vmax = np.abs(arr).max()
                if set_vlim:
                    vlim = max(0, -arr.min())
            else:
                vmax = max([np.abs(im.pixels.data).max() for im in im_list[1:]])
                if set_vlim:
                    vlim = max(0, -min([im.pixels.data.min() for im in im_list[1:]]))
        # ims = ax.imshow(arr, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
        im_pos = np.ma.masked_array(arr, arr < vlim, fill_value=vlim)
        if (arr < vlim).any():
            im_neg = np.ma.masked_array(arr, arr > vlim, fill_value=vlim)
            cmapn = truncate_colormap('Greys', offset_cm, 1.-offset_cm)
            aximn = ax.imshow(im_neg, origin="lower", cmap=cmapn, interpolation='none', alpha=alpha,
                              vmin=-vlim, vmax=vlim)
            any_lim = True
        cmapp = truncate_colormap('hot', offset_cm, 1.)
        aximp = ax.imshow(im_pos, origin="lower", cmap=cmapp, interpolation='none', vmax=vmax, vmin=vlim)
        if i == 0:
            ax.set_ylabel(im_list[i].image_acc.wcs.wcs.ctype[1])
        else:
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')
        ax.set_xlabel(im_list[i].image_acc.wcs.wcs.ctype[0])
        ax.set_title(title_list[i])
        # fig.colorbar(ims, orientation="vertical", shrink=0.5, ax=ax)
        if i == 2:
            axinsp = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
            cbp = fig.colorbar(aximp, cax=axinsp, orientation="vertical")
            if any_lim:
                axinsn = inset_axes(axinsp, width="100%", height="100%", loc='center right', borderpad=-4)
                cbn = fig.colorbar(aximn, cax=axinsn, orientation="vertical")
        if not normalize and i == 0:
            axinsp = inset_axes(ax, width="3%", height="100%", loc='center left', borderpad=-8)
            cbp = fig.colorbar(aximp, cax=axinsp, orientation="vertical")
            if any_lim:
                axinsn = inset_axes(axinsp, width="100%", height="100%", loc='center left', borderpad=-4)
                cbn = fig.colorbar(aximn, cax=axinsn, orientation="vertical")

    fig.suptitle(suptitle)
    # plt.subplots_adjust(
    #     top=0.918,
    #     bottom=0.027,
    #     left=0.047,
    #     right=0.991,
    #     hspace=0.2,
    #     wspace=0.023)
    plt.show()


if __name__ == "__main__":
    # Image simulation
    frequency = np.array([1e8])
    channel_bandwidth = np.array([1e6])
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000")
    fov = fov_deg * np.pi / 180.
    cellsize = fov / npixel
    m31image = create_test_image(cellsize=cellsize, frequency=frequency, phasecentre=phasecentre, )
    m31image = ut.image_add_ra_dec_grid(m31image)
    directions = SkyCoord(ra=m31image.ra_grid.data.ravel() * u.rad, dec=m31image.dec_grid.data.ravel() * u.rad,
                          frame="icrs", equinox="J2000")
    direction_cosines = np.stack(skycoord_to_lmn(directions, phasecentre), axis=-1)

    # Visibility simulations
    lowr3 = create_named_configuration("LOWBD2", rmax=rmax)
    vt = create_visibility(lowr3, times, frequency, channel_bandwidth=channel_bandwidth, weight=1.0,
                           phasecentre=phasecentre, polarisation_frame=PolarisationFrame("stokesI"))
    image_model = create_image_from_visibility(vt, cellsize=cellsize, npixel=npixel)  # verify same size as m31image
    uvwlambda = vt.visibility_acc.uvw_lambda.reshape(-1, 3)
    flags_bool = np.any(uvwlambda != 0., axis=-1)
    flagged_uvwlambda = uvwlambda[flags_bool]
    measOp = pc.generatorVisOp(direction_cosines=direction_cosines,
                               vlambda=flagged_uvwlambda,
                               nufft_eps=nufft_eps)

    noiseless_measurements = measOp(m31image.pixels.data.reshape(-1))
    noise_scale = np.abs(noiseless_measurements).max() * 10 ** (-psnrdb / 20) / np.sqrt(2)
    noise = np.random.normal(0, noise_scale, noiseless_measurements.shape)
    # print(20 * np.log10(np.abs(noiseless_measurements).max()/(noise.std()*np.sqrt(2))))
    measurements = noiseless_measurements + noise

    dirty_image = measOp.adjoint(measurements)

    ## rascil clean bias
    # skernels = pck.stackedKernels((npixel,) * 2, scales,
    #                               n_supp=n_supp,
    #                               tight_lipschitz=False,
    #                               verbose=True,
    #                               norm=norm_kernels,
    #                               bias_list=None)
    # psf = measOp.adjoint(np.ones(measOp.shape[0]))
    # conv1 = skernels.adjoint(psf)
    # components_conv1 = pck.stack_list_sources(conv1, skernels, scales, n_supp)
    # components_conv2 = []
    # for i in range(len(components_conv1)):
    #     tmp = skernels.adjoint(components_conv1[i].flatten())
    #     components_conv2.append(pck.stack_list_sources(tmp, skernels, scales, n_supp)[i])
    #
    # rascil_bias = [1/comp.max() for comp in components_conv2]
    # scale_bias = rascil_bias

    # i=3
    # fig = plt.figure(figsize=(14, 4))
    # axes = fig.subplots(1, 3, sharex=True, sharey=True)
    # ax = axes[0]
    # ax.imshow(psf.reshape((npixel, )*2), interpolation="none", norm='log')
    # ax = axes[1]
    # ax.imshow(components_conv1[i], interpolation="none", norm='log')
    # ax = axes[2]
    # ax.imshow(components_conv2[i], interpolation="none", norm='log')
    # plt.show()

    skernels = pck.stackedKernels((npixel,) * 2, scales,
                                  n_supp=n_supp,
                                  tight_lipschitz=False,
                                  verbose=True,
                                  norm=norm_kernels,
                                  bias_list=scale_bias)
    forwardOp = measOp * skernels
    start = time.time()
    fOp_lipschitz = forwardOp.estimate_lipschitz(method='svd', tol=1.)
    lipschitz_time = time.time() - start
    print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)".format(lipschitz_time))

    # unbiased_kernels = pck.stackedKernels((npixel,) * 2, scales,
    #                                       n_supp=n_supp,
    #                                       tight_lipschitz=False,
    #                                       verbose=True,
    #                                       norm=norm_kernels)
    ### Reconstructions

    methods = ["PolyCLEAN",
               "APGD",
               "CLEAN"]
    stop_crit = reco.stop_crit(tmax, min_iter, eps)
    lambda_ = lambda_factor * np.abs(skernels.adjoint(dirty_image)).max()

    # Polyatomic FW
    pclean_parameters = {
        # "lambda_factor": lambda_factor,
        "ms_threshold": ms_threshold,
        "init_correction_prec": init_correction_prec,
        "final_correction_prec": final_correction_prec,
        "min_correction_steps": min_correction_steps,
        "max_correction_steps": max_correction_steps,
        "remove_positions": remove,
        "show_progress": False,
        "shortcut": False,
    }
    fit_parameters = {
        "stop_crit": stop_crit,
        "positivity_constraint": positivity_constraint,
        "diff_lipschitz": fOp_lipschitz ** 2,
        "precision_rule": lambda k: 10 ** (-k / 10),
    }
    # pclean = pyfwl.PFWLasso(
    #     measurements,
    #     forwardOp,
    #     lambda_,
    #     **pclean_parameters,
    # )

    # print("PolyCLEAN: Solving...")
    # pclean_time = time.time()
    # pclean.fit(**fit_parameters)
    # print("\tSolved in {:.3f} seconds".format(time.time() - pclean_time))
    # if diagnostics:
    #     pclean.diagnostics()
    # solution_pc, hist_pc = pclean.stats()
    # # Reconstruction in the image domain
    # sol_pc = skernels(solution_pc["x"])
    # pclean_comp = image_model.copy(deep=True)
    # pclean_comp.pixels.data[0, 0] = sol_pc.reshape((npixel,) * 2)

    print("GaussPolyCLEAN: Solving...")
    pclean = GaussPolyCLEAN(
        scales,
        data=measurements,
        uvwlambda=flagged_uvwlambda,
        direction_cosines=direction_cosines,
        kernel_bias=scale_bias,
        n_supp=n_supp,
        norm_kernels=norm_kernels,
        lambda_=lambda_,
        **pclean_parameters,
    )

    pclean_time = time.time()
    pclean.fit(**fit_parameters)
    print("\tSolved in {:.3f} seconds".format(dt_pclean := time.time() - pclean_time))
    if diagnostics:
        pclean.diagnostics()
    solution_pc, hist_pc = pclean.stats()
    # Reconstruction in the image domain
    sol_pc = skernels(solution_pc["x"])
    pclean_comp = image_model.copy(deep=True)
    pclean_comp.pixels.data[0, 0] = sol_pc.reshape((npixel,) * 2)
    pclean_residual = measOp.adjoint(measurements - measOp(sol_pc))
    sum_vis = measurements.shape[0] // 2
    pclean_residual_im = image_model.copy(deep=True)
    pclean_residual_im.pixels.data[0, 0] = pclean_residual.reshape((npixel,) * 2) / sum_vis

    # APGD
    apgd_comp = image_model.copy(deep=True)
    sol_apgd = np.zeros_like(sol_pc)
    apgd_residual_im = apgd_comp.copy(deep=True)
    if do_apgd:
        data_fid_synth = 0.5 * pxop.SquaredL2Norm(dim=forwardOp.shape[0]).argshift(-measurements) * forwardOp
        regul_synth = lambda_ * pxop.L1Norm(dim=forwardOp.shape[1])
        # regul_synth = lambda_ * pyfwl.L1NormPositivityConstraint(shape=(1, forwardOp.shape[1]))  # No positivity constraint here
        apgd = pxsol.PGD(data_fid_synth, regul_synth, show_progress=False)
        import pyxu.opt.stop as pxos

        supp_count = pxos.AbsError(.1, 'x', f=lambda x: 1 + np.count_nonzero(x))
        fit_parameters_apgd = {
            "x0": np.zeros(forwardOp.shape[1], dtype="float64"),
            "stop_crit": reco.stop_crit(tmax, min_iter, eps,
                                        value=hist_pc["Memorize[objective_func]"][-1]) | supp_count,
            "track_objective": True,
            "tau": 1 / (fOp_lipschitz ** 2)
        }
        print("APGD: Solving ...")
        start = time.time()
        apgd.fit(**fit_parameters_apgd)
        print("\tSolved in {:.3f} seconds".format(time.time() - start))

        solution_apgd, hist_apgd = apgd.stats()
        dcv = abs(forwardOp.adjoint(measurements - forwardOp(solution_apgd["x"]))).max() / lambda_
        solution_apgd["dcv"] = dcv
        sol_apgd = skernels(solution_apgd["x"])
        # apgd_comp = image_model.copy(deep=True)
        apgd_comp.pixels.data[0, 0] = sol_apgd.reshape((npixel,) * 2)
        apgd_residual_im.pixels.data[0, 0] = measOp.adjoint(measurements - measOp(sol_apgd)).reshape(
            (npixel,) * 2) / sum_vis

    # plt.figure()
    # plt.plot((hist_apgd['AbsError[x]']-1).astype(int)[1:], lw=0, marker='.')
    # plt.ylabel("Support size")
    # plt.xlabel("Iteration count")
    # plt.ylim(bottom=0)
    # plt.show()

    # MS-CLEAN
    predicted_visi = predict_visibility(vt, m31image, context=context)
    predicted_visi.vis.data[predicted_visi.visibility_acc.flagged_weight.astype(bool)] += pxc.view_as_complex(noise)
    clean_model = create_image_from_visibility(predicted_visi, cellsize=cellsize, npixel=2 * npixel)
    dirty, sumwt_dirty = invert_visibility(predicted_visi, clean_model, context=context)
    psf, sumwt = invert_visibility(predicted_visi, clean_model, context=context, dopsf=True)
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
    #     algorithm='msclean',
    #     scales=scales_msclean,
    # )
    tmp_clean_comp, tmp_clean_residual = mjCLEAN(
        dirty,
        psf,
        n_major=n_major,
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

    ### Reconstruction quality
    print("\nPolyCLEAN final value: {:.3e}".format(hist_pc["Memorize[objective_func]"][-1]))
    if do_apgd:
        print("APGD final value: {:.3e}".format(hist_apgd["Memorize[objective_func]"][-1]))
    print("PolyCLEAN final DCV: {:.3f}".format(solution_pc["dcv"]))
    if do_apgd:
        print("APGD final DCV: {:.3f}".format(solution_apgd["dcv"]))

    if do_apgd:
        plt.figure()
        plt.scatter(hist_pc['duration'], hist_pc['Memorize[objective_func]'], label="PolyCLEAN", s=20, marker="+")
        plt.scatter(hist_apgd['duration'], hist_apgd['Memorize[objective_func]'], label="APGD", s=20, marker="+")
        plt.title('Reconstruction: LASSO objective function')
        plt.legend()
        plt.show()

    # Images
    ## Convolution with sharp beam
    clean_beam = fit_psf(psf)
    srf = 2
    sharp_beam = clean_beam.copy()
    sharp_beam["bmin"] = clean_beam["bmin"] / srf
    sharp_beam["bmaj"] = clean_beam["bmaj"] / srf
    m31_convolved = restore_list([m31image, ], None, None, clean_beam=sharp_beam)[0]

    # comps = [pclean_comp, apgd_comp, clean_comp]
    # residuals = [pclean_residual_im, apgd_residual_im, clean_residual]
    # convolved = restore_list(comps, None, None, sharp_beam)
    # convolved_res = restore_list(comps, None, residuals, sharp_beam)
    #
    # ut.plot_4_images([m31image, ] + comps,
    #                  ['Source', 'PolyCLEAN', 'APGD', 'CLEAN'],
    #                  suptitle="Comparison components",
    #                  normalize=True)
    # ut.plot_4_images([m31_convolved, ] + convolved,
    #                  ['Source', 'PolyCLEAN', 'APGD', 'CLEAN'],
    #                  suptitle="Comparison components convolved sharp")
    # ut.plot_4_images([m31_convolved, ] + convolved_res,
    #                  ['Source', 'PolyCLEAN', 'APGD', 'CLEAN'],
    #                  suptitle="Comparison restored sharp (components + residual)")
    # ut.plot_4_images([m31image, ] + residuals,
    #                  ['Source', 'PolyCLEAN', 'APGD', 'CLEAN'],
    #                  suptitle="Residuals")

    comps = [pclean_comp, clean_comp]
    residuals = [pclean_residual_im, clean_residual]
    convolved = restore_list(comps, None, None, sharp_beam)
    convolved_res = restore_list(comps, None, residuals, sharp_beam)
    cropped_dirty, _ = invert_visibility(predicted_visi, image_model, context=context)

    save = False
    if save:
        folder_path = "/home/jarret/PycharmProjects/polyclean/figures/ext_sources/setup1"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        plot_3_images([m31image, ] + comps,
                      ['Source', 'PolyCLEAN', 'CLEAN'],
                      suptitle="Comparison components",
                      normalize=True, vlim=0.3, alpha=1)
        plt.savefig(folder_path + "/comps.pdf")
        plt.savefig(folder_path + "/comps.png")
        plot_3_images([m31_convolved, ] + convolved,
                      ['Source', 'PolyCLEAN', 'CLEAN'],
                      suptitle="Comparison components convolved sharp",
                      normalize=True)
        plt.savefig(folder_path + "/comps_conv.pdf")
        plt.savefig(folder_path + "/comps_conv.png")
        plot_3_images([m31_convolved, ] + convolved_res,
                      ['Source', 'PolyCLEAN', 'CLEAN'],
                      suptitle="Comparison restored sharp (components + residual)",
                      normalize=True, offset_cm=0.05)
        plt.savefig(folder_path + "/restored.pdf")
        plt.savefig(folder_path + "/restored.png")

        plot_3_images([cropped_dirty, ] + convolved_res,
                      ['Dirty image', 'PolyCLEAN', 'CLEAN'],
                      suptitle="Comparison restored sharp (components + residual)",
                      normalize=False, offset_cm=0.05)
        plot_3_images([cropped_dirty, ] + residuals,
                      ['Dirty', 'PolyCLEAN', 'CLEAN'],
                      suptitle="Residuals",
                      normalize=False)
        plt.savefig(folder_path + "/residuals.pdf")
        plt.savefig(folder_path + "/residuals.png")

        # basis functions PolyCLEAN
        coeffs = np.zeros(skernels.shape[1])
        cores = pck.stack_list_coeffs(coeffs, skernels, scales)
        offset = [(1, 1), (1, 3), (3, 1), (3, 3)]
        for core, off in zip(cores, offset):
            size = core.size
            index = off[0] * size // 4 + off[1] * np.round(size ** .5).astype(int) // 4
            core[index] = 1.
        coeffs = np.hstack(cores)
        test_im = image_model.copy(deep=True)
        test_im.pixels.data[0, 0] = np.log10(skernels(coeffs)).reshape((npixel, ) * 2)
        ut.plot_image(test_im)
        test_im = image_model.copy(deep=True)
        test_im.pixels.data[0, 0] = (skernels(coeffs)).reshape((npixel, ) * 2)
        ut.plot_image(test_im)

        # basis function CLEAN
        from ska_sdp_func_python.image.cleaners import create_scalestack, convolve_scalestack
        scaleshape = [len(scales_msclean), npixel, npixel]
        scalestack = create_scalestack(scaleshape, scales_msclean, norm=True)
        test_images = []
        for off in offset:
            tmp = np.zeros((npixel, ) * 2)
            # index = npixel * npixel // 4 * off[0] + npixel // 4 * off[1]
            tmp[npixel // 4 * off[0], npixel // 4 * off[1]] = 1.
            test_images.append(tmp)
        res_scalestack = np.zeros((npixel, ) * 2)
        # res = convolve_scalestack(scalestack, test_images[0])
        # for i in range(4):
        #     plt.figure()
        #     plt.imshow(scalestack[i][110:150, 110:150])
        #     plt.show()
        for i, im in enumerate(test_images):
            res_scalestack += convolve_scalestack(scalestack, im)[i]
        test_im = image_model.copy(deep=True)
        test_im.pixels.data[0, 0] = np.log10(res_scalestack, out=-6 * np.ones((npixel, npixel)), where=res_scalestack>1e-6, )
        ut.plot_image(test_im)
        test_im = image_model.copy(deep=True)
        test_im.pixels.data[0, 0] = res_scalestack
        ut.plot_image(test_im)

    ## Separate the different images in the dictionary decomposition

    components_pc = pck.stack_list_components(solution_pc['x'], skernels, scales, (npixel, npixel))
    if do_apgd:
        components_apgd = pck.stack_list_components(solution_apgd['x'], skernels, scales, (npixel, npixel))

    n_comp = len(components_pc)
    if do_apgd:
        vmin, vmax = min(sol_pc.min(), sol_apgd.min()), max(sol_pc.max(), sol_apgd.max())
    else:
        vmin, vmax = sol_pc.min(), sol_pc.max()

    print("Min and max values of the solution images: {:.2f} / {:.2f}".format(vmin, vmax))
    fig = plt.figure(figsize=(4 * n_comp, 9))
    axes = fig.subplots(2, n_comp, sharex=True, sharey=True)
    for i in range(n_comp):
        ax = axes[0, i]
        ims = ax.imshow(components_pc[i], origin="lower", cmap='cubehelix_r', vmin=vmin, vmax=vmax)
        ax.set_title("scale " + str(scales[i]))
        # fig.colorbar(ims, orientation="vertical", shrink=0.5, ax=ax)
    # fig.suptitle("Different components for PolyCLEAN reconstruction")

    # vmin, vmax = sol_apgd.min(), sol_apgd.max()
    # fig = plt.figure(figsize=(4*n_comp, 4))
    # axes = fig.subplots(1, n_comp, sharex=True, sharey=True)
    if do_apgd:
        for i in range(n_comp):
            ax = axes[1, i]
            ims = ax.imshow(components_apgd[i], origin="lower", cmap='cubehelix_r', vmin=vmin, vmax=vmax)
            ax.set_title("scale " + str(scales[i]))
            # fig.colorbar(ims, orientation="vertical", shrink=0.5, ax=ax)
    fig.suptitle("PolyCLEAN (up) and APGD (bottom) components")
    plt.show()

    # sources location & intensities
    sources_list = pck.stack_list_sources(solution_pc['x'], skernels, scales, n_supp)
    fig = plt.figure(figsize=(4 * n_comp, 4))
    marker = '.'
    axes = fig.subplots(1, n_comp, sharex=True, sharey=True)
    for i in range(n_comp):
        ax = axes[i]
        ims = ax.scatter(*np.where(sources_list[i].T > 0), marker=marker, s=10, color='r')
        ims = ax.scatter(*np.where(sources_list[i].T < 0), marker=marker, s=10, color='b')
        ax.set_xlim([0, npixel])
        ax.set_ylim([0, npixel])
        ax.set_title("scale " + str(scales[i]))
    plt.suptitle("Locations of the kernels")
    plt.show()

    # Dual certificate(s)
    plot_dual_certif = False
    if plot_dual_certif:
        vmin, vmax = None, None
        certif = np.abs(forwardOp.adjoint(measurements - measOp(sol_pc))) / lambda_  # belong to the coefficients domain
        subcertif_list = pck.stack_list_sources(certif, skernels, scales, n_supp)
        fig = plt.figure(figsize=(4 * n_comp, 9))
        axes = fig.subplots(2, n_comp, sharex=True, sharey=True)
        for i in range(n_comp):
            ax = axes[0, i]
            ims = ax.imshow(subcertif_list[i], origin="lower", cmap='cubehelix_r', vmin=vmin, vmax=vmax)
            ax.set_title("scale " + str(scales[i]))
            ax = axes[1, i]
            ims = ax.imshow(np.where(subcertif_list[i] > 1., subcertif_list[i], 0), origin="lower", cmap='cubehelix_r',
                            vmin=vmin, vmax=vmax)
            # ax.set_title("scale " + str(scales[i]))
            # fig.colorbar(ims, orientation="vertical", shrink=0.5, ax=ax)
        plt.suptitle("Dual certificates")
        plt.show()

    m31_convolved = restore_list([m31image, ], None, None, clean_beam=sharp_beam)[0]
    # Metrics
    print("\nPolyCLEAN:")
    print("\tFinal sparsity (coefficients): {}/{}".format(np.count_nonzero(solution_pc["x"]), skernels.shape[1]))
    print("\tComponents: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31image, pclean_comp), ut.MAD(m31image, pclean_comp)))
    print("\tComponents convolved sharp: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31_convolved, convolved[0]),
                                                                        ut.MAD(m31_convolved, convolved[0])))
    print("\tComponents restored sharp: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31_convolved, convolved_res[0]),
                                                                       ut.MAD(m31_convolved, convolved_res[0])))
    print("\tTotal weight: {:.2f}/{:.2f}".format(sol_pc.sum(), m31image.pixels.data.sum()))

    if do_apgd:
        print("\nAPGD:")
        print("\tFinal sparsity (coefficients): {}/{}".format(np.count_nonzero(solution_apgd["x"]), skernels.shape[1]))
        print("\tMSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31image, apgd_comp), ut.MAD(m31image, apgd_comp)))
        print("\tSharp: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31_convolved, convolved[1]),
                                                       ut.MAD(m31_convolved, convolved[1])))
    print("\nMS-CLEAN:")
    print("\tComponents: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31image, clean_comp), ut.MAD(m31image, clean_comp)))
    print("\tComponents convolved sharp: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31_convolved, convolved[-1]),
                                                                        ut.MAD(m31_convolved, convolved[-1])))
    print("\tComponents restored sharp: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31_convolved, convolved_res[-1]),
                                                                       ut.MAD(m31_convolved, convolved_res[-1])))
    print("\tTotal weight: {:.2f}/{:.2f}".format(clean_comp.pixels.data.sum(), m31image.pixels.data.sum()))


    if save:
        from scripts.observations.pclean import truncate_colormap
        def plot_1_image(image, title="", cmaps=['hot', 'Greys'], alpha=.95, offset_cm=0., symm=True, ticks=None, vlim=None):
            if ticks is None:
                ticks = [1, 500, 1000, 2000, 3000, 4000]
            arr = image.pixels.data[0, 0]

            fig = plt.figure(figsize=(12, 10))
            ax = fig.subplots(1, 1, subplot_kw={'projection': image.image_acc.wcs.sub([1, 2]), 'frameon': False})
            ax.set_xlabel(image.image_acc.wcs.wcs.ctype[0])
            ax.set_ylabel(image.image_acc.wcs.wcs.ctype[1])
            if vlim is None:
                vlim = -arr.min() if symm else 0.
            mask_comp = np.ma.masked_array(arr, arr < vlim, fill_value=vlim)
            mask_res = np.ma.masked_array(arr, arr > vlim, fill_value=vlim)
            cmapc = truncate_colormap(cmaps[0], offset_cm, 1.)
            aximc = ax.imshow(mask_comp, origin="lower", cmap=cmapc, interpolation='none', alpha=alpha,
                              norm=mplc.PowerNorm(gamma=1., vmin=vlim, vmax=1. * mask_comp.max()))
            cmapr = truncate_colormap(cmaps[1], 0., 1 - offset_cm)
            aximr = ax.imshow(mask_res, origin="lower", interpolation='none', alpha=alpha,
                              cmap=cmapr, norm='linear', vmin=-vlim, vmax=vlim)
            # norm=symm_sqrt_norm(-vlim, vlim))
            axinsc = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
            cbc = fig.colorbar(aximc, cax=axinsc,
                               orientation="vertical", ticks=[round(11 * vlim)/10] + ticks)
            axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center right', borderpad=-6)
            cbr = fig.colorbar(aximr, cax=axinsr, orientation="vertical")
            fig.suptitle(title)
            plt.subplots_adjust(top=0.92, bottom=0.08, left=0.0, right=0.93, hspace=0.15, wspace=0.15)
            fig.show()

        folder_path = "/home/jarret/PycharmProjects/polyclean/figures/ext_sources/setup1"

        plot_1_image(cropped_dirty, title="Dirty image", ticks=[10, 15, 20, 25])
        plt.savefig(folder_path + "/dirty.png")

        plot_1_image(m31image, vlim=0.3, title="Source", ticks=[0.5, 0.8, 1.])
        plt.savefig(folder_path + "/source.png")