import numpy as np
import time
import logging
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord

from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_func_python.imaging import (
    predict_visibility,
    invert_visibility,
    create_image_from_visibility,
)
from ska_sdp_func_python.image import (
    deconvolve_cube,
    restore_list,
    fit_psf,
)
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame

from rascil.processing_components.simulation import create_test_image

import polyclean.reconstructions as reco
import polyclean.image_utils as ut
import polyclean.polyclean as pc

import pycsou_pywt as pycwt

import pyfwl

import pyxu.operator as pxop
import pyxu.opt.solver as pxsol

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("rascil-logger")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

# Simulation parameters
rmax = 1000.
fov_deg = 6.5
npixel = 256
times = np.zeros([1])
nufft_eps = 1e-3

# LASSO parameters
n_wl = 4
levels = [6, 5, 4, 4]
lambda_factor = .000_1
tmax = 240.
eps = 1e-3
include_dirac = True

# PolyCLEAN parameters
min_iter = 5
ms_threshold = 0.7
init_correction_prec = 5e-2
final_correction_prec = min(1e-4, eps)
remove = True
min_correction_steps = 5
max_correction_steps = 150
positivity_constraint = False
diagnostics = True

# CLEAN parameters
niter = 10_000
context = "ng"

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
    start = time.time()
    mOp_lipschitz = measOp.lipschitz(tol=1., tight=True)
    lipschitz_time = time.time() - start
    print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)".format(lipschitz_time))
    measurements = measOp(m31image.pixels.data.reshape(-1))
    dirty_image = measOp.adjoint(measurements)

    wl_list = ['db' + str(i) for i in range(1, n_wl + 1)]
    swavedec = pycwt.stackedWaveletDec((npixel, npixel), wl_list, levels, include_id=include_dirac)

    forwardOp = measOp * swavedec.T
    forwardOp._lipschitz = mOp_lipschitz

    ### Reconstructions

    methods = ["PolyCLEAN",
               "APGD",
               "CLEAN"]
    stop_crit = reco.stop_crit(tmax, min_iter, eps)
    lambda_ = lambda_factor * np.abs(swavedec(dirty_image)).max()

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
        "diff_lipschitz": mOp_lipschitz ** 2,
        "precision_rule": lambda k: 10 ** (-k / 10),
    }
    pclean = pyfwl.PFWLasso(
        measurements,
        forwardOp,
        lambda_,
        **pclean_parameters,
    )

    print("PolyCLEAN: Solving...")
    pclean_time = time.time()
    pclean.fit(**fit_parameters)
    print("\tSolved in {:.3f} seconds".format(time.time() - pclean_time))
    if diagnostics:
        pclean.diagnostics()
    solution_pc, hist_pc = pclean.stats()
    # Reconstruction in the image domain
    sol_pc = swavedec.adjoint(solution_pc["x"])
    pclean_comp = image_model.copy(deep=True)
    pclean_comp.pixels.data[0, 0] = sol_pc.reshape((npixel,) * 2)

    # APGD
    data_fid_synth = 0.5 * pxop.SquaredL2Norm(dim=forwardOp.shape[0]).argshift(-measurements) * forwardOp
    regul_synth = lambda_ * pxop.L1Norm(dim=forwardOp.shape[1])
    # regul_synth = lambda_ * pyfwl.L1NormPositivityConstraint(shape=(1, forwardOp.shape[1]))  # No positivity constraint here
    apgd = pxsol.PGD(data_fid_synth, regul_synth, show_progress=False)
    fit_parameters_apgd = {
        "x0": np.zeros(forwardOp.shape[1], dtype="float64"),
        "stop_crit": reco.stop_crit(tmax, min_iter, eps,), # value=hist_pc["Memorize[objective_func]"][-1]),
        "track_objective": True,
        "tau": 1 / (mOp_lipschitz ** 2)
    }
    print("APGD: Solving ...")
    start = time.time()
    apgd.fit(**fit_parameters_apgd)
    print("\tSolved in {:.3f} seconds".format(time.time() - start))

    solution_apgd, hist_apgd = apgd.stats()
    dcv = abs(forwardOp.adjoint(measurements - forwardOp(solution_apgd["x"]))).max() / lambda_
    solution_apgd["dcv"] = dcv
    sol_apgd = swavedec.adjoint(solution_apgd["x"])
    apgd_comp = image_model.copy(deep=True)
    apgd_comp.pixels.data[0, 0] = sol_apgd.reshape((npixel,) * 2)

    # MS-CLEAN
    predicted_visi = predict_visibility(vt, m31image, context=context)
    clean_model = create_image_from_visibility(predicted_visi, cellsize=cellsize, npixel=2 * npixel)
    dirty, sumwt_dirty = invert_visibility(predicted_visi, clean_model, context=context)
    psf, sumwt = invert_visibility(predicted_visi, clean_model, context=context, dopsf=True)
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
        algorithm='msclean',
        scales=[0, 3, 10, 30],
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
    print("APGD final value: {:.3e}".format(hist_apgd["Memorize[objective_func]"][-1]))
    print("PolyCLEAN final DCV: {:.3f}".format(solution_pc["dcv"]))
    print("APGD final DCV: {:.3f}".format(solution_apgd["dcv"]))

    plt.figure()
    plt.scatter(hist_pc['duration'], hist_pc['Memorize[objective_func]'], label="PolyCLEAN", s=20, marker="+")
    plt.scatter(hist_apgd['duration'], hist_apgd['Memorize[objective_func]'], label="APGD", s=20, marker="+")
    plt.title('Reconstruction: LASSO objective function')
    plt.legend()
    plt.show()

    # Images
    comps = [pclean_comp, apgd_comp, clean_comp]
    ut.plot_4_images([m31image, ] + comps,
                     ['Source', 'PolyCLEAN', 'APGD', 'CLEAN'],
                     suptitle="Comparison components",
                     normalize=True)

    clean_beam = fit_psf(psf)
    srf = 2
    sharp_beam = clean_beam.copy()
    sharp_beam["bmin"] = clean_beam["bmin"] / srf
    sharp_beam["bmaj"] = clean_beam["bmaj"] / srf
    convolved = restore_list([m31image, ] + comps, None, None, sharp_beam)
    ut.plot_4_images(convolved,
                     ['Source', 'PolyCLEAN', 'APGD', 'CLEAN'],
                     suptitle="Comparison components convolved")

    ## Separate the different images in the wavelet reconstruction
    wbases = list(swavedec._op._block.values())
    wdims = [op.shape[0] for op in wbases]
    cumdims = np.cumsum(np.array(wdims))
    slices = [slice(0, wdims[0]), ] + [slice(cumdims[i], cumdims[i + 1]) for i in range(cumdims.shape[0] - 1)]
    components = [op.adjoint(solution_pc["x"][s]) for op, s in zip(wbases, slices)]

    window = slice(None, None), slice(None, None) # slice(130, 160), slice(100, 150)
    fig = plt.figure()
    for i, comp in enumerate(components):
        ax = plt.subplot(1, n_wl + include_dirac, i + 1)
        ims = ax.imshow(comp.reshape((npixel, npixel))[window], origin="lower", cmap='cubehelix_r')
        ax.set_title("db" + str(i+1) if i < n_wl else "dirac")
        fig.colorbar(ims, orientation="vertical", shrink=0.5, ax=ax)
    fig.suptitle("Different wt components for PolyCLEAN reconstruction")
    plt.show()

    components_apgd = [op.adjoint(solution_apgd["x"][s]) for op, s in zip(wbases, slices)]
    fig = plt.figure()
    for i, comp in enumerate(components_apgd):
        ax = plt.subplot(1, n_wl + include_dirac, i + 1)
        ims = ax.imshow(comp.reshape((npixel, npixel))[window], origin="lower", cmap='cubehelix_r')
        ax.set_title("db" + str(i+1) if i < n_wl else "dirac")
        fig.colorbar(ims, orientation="vertical", shrink=0.5, ax=ax)
    fig.suptitle("Different wt components for APGD reconstruction")
    plt.show()

    # Dual certificate(s)
    certif = np.abs(forwardOp.adjoint(measurements - measOp(sol_pc)))/lambda_
    fig = plt.figure()
    for i, s in enumerate(slices):
        ax = plt.subplot(3, 3, i + 1)
        if i==0 or (include_dirac and i == len(slices) - 1) :
            ims = ax.imshow(certif[s].reshape((npixel, npixel)), origin="lower", cmap='cubehelix_r')
        else:
            ims = ax.imshow(certif[s].reshape(wbases[i].coeff_shape), origin="lower", cmap='cubehelix_r')
        ax.set_title("db" + str(i+1) if i < 8 else "dirac")
        fig.colorbar(ims, orientation="vertical", shrink=0.5, ax=ax)
    plt.suptitle("Dual certificates")
    plt.show()

    # Metrics
    print("\nPolyCLEAN:")
    print("\tFinal sparsity (coefficients): {}".format(np.count_nonzero(solution_pc["x"])))
    print("\tMSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31image, pclean_comp), ut.MAD(m31image, pclean_comp)))
    print("\nAPGD:")
    print("\tFinal sparsity (coefficients): {}".format(np.count_nonzero(solution_apgd["x"])))
    print("\tMSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31image, apgd_comp), ut.MAD(m31image, apgd_comp)))
    print("\nMS-CLEAN:")
    # print("\tMSE {:.2e}".format(ut.MSE(m31image, convolved[-1])))
    print("\tMSE {:.2e}, MAD {:.2e}".format(ut.MSE(m31image, clean_comp), ut.MAD(m31image, clean_comp)))

    print("\tSharp: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(convolved[0], convolved[1]), ut.MAD(convolved[0], convolved[1])))
    print("\tSharp: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(convolved[0], convolved[2]), ut.MAD(convolved[0], convolved[2])))
    print("\tSharp: MSE {:.2e}, MAD {:.2e}".format(ut.MSE(convolved[0], convolved[3]), ut.MAD(convolved[0], convolved[3])))

## To compare: Can we recreate the source image with the wavelet operator ? (without the diracs)
# --> The answer is yes

    # wOp = pycwt.stackedWaveletDec((npixel, npixel), wl_list, levels, include_id=False)
    # source_arr = m31image.pixels.data[0, 0]
    # coeffs_arr = wOp.apply(source_arr.flatten())
    # print("Sparsity index of the coefficients: {:d}/{:d}".format(np.count_nonzero(coeffs_arr), coeffs_arr.size))
    # print("Number of elements with magnitude larger than 1e-3: {:d}/{:d}".format(np.count_nonzero(coeffs_arr[np.abs(coeffs_arr) > 1e-3]), coeffs_arr.size))
    # plt.figure(figsize=(10, 4))
    # plt.subplot(121)
    # plt.hist(coeffs_arr, bins=100, log=True)
    # plt.subplot(122)
    # plt.hist(coeffs_arr[np.abs(coeffs_arr) > 1e-3], bins=100, log=True)
    # plt.show()
    # reco_arr = wOp.adjoint(coeffs_arr).reshape(source_arr.shape)
    # fig = plt.figure(figsize=(12, 5))
    # axes = fig.subplots(1, 2, sharex=True, sharey=True, subplot_kw={'projection': m31image.image_acc.wcs.sub([1, 2]),
    #                                                                 'frameon': False})
    # for ax, image in zip(axes, [source_arr, reco_arr]):
    #     ims = ax.imshow(image, origin="lower", cmap="cubehelix_r")
    #     fig.colorbar(ims)
    # fig.suptitle("Error : {:.3e}".format(np.linalg.norm(source_arr - reco_arr)/source_arr.size))
    # plt.show()

