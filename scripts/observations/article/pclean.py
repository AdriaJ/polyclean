import pickle
import os

import numpy as np
import time
import polyclean.image_utils as ut
import polyclean.polyclean as pc
import polyclean.reconstructions as reco
import pyxu.util.complex as pxc

from astropy.coordinates import SkyCoord
from astropy import units as u

from ska_sdp_func_python.imaging import invert_visibility, create_image_from_visibility
from ska_sdp_func_python.image import restore_cube, fit_psf
from ska_sdp_func_python.util import skycoord_to_lmn

lambda_factors = [0.05, 0.02, 0.005]

npixel = 1024
fov_deg = 6.
context = "ng"

nufft_eps = 1e-3
eps = 1e-4
tmax = 120. * 2
min_iter = 5
ms_threshold = 0.9
init_correction_prec = 1e-2
final_correction_prec = min(1e-5, eps)
remove = True
min_correction_steps = 5
max_correction_steps = 1000
diagnostics = False

do_sharp_beam = True

if __name__ == "__main__":

    with open(os.path.join(os.getcwd(), "vis", "vis.pkl"), 'rb') as handle:
        vis = pickle.load(handle)
    print("Selected vis: ", vis.dims)

    phasecentre = vis.phasecentre
    fov = fov_deg * np.pi / 180.
    cellsize = fov / npixel
    image_model = create_image_from_visibility(vis, npixel=npixel, cellsize=cellsize, override_cellsize=False)

    print("Field of view in degrees: {:.3f}".format(fov_deg))

    ## PolyCLEAN reconstruction

    image_model = ut.image_add_ra_dec_grid(image_model)
    directions = SkyCoord(
        ra=image_model.ra_grid.data.ravel() * u.rad,
        dec=image_model.dec_grid.data.ravel() * u.rad,
        frame="icrs", equinox="J2000", )
    direction_cosines = np.stack(skycoord_to_lmn(directions, phasecentre), axis=-1)
    uvwlambda = vis.visibility_acc.uvw_lambda.reshape(-1, 3)
    flags_bool = (vis.weight.data != 0.).reshape(-1)
    flagged_uvwlambda = uvwlambda[flags_bool]

    forwardOp = pc.generatorVisOp(direction_cosines=direction_cosines, vlambda=flagged_uvwlambda,
                                  nufft_eps=nufft_eps, chunked=False)
    start = time.time()
    fOp_lipschitz = forwardOp.estimate_lipschitz(method='svd', tol=1.)  # ~8000 in 18 min in chunked mode, no memory issue
    dt_lipschitz = time.time() - start
    print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)\n".format(dt_lipschitz))

    vis_array = pxc.view_as_real(vis.vis.data.reshape(-1)[flags_bool])
    sum_vis = vis_array.shape[0] // 2
    dirty_array = forwardOp.adjoint(vis_array)

    stop_crit = reco.stop_crit(tmax, min_iter, eps)
    pclean_parameters = {
        # "lambda_factor": lambda_factor,
        "ms_threshold": ms_threshold,
        "init_correction_prec": init_correction_prec,
        "final_correction_prec": final_correction_prec,
        "min_correction_steps": min_correction_steps,
        "max_correction_steps": max_correction_steps,
        "remove_positions": remove,
        "nufft_eps": nufft_eps,
        "show_progress": False, }
    fit_parameters = {
        "stop_crit": stop_crit,
        "positivity_constraint": True,
        "diff_lipschitz": fOp_lipschitz ** 2,
        "precision_rule": lambda k: 10 ** (-k / 10), }

    psf, _ = invert_visibility(vis, image_model, context=context, dopsf=True)
    clean_beam = fit_psf(psf)
    if do_sharp_beam:
        sharp_beam = clean_beam.copy()
        sharp_beam["bmin"] = clean_beam["bmin"] / 2
        sharp_beam["bmaj"] = clean_beam["bmaj"] / 2

    dirty_image = image_model.copy(deep=True)
    dirty_image.pixels.data = dirty_array.reshape(dirty_image.pixels.data.shape) / sum_vis
    with open(os.path.join(os.getcwd(), "dirty.pkl"), 'wb') as handle:
        pickle.dump(dirty_image, handle)

    for factor in lambda_factors:
        lambda_ = factor * np.abs(dirty_array).max()

        # Computations
        pclean = pc.PolyCLEAN(
            data=vis_array,
            uvwlambda=flagged_uvwlambda,
            direction_cosines=direction_cosines,
            lambda_=lambda_,
            **pclean_parameters,
        )
        print(f"PolyCLEAN: Solving for lambda factor {factor:.2f} ...")
        pclean_time = time.time()
        pclean.fit(**fit_parameters)
        print("\tSolved in {:.3f} seconds".format(dt_pclean := time.time() - pclean_time))
        if diagnostics:
            pclean.diagnostics()
        data, hist = pclean.stats()
        pclean_residual = forwardOp.adjoint(vis_array - forwardOp(data["x"]))

        print("PolyCLEAN final DCV: {:.3f}".format(data["dcv"]))
        print("Iterations: {}".format(int(hist['N_iter'][-1])))
        print("Final sparsity of the components: {}".format(np.count_nonzero(data["x"])))

        ## Convolve the images
        pclean_comp = image_model.copy(deep=True)
        pclean_comp.pixels.data[0, 0] = data["x"].reshape((npixel,) * 2)
        pclean_comp_restored = restore_cube(pclean_comp, None, None, clean_beam=clean_beam)

        pclean_residual_im = image_model.copy(deep=True)
        pclean_residual_im.pixels.data = pclean_residual.reshape(pclean_residual_im.pixels.data.shape) / sum_vis
        pclean_restored = restore_cube(pclean_comp, None, pclean_residual_im, clean_beam=clean_beam)

        if do_sharp_beam:
            pclean_comp_sharp = restore_cube(pclean_comp, None, None, clean_beam=sharp_beam)
            pclean_sharp = restore_cube(pclean_comp, None, pclean_residual_im, clean_beam=sharp_beam)

        ## Save the reconstructions
        folder_path = os.path.join(os.getcwd(), 'reco_pkl', str(factor))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(os.path.join(folder_path, "restored.pkl"), 'wb') as handle:
            pickle.dump(pclean_restored, handle)
        if do_sharp_beam:
            with open(os.path.join(folder_path, "restored_sharp.pkl"), 'wb') as handle:
                pickle.dump(pclean_sharp, handle)
