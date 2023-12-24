import pickle
import os
import yaml
import time
import numpy as np
import argparse

from astropy.coordinates import SkyCoord
import astropy.units as u
from ska_sdp_func_python.image import fit_psf, restore_cube
from ska_sdp_func_python.imaging import create_image_from_visibility, invert_visibility
from ska_sdp_func_python.util import skycoord_to_lmn

import pyxu.util.complex as pxc
import pyxu.operator as pxop
import pyxu.opt.solver as pxsol

import pyfwl

import polyclean as pc
import polyclean.image_utils as ut

TMP_DATA_DIR = 'tmpdir'
metrics = ['time', 'mse', 'mad', 'objf', 'dcv']

if __name__ == "__main__":
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    with open(os.path.join(TMP_DATA_DIR, 'data.pkl'), 'rb') as file:
        predicted_visi = pickle.load(file)  # Rascil Visibility instance
    with open(os.path.join(TMP_DATA_DIR, 'gtimage.pkl'), 'rb') as file:
        sky_im = pickle.load(file)  # Rascil Image instance

    parser = argparse.ArgumentParser(
        description='Performs the LASSO reconstructions with Python: PolyCLEAN, APGD and Monoatomic FW.')
    parser.add_argument('--save_path', type=str, default=None, required=False,
                        help="Location where to save the produced images. If None, the images are not saved.")
    args = parser.parse_args()
    save_images = args.save_path is not None

    uvwlambda = predicted_visi.visibility_acc.uvw_lambda.reshape(-1, 3)
    flags_bool = (predicted_visi.weight.data != 0.).reshape(-1)
    flagged_uvwlambda = uvwlambda[flags_bool]

    # parametrization of the image
    with open(os.path.join(TMP_DATA_DIR, 'ws_args.txt'), 'rb') as file:
        d = np.fromfile(file, sep='\n')
    npix = int(d[0])
    cellsize_rad = d[1] * np.pi / 180.
    image_model = create_image_from_visibility(predicted_visi, npixel=npix, cellsize=cellsize_rad)
    image_model = pc.image_add_ra_dec_grid(image_model)

    # generation of forward operator
    directions = SkyCoord(
        ra=image_model.ra_grid.data.ravel() * u.rad,
        dec=image_model.dec_grid.data.ravel() * u.rad,
        frame="icrs",
        equinox="J2000",
    )
    direction_cosines = np.stack(skycoord_to_lmn(directions, predicted_visi.phasecentre), axis=-1)
    forwardOp = pc.generatorVisOp(direction_cosines=direction_cosines,
                                  vlambda=flagged_uvwlambda,
                                  nufft_eps=config['lasso_params']['nufft_eps'],
                                  chunked=config['lasso_params']['chunked'])

    start = time.time()
    fOp_lipschitz = forwardOp.estimate_lipschitz(method='svd', tol=1.)  # ~8000 in 18 min in chunked mode, no memory issue
    dt_lipschitz = time.time() - start
    print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)\n".format(dt_lipschitz))

    measurements = pxc.view_as_real(predicted_visi.vis.data.reshape(-1)[flags_bool])
    sum_vis = measurements.shape[0] // 2
    dirty_array = forwardOp.adjoint(measurements)
    lambda_ = config['lasso_params']['lambda_factor'] * np.abs(dirty_array).max()

    # Prepare the convolved sky image for the computation of the metrics
    psf, _ = invert_visibility(predicted_visi, image_model, context=config['clean_params']['context'], dopsf=True)
    cb = fit_psf(psf)
    restored_sources = restore_cube(sky_im, None, None, clean_beam=cb)

    # PolyCLEAN
    def solve_pclean(save_im):
        fit_params_pc = {
            "stop_crit": pc.stop_crit(config['lasso_params']['tmax'],
                                      config['lasso_params']['min_iter'],
                                      config['lasso_params']['eps']),
            "positivity_constraint": True,
            "diff_lipschitz": fOp_lipschitz ** 2,
            "precision_rule": lambda k: 10 ** (-k / 10),
        }
        pclean = pc.PolyCLEAN(
            data=measurements,
            uvwlambda=flagged_uvwlambda,
            direction_cosines=direction_cosines,
            lambda_=lambda_,
            chunked=config['lasso_params']['chunked'],
            nufft_eps=config['lasso_params']['nufft_eps'],
            **config['pclean_params']
        )
        print("PolyCLEAN: Solving...")
        pclean_time = time.time()
        pclean.fit(**fit_params_pc)
        print("\tSolved in {:.3f} seconds".format(dt_pclean := time.time() - pclean_time))
        sol_pc, hist_pc = pclean.stats()
        pclean_residual = forwardOp.adjoint(measurements - forwardOp(sol_pc["x"]))
        pclean_comp = image_model.copy(deep=True)
        pclean_comp.pixels.data[0, 0] = sol_pc["x"].reshape((npix,) * 2)
        restored_comp = restore_cube(pclean_comp, None, None, clean_beam=cb)
        if save_im:
            restored_comp.image_acc.export_to_fits(os.path.join(args.save_path, 'pclean_comp_restored.fits'))
            pclean_residual_im = image_model.copy(deep=True)
            pclean_residual_im.pixels.data = pclean_residual.reshape(pclean_residual_im.pixels.data.shape) / sum_vis
            restored = restore_cube(pclean_comp, None, pclean_residual_im, clean_beam=cb)
            restored.image_acc.export_to_fits(os.path.join(args.save_path, 'pclean_restored.fits'))
        # compute mse and mad between restored comp and gt convolved
        mse_pc = ut.MSE(restored_comp, restored_sources)
        mad_pc = ut.MAD(restored_comp, restored_sources)

        res_pc = dict(zip(metrics, [dt_pclean, mse_pc, mad_pc, hist_pc["Memorize[objective_func]"][-1], sol_pc['dcv']]))

        return res_pc, hist_pc, dt_pclean

    res_pc, hist_pc, dt_pclean = solve_pclean(save_images)

    # APGD
    def solve_apgd(save_im):
        fit_params_apgd = {
            "x0": np.zeros(forwardOp.shape[1], dtype="float64"),
            "stop_crit": pc.stop_crit(40 * dt_pclean, config['lasso_params']['min_iter'], config['lasso_params']['eps'],
                                        value=hist_pc["Memorize[objective_func]"][-1]),
            # APGD stops when the objective function is a good as PolyCLEAN's one
            "track_objective": True,
            "tau": 1 / (fOp_lipschitz ** 2)
        }
        data_fid_synth = 0.5 * pxop.SquaredL2Norm(dim=forwardOp.shape[0]).argshift(-measurements) * forwardOp
        regul_synth = lambda_ * pyfwl.L1NormPositivityConstraint(shape=(1, forwardOp.shape[1]))
        apgd = pxsol.PGD(data_fid_synth, regul_synth, show_progress=False)
        print("APGD: Solving ...")
        start = time.time()
        apgd.fit(**fit_params_apgd)
        print("\tSolved in {:.3f} seconds".format(dt_apgd := time.time() - start))
        sol_apgd, _ = apgd.stats()
        apgd_residual = forwardOp.adjoint(measurements - forwardOp(sol_apgd["x"]))
        dcv_apgd = abs(apgd_residual).max() / lambda_
        apgd_comp = image_model.copy(deep=True)
        apgd_comp.pixels.data[0, 0] = sol_apgd["x"].reshape((npix,) * 2)
        restored_comp = restore_cube(apgd_comp, None, None, clean_beam=cb)
        if save_im:
            restored_comp.image_acc.export_to_fits(os.path.join(args.save_path, 'apgd_comp_restored.fits'))
            apgd_residual_im = image_model.copy(deep=True)
            apgd_residual_im.pixels.data = apgd_residual.reshape(apgd_residual_im.pixels.data.shape) / sum_vis
            restored = restore_cube(apgd_comp, None, apgd_residual_im, clean_beam=cb)
            restored.image_acc.export_to_fits(os.path.join(args.save_path, 'apgd_restored.fits'))

        # compute mse and mad between restored comp and gt convolved
        mse_apgd = ut.MSE(restored_comp, restored_sources)
        mad_apgd = ut.MAD(restored_comp, restored_sources)

        res_apgd = dict(zip(metrics, [dt_apgd, mse_apgd, mad_apgd, apgd.objective_func()[0], dcv_apgd]))

        return res_apgd

    res_apgd = solve_apgd(save_images)

    # Monoatomic FW
    # def res_monofw(save_im):
    #     monofw = pc.MonoFW(
    #         data=measurements,
    #         uvwlambda=flagged_uvwlambda,
    #         direction_cosines=direction_cosines,
    #         lambda_=lambda_,
    #         chunked=config['lasso_params']['chunked'],
    #         nufft_eps=config['lasso_params']['nufft_eps'],
    #         **config['monofw_params']
    #     )
    #     print("Monoatomic FW: Solving ...")
    #     start = time.time()
    #     monofw.fit(stop_crit=pc.stop_crit(config['monofw_params']['max_time_factor'] * dt_pclean, config['lasso_params']['min_iter'], config['lasso_params']['eps'],
    #                                         value=hist_pc["Memorize[objective_func]"][-1]))
    #     print("\tSolved in {:.3f} seconds".format(dt_monofw := time.time() - start))
    #     sol_monofw, _ = monofw.stats()
    #     monofw_residual = forwardOp.adjoint(measurements - forwardOp(sol_monofw["x"]))
    #     dcv_monofw = abs(monofw_residual).max() / lambda_
    #     monofw_comp = image_model.copy(deep=True)
    #     monofw_comp.pixels.data[0, 0] = sol_monofw["x"].reshape((npix,) * 2)
    #
    #
    #     restored_comp = restore_cube(monofw_comp, None, None, clean_beam=cb)
    #     if save_im:
    #         restored_comp.image_acc.export_to_fits(os.path.join(args.save_path, 'monofw_comp_restored.fits'))
    #         monofw_residual_im = image_model.copy(deep=True)
    #         monofw_residual_im.pixels.data = monofw_residual.reshape(monofw_residual_im.pixels.data.shape) / sum_vis
    #         restored = restore_cube(monofw_comp, None, monofw_residual_im, clean_beam=cb)
    #         restored.image_acc.export_to_fits(os.path.join(args.save_path, 'monofw_restored.fits'))
    #     # compute mse and mad between restored comp and gt convolved
    #     mse_monofw = ut.MSE(restored_comp, restored_sources)
    #     mad_monofw = ut.MAD(restored_comp, restored_sources)
    #
    #     res_monofw = dict(zip(metrics, [dt_monofw, mse_monofw, mad_monofw, monofw.objective_func()[0], dcv_monofw]))
    #
    #     return res_monofw

    # res_monofw = res_monofw(save_images)

    # lasso_res = dict(zip(['pclean', 'apgd', 'monofw'], [res_pc, res_apgd, res_monofw]))
    lasso_res = dict(zip(['pclean', 'apgd'], [res_pc, res_apgd]))

    with open(os.path.join(TMP_DATA_DIR, 'lips_t.pkl'), 'wb') as file:
        pickle.dump({'lips_t': dt_lipschitz}, file)
    with open(os.path.join(TMP_DATA_DIR, 'lasso_res.pkl'), 'wb') as file:
        pickle.dump(lasso_res, file)

    with open(os.path.join(TMP_DATA_DIR, 'clean_beam.pkl'), 'wb') as file:
        pickle.dump(cb, file)
    with open(os.path.join(TMP_DATA_DIR, 'gt_conv.pkl'), 'wb') as file:
        pickle.dump(restored_sources, file)
