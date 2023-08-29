# 1. setup
# 2. solving
# 3. Report solving time
# 4. evaluate convergence of the LASSO-based methods (dcv)
# 5. report metrics for each method (make sure they are similar)
# 6. report sparsity

import numpy as np
import time

from astropy import units as u
from astropy.coordinates import SkyCoord

from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_func_python.image import fit_psf
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_func_python.imaging import (
    predict_visibility,
    invert_visibility,
    create_image_from_visibility,
    advise_wide_field
)

import pycsou.operator.linop as pycl
import pycsou.util.complex as pycuc
import pycsou.operator as pycop
import pycsou.opt.solver as pycsol

import pyfwl

import polyclean as pc
import polyclean.image_utils as ut
import polyclean.reconstructions as reco
from polyclean.clean_utils import mjCLEAN

seed = None

fov_deg = 5
ntimes = 7
times = (2/(ntimes-1) * np.arange(ntimes) - 1) * np.pi/3
frequency = np.array([1e8])
channel_bandwidth = np.array([1e6])
phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000")
npoints = 200
psnrdb = 20

rmax = [1500, 2000]  # 600, 900]  # 1200, ]  # 3000, 5000] 1500 3000

lasso_params = {
    'lambda_factor': 0.01,
    'nufft_eps': 1e-3,
    'chunked': False,
    'eps': 1e-4,
    'tmax': 600,
    'min_iter': 5,
}
clean_params = {
    'n_minor': 10_000,
    'n_major': 5,
    'gain': 0.1,
    'context': 'ng',
    'algorithm': 'hogbom',
}
pclean_params = {
    'ms_threshold': .8,
    'init_correction_prec': 5e-2,
    'final_correction_prec': 1e-4,
    'remove': True,
    'min_correction_steps': 3,
    'max_correction_steps': 1000,
    'show_progress': False,
}
monofw_params = {
    'step_size': 'optimal',
    'show_progress': False,
}
apgd_params = {
    'd': 75,
}


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


if __name__ == "__main__":
    methods = ['CLEAN', 'PolyCLEAN', 'APGD', 'MonoFW']
    durations = {m: [] for m in methods}
    durations_wsclean = []
    components = {m: [] for m in methods}
    residuals = {m: [] for m in methods}
    dcvs = {'CLEAN': None}
    dcvs.update({m: [] for m in methods[1:]})
    objective_func = {'CLEAN': None}
    objective_func.update({m: [] for m in methods[1:]})
    beams = []
    lips_durations = []
    sources = []
    npixels = []

    for r in rmax:
        print(f"\nSimulation for r={r:d}")
        # baselines configuration
        lowr3 = create_named_configuration("LOWBD2", rmax=r)
        # visibility template
        vt = create_visibility(
            lowr3,
            times,
            frequency,
            channel_bandwidth=channel_bandwidth,
            weight=1.0,
            phasecentre=phasecentre,
            polarisation_frame=PolarisationFrame("stokesI"),
        )
        # side size of the image
        npix = get_npixels(vt, fov_deg, phasecentre, lasso_params['nufft_eps'])
        npixels.append(npix)
        print(f"number of pixels: {npix:d}")
        # simulation of a sky image
        sky_im, sc = ut.generate_point_sources(npoints,
                                               fov_deg,
                                               npix,
                                               flux_sigma=.8,
                                               radius_rate=.9,
                                               phasecentre=phasecentre,
                                               frequency=frequency,
                                               channel_bandwidth=channel_bandwidth,
                                               seed=seed)
        sources.append(sky_im)
        # parametrization of the image
        directions = SkyCoord(
            ra=sky_im.ra_grid.data.ravel() * u.rad,
            dec=sky_im.dec_grid.data.ravel() * u.rad,
            frame="icrs",
            equinox="J2000",
        )
        direction_cosines = np.stack(skycoord_to_lmn(directions, phasecentre), axis=-1)

        ## CLEAN advice
        advice = advise_wide_field(
            vt, guard_band_image=3.0, delA=0.1, oversampling_synthesised_beam=4.0
        )
        cellsize_advice = advice["cellsize"]  # radians
        cellsize_nufft = fov_deg * np.pi / 180. / npix
        print(f"Rascil recommends a cellsize of {cellsize_advice:.3e} rad while NUFFT has "
              f"a cellsize of {cellsize_nufft:.3e} rad.")

        ## Reconstructions

        image_model = create_image_from_visibility(vt, cellsize=cellsize_nufft, npixel=npix)

        # CLEAN
        predicted_visi = predict_visibility(vt, sky_im, context=clean_params['context'])
        real_visi = pycuc.view_as_real(predicted_visi.vis.data[:, :, 0, 0] * predicted_visi.weight.data[:, :, 0, 0])
        noise_scale = np.abs(real_visi).max() * 10 ** (-psnrdb / 20) / np.sqrt(2)
        noise = np.random.normal(0, noise_scale, real_visi.shape)
        predicted_visi.vis.data += pycuc.view_as_complex(noise)[:, :, None, None]
        clean_model = create_image_from_visibility(predicted_visi, cellsize=cellsize_nufft, npixel=2 * npix)
        dirty, _ = invert_visibility(predicted_visi, clean_model, context=clean_params['context'])
        psf, _ = invert_visibility(predicted_visi, image_model, context=clean_params['context'], dopsf=True)
        print("CLEAN: Solving...")
        start = time.time()
        tmp_clean_comp, tmp_clean_residual = mjCLEAN(
            dirty,
            psf,
            **clean_params,
            vt=vt,
            threshold=0.001,
            fractional_threshold=0.001,
            window_shape="quarter",
        )

        clean_comp = image_model.copy(deep=True)
        clean_comp['pixels'].data[0, 0, ...] = \
            tmp_clean_comp['pixels'].data[0, 0, npix // 2: npix + npix // 2, npix // 2: npix + npix // 2]
        clean_residual = image_model.copy(deep=True)
        clean_residual['pixels'].data[0, 0, ...] = \
            tmp_clean_residual['pixels'].data[0, 0, npix // 2: npix + npix // 2, npix // 2: npix + npix // 2]
        print("\tSolved in {:.3f} seconds".format(dt_clean := time.time() - start))
        durations['CLEAN'].append(dt_clean)
        components['CLEAN'].append(clean_comp)
        residuals['CLEAN'].append(clean_residual)
        beams.append(fit_psf(psf))

        # WS-CLEAN
        print("WS-CLEAN solving:")
        from rascil.processing_components.visibility.base import export_visibility_to_ms
        import os
        ws_dir = "/home/jarret/PycharmProjects/polyclean/scripts/time_comparison/wsclean-dir" + f"/rmax{int(r):d}"
        filename = f"rmax{int(r):d}.ms"
        if not os.path.exists(ws_dir):
            os.makedirs(ws_dir)
        export_visibility_to_ms(ws_dir + "/" + filename, [predicted_visi], )
        start = time.time()
        os.system(
            f"wsclean -auto-threshold 1 -size {npix:d} {npix:d} -scale {fov_deg/npix:.6f} -mgain 0.7 -niter 10000 "
            f"-weight natural -name ws-rmax{int(r):d} -quiet -no-dirty wsclean-dir/rmax{int(r):d}/rmax{int(r):d}.ms")
        print("\tRun in {:.3f}s".format(dt_wsclean := time.time() - start))
        os.system(f"mv ws-rmax{int(r):d}* wsclean-dir/rmax{int(r):d}/")
        durations_wsclean.append(dt_wsclean)

        ## LASSO common parameters
        uvwlambda = vt.visibility_acc.uvw_lambda.reshape(-1, 3)
        flags_bool = (vt.weight.data != 0.).reshape(-1)
        flagged_uvwlambda = uvwlambda[flags_bool]

        ### Simulation of the measurements
        forwardOp = pc.generatorVisOp(direction_cosines=direction_cosines,
                                      vlambda=flagged_uvwlambda,
                                      nufft_eps=lasso_params['nufft_eps'],
                                      chunked=lasso_params['chunked'])  # todo: when do I need to chunk ? At worst it only gives a bad lipschitz cmputation time
        start = time.time()
        fOp_lipschitz = forwardOp.lipschitz(tol=1., tight=True)
        lips_durations.append(lips_time := time.time() - start)
        print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)".format(lips_time))

        # single_time_vt = vt.sel({"time": vt.time[3]})
        # uvwlambda = single_time_vt.visibility_acc.uvw_lambda.reshape(-1, 3)
        # flags_bool = (single_time_vt.weight.data != 0.).reshape(-1)
        # flagged_uvwlambda = uvwlambda[flags_bool]
        # forwardOp = pc.generatorVisOp(direction_cosines=direction_cosines,
        #                               vlambda=flagged_uvwlambda,
        #                               nufft_eps=lasso_params['nufft_eps'],
        #                               chunked=True)
        # start = time.time()
        # fOp_lipschitz = forwardOp.lipschitz(tol=1., tight=True)
        # lips_durations.append(lips_time := time.time() - start)
        # print("Computation of the Lipschitz constant of the forward operator in: {:.3f} (s)".format(lips_time))
        # print(fOp_lipschitz)

        noiseless_measurements = forwardOp(sky_im.pixels.data.reshape(-1))
        # noise_scale = np.abs(noiseless_measurements).max() * 10 ** (-psnrdb / 20) / np.sqrt(2)
        # noise = np.random.normal(0, noise_scale, noiseless_measurements.shape)
        noise_pclean = pycuc.view_as_real(pycuc.view_as_complex(noise).flatten()[flags_bool])
        measurements = noiseless_measurements + noise_pclean

        dirty_image = forwardOp.adjoint(measurements)
        lambda_ = lasso_params['lambda_factor'] * np.abs(dirty_image).max()

        # # make sure that both the measuremetns are equal and that the dirty image are also equal (up to the multiplication factor)
        # cropped_dirty = image_model.copy(deep=True)
        # cropped_dirty['pixels'].data[0, 0, ...] = \
        #     dirty['pixels'].data[0, 0, npix // 2: npix + npix // 2, npix // 2: npix + npix // 2]
        # np.linalg.norm(cropped_dirty.pixels.data.flatten() - dirty_image / (measurements.shape[0] // 2)) / np.linalg.norm(cropped_dirty.pixels.data.flatten())
        # np.linalg.norm(pycuc.view_as_complex(real_visi).flatten()[flags_bool] - pycuc.view_as_complex(noiseless_measurements)) / np.linalg.norm(pycuc.view_as_complex(real_visi).flatten()[flags_bool])
        # np.linalg.norm(predicted_visi.vis.data.flatten()[flags_bool] - pycuc.view_as_complex(measurements)) / np.linalg.norm(measurements)


        # PolyCLEAN
        fit_params_pc = {
            "stop_crit": reco.stop_crit(lasso_params['tmax'], lasso_params['min_iter'], lasso_params['eps']),
            "positivity_constraint": True,
            "diff_lipschitz": fOp_lipschitz ** 2,
            "precision_rule": lambda k: 10 ** (-k / 10),
        }
        pclean = pc.PolyCLEAN(
            data=measurements,
            uvwlambda=flagged_uvwlambda,
            direction_cosines=direction_cosines,
            lambda_=lambda_,
            chunked=lasso_params['chunked'],
            nufft_eps=lasso_params['nufft_eps'],
            **pclean_params
        )
        print("PolyCLEAN: Solving...")
        pclean_time = time.time()
        pclean.fit(**fit_params_pc)
        print("\tSolved in {:.3f} seconds".format(dt_pclean := time.time() - pclean_time))
        sol_pc, hist_pc = pclean.stats()
        pclean_residual = forwardOp.adjoint(measurements - forwardOp(sol_pc["x"]))
        pclean_comp = sky_im.copy(deep=True)
        pclean_comp.pixels.data[0, 0] = sol_pc["x"].reshape((npix,) * 2)
        pclean_residual_im = sky_im.copy(deep=True)
        pclean_residual_im.pixels.data = pclean_residual.reshape(pclean_residual_im.pixels.data.shape) / (
                    measurements.shape[0] // 2)
        durations['PolyCLEAN'].append(dt_pclean)
        components['PolyCLEAN'].append(pclean_comp)
        residuals['PolyCLEAN'].append(pclean_residual_im)
        dcvs['PolyCLEAN'].append(sol_pc['dcv'])
        objective_func['PolyCLEAN'].append(pclean.objective_func())

        # APGD
        fit_params_apgd = {
            "x0": np.zeros(forwardOp.shape[1], dtype="float64"),
            "stop_crit": reco.stop_crit(40 * dt_pclean, lasso_params['min_iter'], lasso_params['eps'],
                                        value=hist_pc["Memorize[objective_func]"][-1]),  # APGD stops when the objective function is a good as PolyCLEAN's one
            "track_objective": True,
            "tau": 1 / (fOp_lipschitz ** 2)
        }
        fit_params_apgd.update(apgd_params)
        data_fid_synth = 0.5 * pycop.SquaredL2Norm(dim=forwardOp.shape[0]).argshift(-measurements) * forwardOp
        regul_synth = lambda_ * pyfwl.L1NormPositivityConstraint(shape=(1, None))
        apgd = pycsol.PGD(data_fid_synth, regul_synth, show_progress=False)
        print("APGD: Solving ...")
        start = time.time()
        apgd.fit(**fit_params_apgd)
        print("\tSolved in {:.3f} seconds".format(dt_apgd := time.time() - start))
        sol_apgd, _ = apgd.stats()
        apgd_residual = forwardOp.adjoint(measurements - forwardOp(sol_apgd["x"]))
        dcv_apgd = abs(apgd_residual).max() / lambda_
        apgd_comp = sky_im.copy(deep=True)
        apgd_comp.pixels.data[0, 0] = sol_apgd["x"].reshape((npix,) * 2)
        apgd_residual_im = sky_im.copy(deep=True)
        apgd_residual_im.pixels.data = apgd_residual.reshape(apgd_residual_im.pixels.data.shape) / (
                    measurements.shape[0] // 2)
        durations['APGD'].append(dt_apgd)
        components['APGD'].append(apgd_comp)
        residuals['APGD'].append(apgd_residual_im)
        dcvs['APGD'].append(dcv_apgd)
        objective_func['APGD'].append(apgd.objective_func())

        # Monoatomic FW
        monofw = pc.MonoFW(
            data=measurements,
            uvwlambda=flagged_uvwlambda,
            direction_cosines=direction_cosines,
            lambda_=lambda_,
            chunked=lasso_params['chunked'],
            nufft_eps=lasso_params['nufft_eps'],
            **monofw_params
        )
        print("Monoatomic FW: Solving ...")
        start = time.time()
        monofw.fit(stop_crit=reco.stop_crit(40 * dt_pclean, lasso_params['min_iter'], lasso_params['eps'],
                                        value=hist_pc["Memorize[objective_func]"][-1]))
        print("\tSolved in {:.3f} seconds".format(dt_monofw := time.time() - start))
        sol_monofw, _ = monofw.stats()
        monofw_residual = forwardOp.adjoint(measurements - forwardOp(sol_monofw["x"]))
        dcv_monofw = abs(monofw_residual).max() / lambda_
        monofw_comp = sky_im.copy(deep=True)
        monofw_comp.pixels.data[0, 0] = sol_monofw["x"].reshape((npix,) * 2)
        monofw_residual_im = sky_im.copy(deep=True)
        monofw_residual_im.pixels.data = monofw_residual.reshape(monofw_residual_im.pixels.data.shape) / (
                measurements.shape[0] // 2)
        durations['MonoFW'].append(dt_monofw)
        components['MonoFW'].append(monofw_comp)
        residuals['MonoFW'].append(monofw_residual_im)
        dcvs['MonoFW'].append(dcv_monofw)
        objective_func['MonoFW'].append(monofw.objective_func())

    folder = "/home/jarret/PycharmProjects/polyclean/scripts/time_comparison"
    filename = "res" + "_".join([str(r) for r in rmax]) + ".pkl"
    res = {
        "rmax": rmax,
        "npixels": npixels,
        "methods": methods,
        "durations": durations,
        "durations_wsclean": durations_wsclean,
        "components": components,
        "residuals": residuals,
        "dcvs": dcvs,
        "beams": beams,
        "lips_durations": lips_durations,
        "objective_func": objective_func,
        "sources": sources,
    }
    import pickle
    with open(folder + "/" + filename, 'wb') as file:
        pickle.dump(res, file)
