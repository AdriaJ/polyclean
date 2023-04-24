import time
import numpy as np

from ska_sdp_datamodels.image import Image
from ska_sdp_datamodels.visibility.vis_model import Visibility

import pycsou.abc as pyca
import pycsou.util.ptype as pyct


import polyclean as pc


def stop_crit(
        tmax: float,
        min_iter: int,
        eps: float,
        value: float = None,
) -> pyca.StoppingCriterion:
    import datetime as dt
    import pycsou.opt.stop as pycos

    duration_stop = pycos.MaxDuration(t=dt.timedelta(seconds=tmax))
    min_iter_stop = pycos.MaxIter(n=min_iter)
    if value is None:
        stop_crit = pycos.RelError(
            eps=eps,
            var="objective_func",
            f=None,
            norm=2,
            satisfy_all=True,
        )
    else:
        stop_crit = pycos.AbsError(eps=value, var="objective_func")
    return (stop_crit & min_iter_stop) | duration_stop


def get_clean_default_params():
    DEFAULT_CLEAN_PARAMS = {
        "niter": 10000,
        "threshold": 0.001,
        "fractional_threshold": 0.001,
        "window_shape": "quarter",
        "gain": 0.7,
        "algorithm": 'hogbom',
    }
    return DEFAULT_CLEAN_PARAMS


def reco_clean(
        sky_image: Image,
        vt: Visibility,
        clean_paramters: dict,
        context: str = "ng",
):
    from ska_sdp_func_python.imaging import (
        predict_visibility,
        invert_visibility,
        create_image_from_visibility,
    )
    from ska_sdp_func_python.image import deconvolve_cube

    cellsize = abs(sky_image.coords["x"].data[1] - sky_image.coords["x"].data[0])
    npixel = sky_image.dims["x"]

    predicted_visi = predict_visibility(vt, sky_image, context=context)
    image_model = create_image_from_visibility(predicted_visi, cellsize=cellsize, npixel=npixel)

    clean_model = create_image_from_visibility(predicted_visi, cellsize=cellsize, npixel=2 * npixel)
    dirty, sumwt_dirty = invert_visibility(predicted_visi, clean_model, context=context)
    psf, sumwt = invert_visibility(predicted_visi, image_model, context=context, dopsf=True)
    print("CLEAN: Solving...")
    start = time.time()
    tmp_clean_comp, tmp_clean_residual = deconvolve_cube(
        dirty,
        psf,
        **clean_paramters,
    )
    clean_comp = image_model.copy(deep=True)
    clean_comp['pixels'].data[0, 0, ...] = \
        tmp_clean_comp['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]
    clean_residual = image_model.copy(deep=True)
    clean_residual['pixels'].data[0, 0, ...] = \
        tmp_clean_residual['pixels'].data[0, 0, npixel // 2: npixel + npixel // 2, npixel // 2: npixel + npixel // 2]
    dt = time.time() - start
    print("\tSolved in {:.3f} seconds".format(dt))

    return clean_comp, clean_residual, dt


def reco_pclean(
        uvwlambda: pyct.NDArray,
        direction_cosines: pyct.NDArray,
        data: pyct.NDArray,
        lambda_: float,
        pclean_parameters: dict,
        fit_parameters: dict,
        diagnostics: bool = True,
        log_diagnostics: bool = False,
):
    pclean = pc.PolyCLEAN(
        uvwlambda,
        direction_cosines,
        data,
        lambda_=lambda_,
        **pclean_parameters,
    )
    print("PolyCLEAN: Solving...")
    pclean_time = time.time()
    pclean.fit(**fit_parameters)
    print("\tSolved in {:.3f} seconds".format(time.time() - pclean_time))
    if diagnostics:
        pclean.diagnostics(log=log_diagnostics)
    return pclean.stats()


def reco_pclean_plus(
        uvwlambda: pyct.NDArray,
        direction_cosines: pyct.NDArray,
        data: pyct.NDArray,
        lambda_: float,
        pcleanp_parameters: dict,
        fit_parameters: dict,
        diagnostics: bool = True,
        log_diagnostics: bool = False,
):
    import pycsou.operator as pycop
    import pycsou.opt.solver as pycsol

    rate_tk = pcleanp_parameters.get("rate_lsr", 0.)
    assert rate_tk >= 0.

    pclean = pc.PolyCLEAN(
        uvwlambda,
        direction_cosines,
        data,
        lambda_=lambda_,
        **pcleanp_parameters,
    )
    print("PolyCLEAN: Solving...")
    pclean_time = time.time()
    pclean.fit(**fit_parameters)
    print("\tSolved in {:.3f} seconds".format(time.time() - pclean_time))
    if diagnostics:
        pclean.diagnostics(log=log_diagnostics)
    solution, hist = pclean.stats()

    s = stop_crit(tmax=hist["duration"][-1] * pcleanp_parameters.get("overtime_lsr", .2),
                  min_iter=pcleanp_parameters.get("min_iter_lsr", 5),
                  eps=pcleanp_parameters.get("eps_lsr", 1e-4))
    sol = solution["x"]
    support = np.nonzero(sol)[0]
    rs_forwardOp = pclean.rs_forwardOp(support)
    rs_data_fid = .5 * pycop.SquaredL2Norm(dim=rs_forwardOp.shape[0]).argshift(-data) * rs_forwardOp
    if rate_tk > 0.:
        rs_data_fid = rs_data_fid + 0.5 * rate_tk * fit_parameters["diff_lipschitz"] * pycop.SquaredL2Norm(dim=rs_forwardOp.shape[1])
    rs_regul = pycop.PositiveOrthant(dim=rs_forwardOp.shape[1])
    lsr_apgd = pycsol.PGD(rs_data_fid, rs_regul, show_progress=False)
    print("Least squares reweighting:")
    lsr_apgd.fit(x0=sol[support],
                 stop_crit=s,
                 track_objective=True,
                 tau=1 / (fit_parameters["diff_lipschitz"] * (1 + rate_tk)))
    data_lsr, hist_lsr = lsr_apgd.stats()
    print("\tSolved in {:.3f} seconds ({:d} iterations)".format(hist_lsr['duration'][-1],
                                                              int(hist_lsr['N_iter'][-1])))
    res = np.zeros_like(sol)
    res[support] = data_lsr["x"]
    solution["x_old"] = solution.pop("x")
    solution["x"] = res

    return solution, hist, hist_lsr


def reco_apgd(
        uvwlambda: pyct.NDArray,
        direction_cosines: pyct.NDArray,
        data: pyct.NDArray,
        lambda_: float,
        apgd_parameters: dict,
        fit_parameters: dict,
):
    import pycsou.operator as pycop
    import pycsou.opt.solver as pycsol
    import pyfwl

    forwardOp = pc.generatorVisOp(direction_cosines,
                                  uvwlambda,
                                  apgd_parameters["nufft_eps"])
    data_fid_synth = 0.5 * pycop.SquaredL2Norm(dim=forwardOp.shape[0]).argshift(-data) * forwardOp
    regul_synth = lambda_ * pyfwl.L1NormPositivityConstraint(shape=(1, None))
    apgd = pycsol.PGD(data_fid_synth, regul_synth, show_progress=False)
    print("APGD: Solving ...")
    start = time.time()
    apgd.fit(**fit_parameters)
        # x0=np.zeros(forwardOp.shape[1], dtype="float64"),
        # # stop_crit=(min_iter_stop & apgd.default_stop_crit()) | duration_stop,
        # stop_crit=(min_iter_stop & pycos.AbsError(eps=hist["Memorize[objective_func]"][-1],
        #                                           var="objective_func")) | duration_stop,
        # track_objective=True,
        # tau=1 / (fOp_lipschitz ** 2),
    print("\tSolved in {:.3f} seconds".format(time.time() - start))

    sol, hist = apgd.stats()
    dcv = abs(forwardOp.adjoint(data - forwardOp(sol["x"]))).max()/lambda_
    sol["dcv"] = dcv

    return sol, hist
