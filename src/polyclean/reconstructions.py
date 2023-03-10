import time
import numpy as np

from rascil.data_models import Image, Visibility

import pycsou.abc as pyca
import pycsou.util.ptype as pyct

import polyclean as pc


def stop_crit(
        tmax: float,
        min_iter: int,
        eps: float,
) -> pyca.StoppingCriterion:
    import datetime as dt
    import pycsou.opt.stop as pycos

    duration_stop = pycos.MaxDuration(t=dt.timedelta(seconds=tmax))
    min_iter_stop = pycos.MaxIter(n=min_iter)
    stop_crit = pycos.RelError(
        eps=eps,
        var="objective_func",
        f=None,
        norm=2,
        satisfy_all=True,
    )
    return (stop_crit & min_iter_stop) | duration_stop


def reco_clean(
        dirty: Image,
        psf: Image,
        clean_paramters: dict
):
    pass


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
        pc.diagnostics_polyclean(pclean, log=log_diagnostics)
    return pclean.stats()


def reco_pclean_plus(
        uvwlambda: pyct.NDArray,
        direction_cosines: pyct.NDArray,
        data: pyct.NDArray,
        lambda_: float,
        pclean_parameters: dict,
        fit_parameters: dict,
        diagnostics: bool = True,
        log_diagnostics: bool = False,
):
    import pycsou.operator as pycop
    import pycsou.opt.solver as pycsol

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
        pc.diagnostics_polyclean(pclean, log=log_diagnostics)
    solution, hist = pclean.stats()

    s = stop_crit(tmax=hist["duration"][-1] * pclean_parameters.get("overtime_lsr", .2),
                  min_iter=pclean_parameters.get("min_iter_lsr", 5),
                  eps=pclean_parameters.get("eps_lsr", 1e-4))
    sol = solution["x"]
    support = np.nonzero(sol)[0]
    rs_forwardOp = pclean.rs_forwardOp(support)
    rs_data_fid = .5 * pycop.SquaredL2Norm(dim=rs_forwardOp.shape[0]).argshift(-data) * rs_forwardOp
    rs_regul = pycop.PositiveOrthant(dim=rs_forwardOp.shape[1])
    lsr_apgd = pycsol.PGD(rs_data_fid, rs_regul, show_progress=False)
    print("Least squares reweighting:")
    lsr_apgd.fit(x0=sol[support],
                 stop_crit=s,
                 track_objective=True,
                 tau=1 / fit_parameters["diff_lipschitz"])
    print("\tSolved in {:.3f} seconds".format(lsr_apgd.stats()[1]['duration'][-1]))
    solution["lsr_x"] = lsr_apgd.stats()[0]["x"]

    return solution, hist

