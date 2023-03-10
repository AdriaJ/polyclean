import time
from rascil.data_models import Image, Visibility

import pycsou.abc as pyca
import pycsou.util.ptype as pyct

import polyclean as pc


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
