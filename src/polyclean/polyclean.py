import numpy as np
import typing as typ
import datetime as dt

import pyfwl
import pyxu.abc.solver as pxas
import pyxu.info.ptype as pxt
import pyxu.abc.operator as pxao
import pyxu.operator as pxop
import pyxu.util.complex as pxc
import pyxu.opt.stop as pxos

#  from operators import stackedWaveletDec

__all__ = [
    "PolyCLEAN",
    "MonoFW",
    "generatorVisOp",
    "diagnostics_polyclean",
    "stop_crit"
]


class _RALassoHvoxImager:
    """
    This class should not be use outside the classes PolyCLEAN and MonoFW, in the context on radio interferometric
    image reconstruction.
    """

    def __init__(
            self,
            data: pxt.NDArray,
            uvwlambda: pxt.NDArray,
            direction_cosines: pxt.NDArray,
            lambda_: float = None,
            lambda_factor: float = 0.1,
            nufft_eps: float = 1e-3,
            chunked=False,
            flagged_bool_mask: pxt.NDArray = None,
            **kwargs,
    ):
        """

        Parameters
        ----------
        data
        uvwlambda
        direction_cosines
        lambda_
        lambda_factor
        nufft_eps
        chunked
        flagged_bool_mask
        kwargs
        """
        if flagged_bool_mask is not None:
            # mask is 2D (times, baselines)
            self._uvw = uvwlambda[flagged_bool_mask].reshape(-1, 3)
        else:
            self._uvw = uvwlambda

        self._direction_cosines = direction_cosines
        self._nufft_eps = nufft_eps
        self._chunked = chunked
        self.forwardOp = generatorVisOp(self._direction_cosines,
                                        self._uvw,
                                        self._nufft_eps,
                                        chunked=chunked,
                                        )
        if lambda_ is None:
            lambda_ = lambda_factor * np.abs(self.forwardOp.adjoint(data)).max()
        self.lambda_ = lambda_

    def rs_forwardOp(self, support_indices: pxt.NDArray) -> pxao.LinOp:
        if support_indices.size == 0:
            return pxop.NullOp(
                shape=(2 * self._uvw.shape[0], 0))  # complex valued visibilities => 2 measurements per baseline
        else:
            return generatorVisOp(self._direction_cosines[support_indices, :],
                                  self._uvw,
                                  self._nufft_eps,
                                  chunked=self._chunked,
                                  )


class PolyCLEAN(_RALassoHvoxImager, pyfwl.PFWLasso):
    def __init__(
            self,
            ms_threshold: float = 0.8,  # multi spikes threshold at init
            init_correction_prec: float = 5e-2,
            final_correction_prec: float = 1e-4,
            remove_positions: bool = True,
            min_correction_steps: int = 5,
            max_correction_steps: int = 100,
            *,
            folder=None,  # : typ.Optional[pxt.PathLike] = None,
            exist_ok: bool = False,
            stop_rate: int = 1,
            writeback_rate: typ.Optional[int] = None,
            verbosity: int = 10,
            show_progress: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)

        super(_RALassoHvoxImager, self).__init__(
            data=kwargs.get('data'),
            forwardOp=self.forwardOp,
            lambda_=self.lambda_,
            ms_threshold=ms_threshold,
            init_correction_prec=init_correction_prec,
            final_correction_prec=final_correction_prec,
            remove_positions=remove_positions,
            min_correction_steps=min_correction_steps,
            max_correction_steps=max_correction_steps,
            folder=folder,
            exist_ok=exist_ok,
            stop_rate=stop_rate,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
        )


class MonoFW(_RALassoHvoxImager, pyfwl.VFWLasso):
    def __init__(
            self,
            step_size: str = "optimal",
            *,
            folder=None,  # : typ.Optional[pxt.PathLike] = None,
            exist_ok: bool = False,
            stop_rate: int = 1,
            writeback_rate: typ.Optional[int] = None,
            verbosity: int = 10,
            show_progress: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        super(_RALassoHvoxImager, self).__init__(
            data=kwargs.get('data'),
            forwardOp=self.forwardOp,
            lambda_=self.lambda_,
            step_size=step_size,
            folder=folder,
            exist_ok=exist_ok,
            stop_rate=stop_rate,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
        )


def generatorVisOp(direction_cosines: pxt.NDArray,
                   vlambda: pxt.NDArray,
                   nufft_eps: float = 1e-3,
                   chunked=False,
                   **kwargs,
                   ) -> pxop.NUFFT:  # pxt.OpT ??
    r"""

    Parameters
    ----------
    chunked
    direction_cosines: NDArray
        This should be flagged beforehand so that only nonzero components are computed.
    vlambda
    nufft_eps

    Returns
    -------

    """
    direct_eval_threshold = kwargs.get("direct_eval_threshold", 10_000)
    max_mem = kwargs.get("max_mem", 10)
    if direction_cosines.shape[0] * vlambda.shape[0] < direct_eval_threshold:
        # direct computation for small size FT
        nufft_eps = 0.

    op = pxop.NUFFT.type3(x=direction_cosines, z=2 * np.pi * vlambda, real=True, isign=-1, eps=nufft_eps,
                          chunked=chunked,
                          parallel=chunked,
                          enable_warnings=False,
                          )
    if chunked:
        x_chunks, z_chunks = op.auto_chunk(max_mem=max_mem)
        op.allocate(x_chunks, z_chunks, direct_eval_threshold=direct_eval_threshold)
    n = direction_cosines[..., :, -1]
    diag = pxop.DiagonalOp(1 / (n + 1.))
    op = op * diag
    op._diff_lipschitz = 0.
    return op


def stop_crit(tmax: float,
              min_iter: int,
              eps: float,
              value: float = None,
              ) -> pxas.StoppingCriterion:
    duration_stop = pxos.MaxDuration(t=dt.timedelta(seconds=tmax))
    min_iter_stop = pxos.MaxIter(n=min_iter)
    if value is None:
        stop_crit = pxos.RelError(
            eps=eps,
            var="objective_func",
            f=None,
            norm=2,
            satisfy_all=True,
        )
    else:
        stop_crit = pxos.AbsError(eps=value, var="objective_func")
    return (stop_crit & min_iter_stop) | duration_stop


# import polyclean.image_utils as ut
#
# from ska_sdp_datamodels.image import Image
# from ska_sdp_datamodels.visibility.vis_model import Visibility
# from ska_sdp_func_python.util import skycoord_to_lmn
#
# from astropy import units as u
# from astropy.coordinates import SkyCoord

# class WtPolyCLEAN(pyfwl.PolyatomicFWforLasso):
#     def __init__(
#             self,
#             visibility_template: Visibility,
#             image_model: Image,
#             data: pxt.NDArray,
#             lambda_: float = None,
#             lambda_factor: float = 0.1,
#             nufft_eps: float = 1e-3,
#             flagged_bool_mask: pxt.NDArray = None,
#             ms_threshold: float = 0.7,  # multi spikes threshold at init
#             init_correction_prec: float = 0.2,
#             final_correction_prec: float = 1e-4,
#             remove_positions: bool = True,
#             min_correction_steps: int = 5,
#             wl_list=["haar"],
#             level: int = 4,
#             include_dirac: bool = False,
#             *,
#             folder=None,  # : typ.Optional[pxt.PathLike] = None,
#             exist_ok: bool = False,
#             stop_rate: int = 1,
#             writeback_rate: typ.Optional[int] = None,
#             verbosity: int = 10,
#             show_progress: bool = True,
#             log_var: pxt.VarName = (
#                     "x",
#                     "dcv",
#             ),
#     ):
#
#         if flagged_bool_mask is None:
#             flagged_bool_mask = np.any(visibility_template.uvw.data != 0., axis=-1)
#         self._mask = flagged_bool_mask  # mask is 2D (times, baselines)
#         self._nufft_eps = nufft_eps
#
#         self._vt = visibility_template
#         # \/\/\/\/\/\/\/\/\/ Direction cosines could be entered as input
#         self._image_model = ut.image_add_ra_dec_grid(image_model)
#         directions = SkyCoord(
#             ra=self._image_model.ra_grid.data.ravel() * u.rad,
#             dec=self._image_model.dec_grid.data.ravel() * u.rad,
#             frame="icrs",
#             equinox="J2000",
#         )
#         self._direction_cosines = np.stack(skycoord_to_lmn(directions, self._vt.phasecentre), axis=-1)
#         self._image_shape = self._image_model.pixels.data.shape[-2:]
#         # /\/\/\/\/\/\/\/\/\
#         uvwlambda = self._vt.visibility_acc.uvw_lambda
#         self._flagged_uvw = uvwlambda[self._mask].reshape(-1, 3)
#         measurementOp = generatorVisOp(self._direction_cosines,
#                                    self._flagged_uvw,
#                                    self._nufft_eps)
#
#         if lambda_ is None:
#             lambda_ = lambda_factor * np.abs(measurementOp.adjoint(data)).max()
#
#         sWaveDec = stackedWaveletDec(self._image_shape, wl_list, level, 'zero', include_dirac=include_dirac)
#         sWaveDec._lipschitz = 1.
#
#         forwardOp = measurementOp * sWaveDec.transpose()
#
#         super().__init__(
#             data=data,
#             forwardOp=forwardOp,
#             lambda_=lambda_,
#             ms_threshold=ms_threshold,
#             init_correction_prec=init_correction_prec,
#             final_correction_prec=final_correction_prec,
#             remove_positions=remove_positions,
#             min_correction_steps=min_correction_steps,
#             folder=folder,
#             exist_ok=exist_ok,
#             stop_rate=stop_rate,
#             writeback_rate=writeback_rate,
#             verbosity=verbosity,
#             show_progress=show_progress,
#             log_var=log_var,
#         )


# x_idx, x_chunks = A.order("x")  # get a good x-ordering
# z_idx, z_chunks = A.order("z")  # get a good z-ordering
# A = pxl.NUFFT.type3(
#         x[x_idx], z[z_idx]  # re-order x/z accordingly
#         ...                 # same as before
#      )
# A.allocate(x_chunks, z_chunks)


# todo lipschitz constant of NUFFT type 3: computation time ?
#  We can probably deduce it from the largest frequency (?)

class InnerProductRef(pxao.LinFunc):
    def __init__(self, reference: pxt.NDArray, support=None):
        if support is None:
            support = slice(None)
        self._support = support
        self._ref = reference[support]
        super().__init__(shape=(1, self._ref.shape[0]))

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        return np.array((self._ref * arr).sum(axis=-1))

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        assert np.r_[arr].shape[-1] == 1
        return arr * self._ref


def psf_kernel(direction_cosines, flagged_uvwlambda, npixel, intensity_rate=.1):
    forwardOp = generatorVisOp(direction_cosines=direction_cosines,
                               vlambda=flagged_uvwlambda,
                               nufft_eps=1e-4, )
    simulated_vis_psf = pxc.view_as_real(np.ones(forwardOp.shape[0] // 2).astype(complex))
    full_psf = forwardOp.adjoint(simulated_vis_psf).reshape((npixel, npixel))
    index_center = np.array(np.unravel_index(np.argmax(full_psf), (npixel,) * 2))
    indices_mask = np.array(
        np.where(np.abs(full_psf) >= intensity_rate * full_psf.max()))  # tuple of array of indices
    radius = np.abs(indices_mask - index_center.reshape((2, 1))).max()
    kernel = full_psf[index_center[0] - radius:index_center[0] + radius + 1,
             index_center[1] - radius:index_center[1] + radius + 1]
    return kernel


def diagnostics_polyclean(pclean, log=False):
    import matplotlib.pyplot as plt

    hist = pclean.stats()[1]

    plt.figure(figsize=(15, 8))
    plt.suptitle("Performance analysis of PolyCLEAN")
    plt.subplot(243)
    plt.plot(pclean._mstate["N_indices"], label="Support size", marker="o", alpha=.5, c='g')
    plt.plot(pclean._mstate["N_candidates"], label="candidates", marker="s", alpha=.5, c='g')
    plt.legend()
    plt.subplot(244)
    plt.plot(pclean._mstate["correction_iterations"], label="iterations", marker="o", alpha=.5, c='g')
    plt.ylim(bottom=0.)
    plt.twinx()
    plt.plot(pclean._mstate["correction_durations"], label="duration", marker="x", alpha=.5, c='g')
    plt.ylim(bottom=0.)
    plt.title("Correction iterations")
    plt.legend()

    plt.subplot(247)
    plt.plot(hist['duration'][1:], pclean._mstate["N_indices"], label="Support size", marker="o", alpha=.5, c='g')
    plt.plot(hist['duration'][1:], pclean._mstate["N_candidates"], label="candidates", marker="s", alpha=.5, c='g')
    plt.xlim(left=0.)
    plt.legend()
    plt.subplot(248)
    plt.plot(hist['duration'][1:], pclean._mstate["correction_iterations"], label="iterations", marker=".", c='#1f77b4',
             alpha=.5)
    plt.ylim(bottom=0.)
    plt.xlim(left=0.)
    plt.twinx()
    plt.plot(hist['duration'][1:], pclean._mstate["correction_durations"], label="duration", marker="x", c='#ff7f0e',
             alpha=.5)
    plt.ylim(bottom=0.)
    plt.xlim(left=0.)
    plt.title("Correction iterations")
    plt.legend()

    plt.subplot(121)
    if log:
        plt.yscale('log')
    plt.scatter(hist['duration'], (hist['Memorize[objective_func]'] - hist['Memorize[objective_func]'][-1]) / (
            hist['Memorize[objective_func]'][0] - hist['Memorize[objective_func]'][-1]), label="PolyCLEAN", s=20,
                marker="+")
    plt.title('Reconstruction: LASSO objective function')
    plt.legend()
    plt.xlim(left=0.)
    plt.show()
