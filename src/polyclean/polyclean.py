import numpy as np
import typing as typ
import time

import pyfwl

import pycsou.util.ptype as pyct
import pycsou.abc.operator as pyco
import pycsou.operator as pycop
import pycsou.util.complex as pycuc

from rascil.data_models import Visibility, Image
from rascil.processing_components import (
    image_add_ra_dec_grid,
)
from rascil.processing_components.util import skycoord_to_lmn
from astropy import units as u
from astropy.coordinates import SkyCoord

#  from operators import stackedWaveletDec

__all__ = [
    "PolyCLEAN",
    "generatorVisOp",
    "diagnostics_polyclean"
]


class PolyCLEAN(pyfwl.PolyatomicFWforLasso):
    def __init__(
            self,
            # visibility_template: Visibility,
            # image_model: Image,
            uvwlambda: pyct.NDArray,
            direction_cosines: pyct.NDArray,
            data: pyct.NDArray,
            lambda_: float = None,
            lambda_factor: float = 0.1,
            nufft_eps: float = 1e-3,
            flagged_bool_mask: pyct.NDArray = None,
            ms_threshold: float = 0.7,  # multi spikes threshold at init
            init_correction_prec: float = 0.2,
            final_correction_prec: float = 1e-4,
            remove_positions: bool = True,
            min_correction_steps: int = 5,
            minor_cycles: int = 0,
            kernel: pyct.NDArray = np.ones((1, 1)),
            kernel_center: pyct.NDArray = None,
            *,
            folder=None,  # : typ.Optional[pyct.PathLike] = None,
            exist_ok: bool = False,
            stop_rate: int = 1,
            writeback_rate: typ.Optional[int] = None,
            verbosity: int = 10,
            show_progress: bool = True,
            log_var: pyct.VarName = (
                    "x",
                    "dcv",
            ),
    ):
        if flagged_bool_mask is not None:
            # mask is 2D (times, baselines)
            self._uvw = uvwlambda[flagged_bool_mask].reshape(-1, 3)
        else:
            self._uvw = uvwlambda

        self._direction_cosines = direction_cosines
        self._nufft_eps = nufft_eps
        forwardOp = generatorVisOp(self._direction_cosines,
                                   self._uvw,
                                   self._nufft_eps)

        self._minor_cycles = minor_cycles + 1
        if self._minor_cycles > 1:
            self._dirty_image = forwardOp.adjoint(data)
            self._kernel = kernel
            if kernel_center is None:
                kernel_center = np.array(self._kernel.shape) // 2
            self._kernel_center = kernel_center
            # self.convOp = pycop.Stencil(stencil_coefs=self._kernel,
            #                             center=self._kernel_center,
            #                             arg_shape=self._image_shape,
            #                             boundary=0.).T
            stop_rate = self._minor_cycles
            verbosity = verbosity * stop_rate
            start = time.time()
            # self.convOp(np.zeros(self.convOp.shape[1]))
            print("Compile time for stencils: {:.3f}".format(time.time() - start))

        if lambda_ is None:
            lambda_ = lambda_factor * np.abs(forwardOp.adjoint(data)).max()

        super().__init__(
            data=data,
            forwardOp=forwardOp,
            lambda_=lambda_,
            ms_threshold=ms_threshold,
            init_correction_prec=init_correction_prec,
            final_correction_prec=final_correction_prec,
            remove_positions=remove_positions,
            min_correction_steps=min_correction_steps,
            folder=folder,
            exist_ok=exist_ok,
            stop_rate=stop_rate,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
        )

    # def df_grad(self) -> pyct.NDArray:
    #     if self._astate["idx"] % self._minor_cycles == 0:
    #         return super().df_grad()
    #     else:
    #         return pycop.QuadraticFunc(self.convOp,
    #                                    -1. * InnerProductRef(reference=self._dirty_image),
    #                                    init_lipschitz=False
    #                                    ).grad(self._mstate["x"])

    def rs_forwardOp(self, support_indices: pyct.NDArray) -> pyco.LinOp:
        return generatorVisOp(self._direction_cosines[support_indices],
                              self._uvw,
                              self._nufft_eps)

    def rs_data_fid(self, support_indices: pyct.NDArray) -> pyco.DiffFunc:
        if self._astate["idx"] % self._minor_cycles == 0:
            rs_forwardOp = generatorVisOp(self._direction_cosines[support_indices],
                                          self._uvw,
                                          self._nufft_eps)
            return 0.5 * pycop.SquaredL2Norm(dim=self.forwardOp.shape[0]).argshift(-self.data) * rs_forwardOp
        else:
            ss = pycop.SubSample(self.forwardOp.shape[1], support_indices)
            return pycop.QuadraticFunc(ss * self.convOp * ss.T,
                                       -1. * InnerProductRef(reference=ss(self._dirty_image)),
                                       init_lipschitz=False)


def generatorVisOp(direction_cosines: pyct.NDArray,
                   vlambda: pyct.NDArray,
                   nufft_eps: float = 1e-3
                   ) -> pycop.NUFFT:  # pyct.OpT ??
    r"""

    Parameters
    ----------
    direction_cosines: NDArray
        This should be flagged beforehand so that only nonzero components are computed.
    vlambda
    nufft_eps

    Returns
    -------

    """
    op = pycop.NUFFT.type3(x=direction_cosines, z=2 * np.pi * vlambda, real=True, isign=-1, eps=nufft_eps,
                           # chunked=True,
                           # parallel=True,
                           )
    # x_chunks, z_chunks = op.auto_chunk()  # auto-determine a good x/z chunking strategy
    # op.allocate(x_chunks, z_chunks, enable_warnings=False)
    n = direction_cosines[..., :, -1]
    diag = pycop.DiagonalOp(1 / (n + 1.))
    op = op * diag
    op._diff_lipschitz = 0.
    return op

# class WtPolyCLEAN(pyfwl.PolyatomicFWforLasso):
#     def __init__(
#             self,
#             visibility_template: Visibility,
#             image_model: Image,
#             data: pyct.NDArray,
#             lambda_: float = None,
#             lambda_factor: float = 0.1,
#             nufft_eps: float = 1e-3,
#             flagged_bool_mask: pyct.NDArray = None,
#             ms_threshold: float = 0.7,  # multi spikes threshold at init
#             init_correction_prec: float = 0.2,
#             final_correction_prec: float = 1e-4,
#             remove_positions: bool = True,
#             min_correction_steps: int = 5,
#             wl_list=["haar"],
#             level: int = 4,
#             include_dirac: bool = False,
#             *,
#             folder=None,  # : typ.Optional[pyct.PathLike] = None,
#             exist_ok: bool = False,
#             stop_rate: int = 1,
#             writeback_rate: typ.Optional[int] = None,
#             verbosity: int = 10,
#             show_progress: bool = True,
#             log_var: pyct.VarName = (
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
#         self._image_model = image_add_ra_dec_grid(image_model)
#         directions = SkyCoord(
#             ra=self._image_model.ra_grid.data.ravel() * u.rad,
#             dec=self._image_model.dec_grid.data.ravel() * u.rad,
#             frame="icrs",
#             equinox="J2000",
#         )
#         self._direction_cosines = np.stack(skycoord_to_lmn(directions, self._vt.phasecentre), axis=-1)
#         self._image_shape = self._image_model.pixels.data.shape[-2:]
#         # /\/\/\/\/\/\/\/\/\
#         uvwlambda = self._vt.uvw_lambda.data
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
# A = pycl.NUFFT.type3(
#         x[x_idx], z[z_idx]  # re-order x/z accordingly
#         ...                 # same as before
#      )
# A.allocate(x_chunks, z_chunks)


# todo lipschitz constant of NUFFT type 3: computation time ?
#  We can probably deduce it from the largest frequency (?)

class InnerProductRef(pyco.LinFunc):
    def __init__(self, reference: pyct.NDArray, support=None):
        if support is None:
            support = slice(None)
        self._support = support
        self._ref = reference[support]
        super().__init__(shape=(1, self._ref.shape[0]))

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return np.array((self._ref * arr).sum(axis=-1))

    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        assert np.r_[arr].shape[-1] == 1
        return arr * self._ref


def psf_kernel(direction_cosines, flagged_uvwlambda, npixel, intensity_rate=.1):
    forwardOp = generatorVisOp(direction_cosines=direction_cosines,
                               vlambda=flagged_uvwlambda,
                               nufft_eps=1e-4)
    simulated_vis_psf = pycuc.view_as_real(np.ones(forwardOp.shape[0] // 2).astype(complex))
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
