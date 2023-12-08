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

class GaussPolyCLEAN(polyclean.PolyCLEAN):
    def __init__(self,
                 scales: list,
                 kernel_bias: list = None,
                 n_supp: int = 2,
                 norm_kernels: int = 2,
                 **kwargs):
        super().__init__(**kwargs)
        npixel = int(kwargs.get('direction_cosines').shape[0]**.5)
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