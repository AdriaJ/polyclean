import numpy as np
from ska_sdp_datamodels.image.image_model import Image
from ska_sdp_func_python.imaging import predict_visibility, invert_visibility
from ska_sdp_func_python.image import deconvolve_cube


def mjCLEAN(dirty, psf, n_major, n_minor, vt, context="ng", **kwargs):
    """

    Parameters
    ----------
    dirty
    psf
    n_major
    n_minor
    vt:
        Visibility template
    context:
        Context for the gridder
    kwargs:
        regular arguments for CLEAN deconvolution

    Returns
    -------

    """
    assert n_major > 0
    dirty_residual = dirty
    components = Image.constructor(
        data=np.zeros_like(dirty.pixels.data),
        polarisation_frame=dirty.image_acc.polarisation_frame,
        wcs=dirty.image_acc.wcs,
    )
    for i in range(n_major - 1):
        clean_comp, clean_residual = deconvolve_cube(
            dirty_residual,
            psf,
            niter=n_minor // n_major,
            **kwargs
        )
        components.pixels.data += clean_comp.pixels.data
        if i == n_major - 2:  # don't run major cycle for last iteration
            dirty_residual = clean_residual
        else:  # re-evaluate the residual without convolution
            contrib_components, _ = invert_visibility(predict_visibility(vt, components, context=context),
                                                   dirty,  # used as the image model
                                                   context=context)
            dirty_residual = dirty.copy(deep=True)
            dirty_residual.pixels.data -= contrib_components.pixels.data
    return components, dirty_residual
