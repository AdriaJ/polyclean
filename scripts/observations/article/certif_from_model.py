"""
Load the reconstructed image, compute and save the dual certificate.
"""
import pickle
import os
import numpy as np

from examples.scripts.imaging import model

import pyxu.util.complex as pxc
import polyclean.image_utils as ut
import polyclean.polyclean as pc
from astropy.coordinates import SkyCoord
from astropy import units as u
from ska_sdp_func_python.imaging import create_image_from_visibility
from ska_sdp_func_python.util import skycoord_to_lmn

nantennas = 50
lambda_factor = 0.02

npixel = 3072
fov_deg = 6.
nufft_eps = 1e-3

if __name__ == "__main__":
    pkl_dir = "/home/jarret/PycharmProjects/polyclean/scripts/observations/article/reco_pkl"
    pkl_dir += f"/{nantennas}antennas"

    folder = str(lambda_factor)
    with open(os.path.join(pkl_dir, folder, 'model.pkl'), 'rb') as file:
        model = pickle.load(file)

    with open(os.path.join("/home/jarret/PycharmProjects/polyclean/scripts/observations/article",
                           "vis", "vis.pkl"), 'rb') as handle:
        vis = pickle.load(handle)

    ## Get the forward operator
    phasecentre = vis.phasecentre
    fov = fov_deg * np.pi / 180.
    cellsize = fov / npixel
    image_model = create_image_from_visibility(vis, npixel=npixel, cellsize=cellsize, override_cellsize=False)

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

    # Get the value of lambda
    vis_array = pxc.view_as_real(vis.vis.data.reshape(-1)[flags_bool])
    sum_vis = vis_array.shape[0] // 2
    dirty_array = forwardOp.adjoint(vis_array)
    lambda_ = lambda_factor * np.abs(dirty_array).max()

    # Compute the dual certificate
    certif = forwardOp.adjoint(vis_array - forwardOp(model['pixels'].data.flatten())) / lambda_
    dual_certif_im = image_model.copy(deep=True)
    dual_certif_im['pixels'].data = certif.reshape(dual_certif_im['pixels'].data.shape)

    # Save the dual certificate
    with open(os.path.join(pkl_dir, folder, 'certif.pkl'), 'wb') as handle:
        pickle.dump(dual_certif_im, handle)


