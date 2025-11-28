"""
Load the image produced by ws clean
Convolve them with the clean beam obtained with Rascil
compute the metrics
export the results with pickle in a dictionary
"""
import os
import pickle
import numpy as np
import argparse

import polyclean.image_utils as ut

from ska_sdp_func_python.image import restore_cube
from rascil.processing_components.image.operations import import_image_from_fits

TMP_DATA_DIR = 'tmpdir'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the metrics for the image produced by wsclean, optionally save the convolved image.")
    parser.add_argument('--save_path', type=str, default=None, required=False,
                        help="Location where to save the image convolved with RASCIL clean beam."
                             " If None, the images are not saved.")
    args = parser.parse_args()
    save_images = args.save_path is not None

    wsclean_model = import_image_from_fits(os.path.join(TMP_DATA_DIR, 'ws-model.fits'))
    # use pickle to load the clean beam
    with open(os.path.join(TMP_DATA_DIR, 'clean_beam.pkl'), 'rb') as file:
        clean_beam = pickle.load(file)
    # load the convolved sources with pickle
    with open(os.path.join(TMP_DATA_DIR, 'gt_conv.pkl'), 'rb') as file:
        conv_sources = pickle.load(file)
    with open(os.path.join(TMP_DATA_DIR, 'wsclean_time.txt'), 'rb') as file:
        time_wsclean = np.fromfile(file, sep='\n')[0]
    # convolve the model with the clean beam
    restored_comp = restore_cube(wsclean_model, None, None, clean_beam)

    mse_wsclean = ut.MSE(restored_comp, conv_sources)
    mad_wsclean = ut.MAD(restored_comp, conv_sources)

    res_wscean = dict(zip(['time', 'mse', 'mad'], [time_wsclean, mse_wsclean, mad_wsclean]))
    with open(os.path.join(TMP_DATA_DIR, 'clean_res.pkl'), 'wb') as file:
        pickle.dump({'wsclean': res_wscean}, file)

    if save_images:
        wsclean_residual = import_image_from_fits(os.path.join(TMP_DATA_DIR, f"ws-residual.fits"))
        restored_wsclean = restore_cube(wsclean_model, None, wsclean_residual, clean_beam)
        restored_comp.image_acc.export_to_fits(os.path.join(args.save_path, 'ws_comp_restored.fits'))
        restored_wsclean.image_acc.export_to_fits(os.path.join(args.save_path, 'ws_restored.fits'))

