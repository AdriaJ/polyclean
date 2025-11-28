import os
import pickle

from ska_sdp_func_python.imaging.imaging import invert_visibility
from ska_sdp_func_python.image.deconvolution import fit_psf, restore_list
from rascil.processing_components.image.operations import import_image_from_fits

if __name__ == "__main__":
    wsdir = os.path.join(os.getcwd(), 'wsclean-dir')
    imdir = os.path.join(os.getcwd(), 'images')
    with open(os.path.join(wsdir, 'vis.pkl'), 'rb') as handle:
        vis = pickle.load(handle)

    # load the model images
    ws_lr = import_image_from_fits(os.path.join(wsdir, 'ws-lr-model.fits'))
    ws_hr = import_image_from_fits(os.path.join(wsdir, 'ws-hr-model.fits'))
    pc_lr = import_image_from_fits(os.path.join(imdir, 'pc-lr-model.fits'))
    pc_hr = import_image_from_fits(os.path.join(imdir, 'pc-hr-model.fits'))
    source_lr = import_image_from_fits(os.path.join(imdir, 'source_low_res.fits'))
    source_hr = import_image_from_fits(os.path.join(imdir, 'source_high_res.fits'))

    model_hr = ws_hr.copy(deep=True)
    model_hr.pixels.data[:] = 0

    psf, _ = invert_visibility(vis, ws_lr, context='ng', dopsf=True)
    clean_beam = fit_psf(psf)
    sharp_beam = clean_beam.copy()
    sharp_beam["bmin"] = clean_beam["bmin"] / 10
    sharp_beam["bmaj"] = clean_beam["bmaj"] / 10

    restored = restore_list([ws_lr, ws_hr, pc_lr, pc_hr, source_lr, source_hr],
                            None, None, clean_beam=clean_beam)
    restored_sharp = restore_list([ws_lr, ws_hr, pc_lr, pc_hr, source_lr, source_hr],
                                  None, None, clean_beam=sharp_beam)

    for im, name in zip(restored, ['ws-lr', 'ws-hr', 'pc-lr', 'pc-hr', 'source-lr', 'source-hr']):
        im.image_acc.export_to_fits(os.path.join(imdir, f'{name}-restored.fits'))
    for im, name in zip(restored_sharp, ['ws-lr', 'ws-hr', 'pc-lr', 'pc-hr', 'source-lr', 'source-hr']):
        im.image_acc.export_to_fits(os.path.join(imdir, f'{name}-restored-sharp.fits'))

