import pickle
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from ska_sdp_func_python.imaging import invert_visibility, create_image_from_visibility
from ska_sdp_func_python.image import restore_cube, fit_psf
from ska_sdp_datamodels.visibility.vis_utils import generate_baselines
from rascil.processing_components.visibility.base import export_visibility_to_ms
from rascil.processing_components.image.operations import import_image_from_fits

from pclean import plot_1_image

nantennas = 28
ntimes = 50
npixel = 1024
fov_deg = 6.
context = "ng"

thresh = 3
niter = 10_000

save = False
save_im_pkl = True

if __name__ == "__main__":
    data_path = "/home/jarret/Documents/EPFL/PhD/ra_data/"
    ws_dir = "/home/jarret/PycharmProjects/polyclean/scripts/observations/wsclean-dir"

    pklname = data_path + "bootes.pkl"
    with open(pklname, 'rb') as handle:
        total_vis = pickle.load(handle)

    vis = total_vis.isel({"time": slice(0, total_vis.dims["time"], ntimes),})
    vis = vis.sel({"baselines": list(generate_baselines(nantennas)), })
    vis.attrs.update({'configuration': vis.configuration.isel({'id': slice(nantennas)})})
    print("Selected vis: ", vis.dims)
    # Broken antennas: 12, 13, 16, 17, 47
    # Unusable baseline: (22, 23)
    # ut.myplot_uvcoverage(vis, title="Subsampled UV coverage")

    phasecentre = vis.phasecentre
    fov = fov_deg * np.pi / 180.
    cellsize = fov / npixel

    print("Field of view in degrees: {:.3f}".format(fov_deg))

    if not os.path.exists(ws_dir):
        os.makedirs(ws_dir)
    filename = "ssms.ms"
    export_visibility_to_ms(ws_dir + "/" + filename, [vis], )
    start = time.time()
    os.system(
        f"wsclean -auto-threshold {thresh} -size {npixel:d} {npixel:d} -scale {fov_deg / npixel:.6f} -mgain 0.7 "
        f"-niter {niter:d} -name ws -weight natural -quiet -no-dirty wsclean-dir/ssms.ms")
    print("\tRun in {:.3f}s".format(dt_wsclean := time.time() - start))
    os.system(f"mv ws-* wsclean-dir/")

    image_model = create_image_from_visibility(vis, npixel=npixel, cellsize=cellsize, override_cellsize=False)
    psf, sumwt = invert_visibility(vis, image_model, context=context, dopsf=True)

    wsclean_model = import_image_from_fits(ws_dir + '/' + f"ws-model.fits")
    wsclean_residual = import_image_from_fits(ws_dir + '/' + f"ws-residual.fits")
    ws_restored = restore_cube(wsclean_model, psf, wsclean_residual)

    wsclean_image = import_image_from_fits(ws_dir + '/' + f"ws-image.fits")

    # mse["WS-CLEAN"] = [ut.MSE(s, c) for s, c in zip(restored_sources, ws_restored)]
    # mad["WS-CLEAN"] = [ut.MAD(s, c) for s, c in zip(restored_sources, ws_restored)]
    #todo compare wsclean reconstruciton and clean beam cvonvolution with rascil

    # import polyclean.image_utils as ut
    # ut.plot_image(ws_restored, title=f"Rascil restored image {niter:d} iterations", log=True)
    # ut.plot_image(wsclean_image, title="WS-CLEAN image")

    folder_path = f"/home/jarret/PycharmProjects/polyclean/figures/lofar_ps/wsclean/autothresh{thresh}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plot_1_image(ws_restored, f"WS-CLEAN: {thresh} auto-threshold - {dt_wsclean:.2f}s")
    if save:
        plt.savefig(os.path.join(folder_path, "restored.png"))

    plot_1_image(ws_restored, f"WS-CLEAN: {thresh} auto-threshold - {dt_wsclean:.2f}s", vlim=186)
    if save:
        plt.savefig(os.path.join(folder_path, "restored_unified.png"))

    print(f"Error : {np.linalg.norm(ws_restored.pixels.data - wsclean_image.pixels.data)/np.linalg.norm(ws_restored.pixels.data):.3e}")

    print(f'max : {ws_restored.pixels.data.max()} - {wsclean_image.pixels.data.max()}')
    print(f'min : {ws_restored.pixels.data.min()} - {wsclean_image.pixels.data.min()}')

    if save_im_pkl:
        folder_path = f"/home/jarret/PycharmProjects/polyclean/scripts/observations/reco_pkl/autothresh{thresh}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(folder_path + "/restored.pkl", 'wb') as handle:
            pickle.dump(ws_restored, handle)

    # import polyclean.image_utils as ut
    # ut.myplot_uvcoverage(vis, title="Subsampled UV coverage")