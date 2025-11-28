import pickle
import os
import time
import numpy as np

from ska_sdp_func_python.imaging import invert_visibility, create_image_from_visibility
from ska_sdp_func_python.image import restore_cube, fit_psf
from rascil.processing_components.image.operations import import_image_from_fits

thresholds = [1, 2, 3]

npixel = 3072
nantennas = 50

fov_deg = 6.
context = "ng"
niter = 100_000

save_im_pkl = True

if __name__ == "__main__":
    with open(os.path.join(os.getcwd(), "vis", "vis.pkl"), 'rb') as handle:
        vis = pickle.load(handle)
    print("Selected vis: ", vis.dims)
    ws_dir = os.path.join(os.getcwd(), 'wsclean-dir')
    if not os.path.exists(ws_dir):
        os.makedirs(ws_dir)

    phasecentre = vis.phasecentre
    fov = fov_deg * np.pi / 180.
    cellsize = fov / npixel

    image_model = create_image_from_visibility(vis, npixel=npixel, cellsize=cellsize, override_cellsize=False)
    psf, sumwt = invert_visibility(vis, image_model, context=context, dopsf=True)
    clean_beam = fit_psf(psf)
    sharp_beam = clean_beam.copy()
    sharp_beam["bmin"] = clean_beam["bmin"] / 2
    sharp_beam["bmaj"] = clean_beam["bmaj"] / 2

    durations = []

    for thresh in thresholds:
        start = time.time()
        os.system(
            f"wsclean -auto-threshold {thresh} -size {npixel:d} {npixel:d} -scale {fov_deg / npixel:.6f} -mgain 0.7 "
            f"-niter {niter:d} -name ws -weight natural -quiet -no-dirty vis/ssms.ms")
        print("\tRun in {:.3f}s".format(dt_wsclean := time.time() - start))
        os.system(f"mv ws-* wsclean-dir/")
        durations.append(dt_wsclean)

        wsclean_model = import_image_from_fits(ws_dir + '/' + f"ws-model.fits")
        wsclean_residual = import_image_from_fits(ws_dir + '/' + f"ws-residual.fits")
        ws_restored = restore_cube(wsclean_model, None, wsclean_residual, clean_beam=clean_beam)
        comp_restored = restore_cube(wsclean_model, None, None, clean_beam=clean_beam)
        ws_restored_sharp = restore_cube(wsclean_model, None, wsclean_residual, clean_beam=sharp_beam)
        comp_restored_sharp = restore_cube(wsclean_model, None, None, clean_beam=sharp_beam)

        # wsclean_image = import_image_from_fits(ws_dir + '/' + f"ws-image.fits")


        folder_path = f"/home/jarret/PycharmProjects/polyclean/figures/lofar_ps/wsclean/autothresh{thresh}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if save_im_pkl:
            folder_path = os.path.join(os.getcwd(), 'reco_pkl', str(nantennas) + 'antennas', f'autothresh{thresh}')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            with open(os.path.join(folder_path, "restored.pkl"), 'wb') as handle:
                pickle.dump(ws_restored, handle)
            with open(os.path.join(folder_path, "restored_sharp.pkl"), 'wb') as handle:
                pickle.dump(ws_restored_sharp, handle)
            with open(os.path.join(folder_path, "comp_restored.pkl"), 'wb') as handle:
                pickle.dump(comp_restored, handle)
            with open(os.path.join(folder_path, "comp_restored_sharp.pkl"), 'wb') as handle:
                pickle.dump(comp_restored_sharp, handle)
            with open(os.path.join(folder_path, "model.pkl"), 'wb') as handle:
                pickle.dump(wsclean_model, handle)
            with open(os.path.join(folder_path, "residual.pkl"), 'wb') as handle:
                pickle.dump(wsclean_residual, handle)

            with open(os.path.join(os.getcwd(), 'reco_pkl', str(nantennas) + 'antennas', 'report_wsclean.txt'), "w+") as file:
                file.write("WS-CLEAN reconstructions:\n")
                file.write(f"Autothresholds: {thresholds}\n")
                file.write(f"Durations: " + str(durations) + "\n")

    os.system(f"rm -rd wsclean-dir")
