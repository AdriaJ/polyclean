import logging
import sys
import pickle
import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import time

from astropy.wcs.utils import skycoord_to_pixel

from ska_sdp_func_python.imaging import invert_visibility, create_image_from_visibility
from ska_sdp_func_python.image import restore_cube, fit_psf
from ska_sdp_datamodels.visibility.vis_utils import generate_baselines
from rascil.processing_components.visibility.base import export_visibility_to_ms
from rascil.processing_components.image.operations import import_image_from_fits


nantennas = 28
ntimes = 50
npixel = 1024
fov_deg = 6.
context = "ng"

algorithm = 'hogbom'
niter = 500
gain = 0.1
n_major = 10

if __name__ == "__main__":
    data_path = "/home/jarret/Documents/EPFL/PhD/ra_data/"
    ws_dir = "/home/jarret/PycharmProjects/polyclean/scripts/observations/wsclean-dir"


    # # subsample the MS file
    # from casacore import tables
    #
    # input_ms = 'BOOTES24_SB180-189.2ch8s_SIM.ms'
    # ms = tables.table(data_path + input_ms, readonly=True)  # Open with write access
    # output_ms = 'ssms.ms'
    # ms_out = tables.table(ws_dir + '/' + output_ms, tabledesc=ms.getdesc(), nrow=0, readonly=False)
    # # Specify parameters
    # data_column = 'DATA_SIMULATED'  # Specify the data column you want to extract
    # selected_frequency_channel = 0  # Choose the desired frequency channel (zero-based index)
    # subsampling_factor = 50  # Subsample every 50 time steps
    # num_antennas = 28  # Number of antennas to keep
    #
    # # Get unique antenna IDs from the MS
    # antenna_ids = set(ms.getcol('ANTENNA1')) | set(ms.getcol('ANTENNA2'))
    # selected_antennas = sorted(antenna_ids)[:num_antennas]
    #
    # # Iterate through the rows and copy selected data to the output MS
    # for row in ms:
    #     time = row['TIME']
    #     antenna1, antenna2 = row['ANTENNA1'], row['ANTENNA2']
    #
    #     # Subsample every subsampling_factor time steps
    #     if time % subsampling_factor == 0:
    #         # Select data for the chosen frequency channel and antennas
    #         if antenna1 in selected_antennas and antenna2 in selected_antennas:
    #             # Create a new row in the output MS and copy data
    #             selected_data = row[data_column][selected_frequency_channel]
    #             print("Got 1!")
    #             row[data_column] = selected_data
    #             ms_out.addrows(row)
    #
    # ms.close()
    # ms_out.close()
    #
    # from casacore.tables import table, tablecopy, taql
    #
    # input_ms = 'BOOTES24_SB180-189.2ch8s_SIM.ms'
    # ms = tables.table(data_path + input_ms, readonly=True)  # Open with write access
    # output_ms = 'ssms.ms'
    # ms_out = tables.table(ws_dir + '/' + output_ms, tabledesc=ms.getdesc(), nrow=0, readonly=False)
    # # Specify parameters
    # data_column = 'DATA_SIMULATED'  # Specify the data column you want to extract
    # selected_frequency_channel = 0  # Choose the desired frequency channel (zero-based index)
    # subsampling_factor = 50  # Subsample every 50 time steps
    # num_antennas = 28  # Number of antennas to keep
    #
    # # Get unique antenna IDs from the MS
    # antenna_ids = set(ms.getcol('ANTENNA1')) | set(ms.getcol('ANTENNA2'))
    # selected_antennas = sorted(antenna_ids)[:num_antennas]
    # ms.selectchannel()
    # ms.select({'antenna1': selected_antennas, 'antenna2': selected_antennas})
    #
    # # Convert selected_antennas to a string for the TAQL query
    # antenna_list = ','.join(str(antenna_id) for antenna_id in selected_antennas)
    #
    # # Define the TAQL query to select data for the selected antennas
    # taql_query = f'SELECT * FROM {input_ms} WHERE ANTENNA1 in [{antenna_list}] AND ANTENNA2 in [{antenna_list}]'
    #
    # # Copy the data for the selected antennas using tablecopy
    # tablecopy(ms.query(taql_query), ms_out, deep=True)
    #
    # # Close the MS tables
    # ms.close()
    # ms_out.close()

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
        f"wsclean -auto-threshold 0.0001 -size {npixel:d} {npixel:d} -scale {fov_deg / npixel:.6f} -mgain 0.7 "
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

    import polyclean.image_utils as ut
    from matplotlib import use
    use("Qt5Agg")
    ut.plot_image(ws_restored, title=f"Rascil restored image {niter:d} iterations", log=True)
    ut.plot_image(wsclean_image, title="WS-CLEAN image")

    print(f"Error : {np.linalg.norm(ws_restored.pixels.data - wsclean_image.pixels.data)/np.linalg.norm(ws_restored.pixels.data):.3e}")

    print(f'max : {ws_restored.pixels.data.max()} - {wsclean_image.pixels.data.max()}')
    print(f'min : {ws_restored.pixels.data.min()} - {wsclean_image.pixels.data.min()}')
