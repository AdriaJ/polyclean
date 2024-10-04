import pickle
import os

from ska_sdp_datamodels.visibility.vis_utils import generate_baselines
from rascil.processing_components.visibility.base import export_visibility_to_ms

data_path = "/home/jarret/Documents/EPFL/PhD/ra_data/"
vis_path = "vis"
pklname = data_path + "bootes.pkl"

nantennas = 28
ntimes = 50

if __name__ == "__main__":
    with open(pklname, 'rb') as handle:
        total_vis = pickle.load(handle)

    print("Dimensions of the whole visibility set:", total_vis.dims)

    vis = total_vis.isel({"time": slice(0, total_vis.dims["time"], ntimes)})
    vis = vis.sel({"baselines": list(generate_baselines(nantennas)), })
    vis.attrs.update({'configuration': vis.configuration.isel({'id': slice(nantennas)})})
    print("Selected vis: ", vis.dims)
    # Broken antennas: 12, 13, 16, 17, 47
    # Unusable baseline: (22, 23)

    import polyclean.image_utils as ut
    ut.myplot_uvcoverage(vis, title="Subsampled UV coverage", show_non_valid=False)
    # ut.myplot_uvcoverage(vis.isel({"time": slice(35, 36)}), show_non_valid=False)

    vis_path = os.path.join(os.getcwd(), vis_path)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    pklname = 'vis.pkl'
    msname = "ssms.ms"
    export_visibility_to_ms(os.path.join(vis_path, msname), [vis], )
    with open(os.path.join(vis_path, pklname), 'wb') as handle:
        pickle.dump(vis, handle)


    # Show UV coverage
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(6, 4))
    uvw_valid = vis.visibility_acc.uvw_lambda.reshape((-1, 3))[vis['flags'].data.flatten() == 0]
    u = uvw_valid[..., 0]
    v = uvw_valid[..., 1]
    plt.plot(u, v, "o", color="b", markersize=0.5, label="Valid")
    plt.plot(-u, -v, "o", color="b", markersize=0.5)
    plt.xlabel("U (wavelengths)", fontsize=11)
    plt.ylabel("V (wavelengths)", fontsize=11)
    lim = 1.05 * max(np.abs(u).max(), np.abs(v).max())
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    plt.title("Subsampled UV coverage")
    plt.subplots_adjust(left=0.16, right=0.95, top=0.9, bottom=0.15)
    plt.show(block=False)
