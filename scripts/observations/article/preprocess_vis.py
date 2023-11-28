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

    # import polyclean.image_utils as ut
    # ut.myplot_uvcoverage(vis, title="Subsampled UV coverage")

    vis_path = os.path.join(os.getcwd(), vis_path)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    pklname = 'vis.pkl'
    msname = "ssms.ms"
    export_visibility_to_ms(os.path.join(vis_path, msname), [vis], )
    with open(os.path.join(vis_path, pklname), 'wb') as handle:
        pickle.dump(vis, handle)
