import pickle
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import polyclean.image_utils as ut

from ska_sdp_func_python.image import restore_cube
from rascil.processing_components.image.operations import import_image_from_fits

from matplotlib import use
use("Qt5Agg")

folder = "/home/jarret/PycharmProjects/polyclean/scripts/time_comparison/"
filename1 = "res300_600_900.pkl"
filenames = ["res1500.pkl"]

if __name__ == "__main__":
    with open(folder + filename1, 'rb') as file:
        res = pickle.load(file)
    methods = res["methods"]  # list  - no change!

    # with open(folder + filenames[0], 'wb') as file:
    #     pickle.dump(d, file)

    for f in filenames:
        with open(folder + f, 'rb') as file:
            # pickle.dump(res, file)
            d = pickle.load(file)
            d.pop('methods')
            for k in d.keys():
                if isinstance(d[k], list):
                    res[k] += d[k]
                elif isinstance(d[k], dict):
                    for m in methods:
                        if res[k][m] is not None:
                            res[k][m] += d[k][m]

    rmax = res['rmax']  # list
    npixels = res["npixels"]  # list
    durations = res["durations"]  # dict
    components = res["components"]  # dict
    residuals = res["residuals"]  # dict
    dcvs = res["dcvs"]  # dict
    beams = res["beams"]  # list
    lips_durations = res["lips_durations"]  # list
    objective_func = res["objective_func"]  # dict
    sources = res["sources"]  # list
    durations_wsclean = res["durations_wsclean"]  # list

    # MSE and MAD
    mse = pd.DataFrame(columns=methods, index=rmax)
    mad = pd.DataFrame(columns=methods, index=rmax)
    restored_sources = [restore_cube(im, None, None, b) for im, b in zip(sources, beams)]
    # restore_list(sources, None, None, beams)
    for m in methods:
        restored_comp = [restore_cube(im, None, None, b) for im, b in zip(components[m], beams)]
        # restored_comp = restore_list(components[m], None, None, beams)
        mse[m] = [ut.MSE(s, c) for s, c in zip(restored_sources, restored_comp)]
        mad[m] = [ut.MAD(s, c) for s, c in zip(restored_sources, restored_comp)]

    ws_folder = "/home/jarret/PycharmProjects/polyclean/scripts/time_comparison/wsclean-dir"
    wsclean_models = [import_image_from_fits(ws_folder + '/' + f"rmax{r:d}/ws-rmax{r:d}-model.fits") for r in rmax]
    ws_restored = [restore_cube(mod, None, None, b) for mod, b in zip(wsclean_models, beams)]
    mse["WS-CLEAN"] = [ut.MSE(s, c) for s, c in zip(restored_sources, ws_restored)]
    mad["WS-CLEAN"] = [ut.MAD(s, c) for s, c in zip(restored_sources, ws_restored)]

    # Dual certificate
    dcvs.pop('CLEAN')
    dcv = pd.DataFrame.from_dict(dcvs)
    dcv.index = rmax

    # Summary
    summ = pd.DataFrame.from_dict({"rmax": rmax, "N pixels": npixels, "Lipschitz time": lips_durations},)
    # print(summ.to_latex(index=False, caption="Summary"))

    # Total duration
    total_lips = sum(lips_durations)
    total_duration = sum([sum(l) for l in durations.values()])
    print("Total time of simulation: ", dt.timedelta(seconds=total_duration + total_lips))

    # Time comparison
    indices = np.arange(len(rmax))
    plt.figure()
    for m in methods:
        plt.scatter(indices, durations[m], label=m, marker='x')
    plt.scatter(indices, 40 * np.array(durations["PolyCLEAN"]), marker=7, s=300)
    plt.scatter(indices, durations_wsclean, label="WS-CLEAN", marker='x')
    plt.xticks(indices, rmax)
    plt.legend()
    plt.xlabel("rmax (m)")
    plt.ylabel("Duration (s)")
    plt.title("Comparative reconstruction durations")
    plt.show()
    save = False
    if save:
        plt.savefig("/home/jarret/PycharmProjects/polyclean/examples/figures/profiles" + "/exp1.pdf")

