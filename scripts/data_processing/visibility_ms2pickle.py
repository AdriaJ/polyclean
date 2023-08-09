"""
Loads the visibility from a MS file, export them as a pickle file.
"""
import logging
import sys
import pickle
import time

from rascil.processing_components.visibility import create_visibility_from_ms, list_ms
from ska_sdp_func_python.visibility import convert_visibility_to_stokesI

if __name__ == "__main__":
    log = logging.getLogger("rascil-logger")
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))

    path = "/home/jarret/Documents/EPFL/PhD/ra_data/"
    msname = path + "BOOTES24_SB180-189.2ch8s_SIM.ms"
    print("Source and data descriptors in the measurement set: " + str(list_ms(msname)))

    print("Loading MS file: ...")
    start = time.time()
    total_vis = create_visibility_from_ms(msname, datacolumn="DATA_SIMULATED", channum=range(0, 1))[0]
    print("\tDone in {:.2f}s".format(time.time() - start))

    print("Convert to StokesI: ...")
    start = time.time()
    total_vis = convert_visibility_to_stokesI(total_vis)
    print("\tDone in {:.2f}s".format(time.time() - start))

    pklname = path + "bootes.pkl"
    with open(pklname, 'wb') as handle:
        pickle.dump(total_vis, handle, protocol=-1)

