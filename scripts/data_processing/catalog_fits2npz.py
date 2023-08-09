"""
Loads the catalog information from a FITS file, export them as an npz file.
"""
import numpy as np
from astropy.io import fits

# catalogpath = "/home/jarret/Documents/EPFL/PhD/ra_data/bootes_pybdsf_source.fits"
catalogpath = "/home/jarret/Documents/EPFL/PhD/ra_data/LOFAR150_BOOTES.fits"

with fits.open(catalogpath) as hdul:
    hdul.info()
    col_names = ['RA', 'DEC', 'Peak_flux']
    src_ra_dec_flux = np.vstack([hdul[1].data[n] for n in col_names])  # degrees
    src_ra_dec_flux[:2, :] *= (np.pi / 180)  # radians

    data = hdul[1].data
npz_name = "/home/jarret/Documents/EPFL/PhD/ra_data/bootes_catalog.npz"
np.savez(npz_name, src_ra_dec_flux=src_ra_dec_flux)
