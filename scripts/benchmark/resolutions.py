import yaml
import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord

from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_func_python.imaging import advise_wide_field

import polyclean.ra_utils as pcrau


rmaxs = [300, 600, 900, 1200, 1500, 2000, 3000, 6000]
srfs = [2, 5, 10]

keys = ['nominal', 'rascil'] + [f'srf {srf:d}' for srf in srfs]
res = {k: [] for k in keys}

TMP_DATA_DIR = 'tmpdir'

if __name__ == "__main__":
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    ntimes = config['ra_config']['ntimes']

    for rmax in rmaxs:
        lowr3 = create_named_configuration("LOWBD2", rmax=rmax)
        phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000")

        # visibility template
        vt = create_visibility(
            lowr3,
            (2 / (ntimes - 1) * np.arange(ntimes) - 1) * np.pi / 3,
            np.r_[config['ra_config']['frequency']],
            channel_bandwidth=np.r_[config['ra_config']['channel_bandwidth']],
            weight=1.0,
            phasecentre=phasecentre,
            polarisation_frame=PolarisationFrame("stokesI"),
        )

        advice = advise_wide_field(
            vt, guard_band_image=6.0, delA=0.1, oversampling_synthesised_beam=3.0
        )
        rascil_res = advice["cellsize"]  # radians
        res['rascil'].append(rascil_res)

        nominal_res = 1 / (3 * np.linalg.norm(vt.visibility_acc.uvw_lambda, axis=-1).max())  # radians
        res['nominal'].append(nominal_res)

        # side size of the image
        for srf in srfs:
            npix = pcrau.get_npixels(vt, config['ra_config']['fov_deg'], phasecentre, config['lasso_params']['nufft_eps'],
                                     srf=srf)
            resolution = np.pi * config['ra_config']['fov_deg'] / (180 * npix)
            res[f'srf {srf:d}'].append(resolution)

    print(res)
    df = pd.DataFrame.from_dict(res)
    df[:] *= 180 / np.pi * 3600  # arcsec
    df.drop('rascil', axis=1, inplace=True)
    df.index = rmaxs
    # df.index.name = 'rmax'
    df.index /= 1000
    print(df.to_latex(formatters={'rmax': "%.1f"}, float_format="%.1f"))

