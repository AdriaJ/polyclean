import argparse
import os.path
import pickle
import yaml

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_func_python.util import skycoord_to_lmn
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_func_python.imaging import predict_visibility
from rascil.processing_components.visibility.base import export_visibility_to_ms

import pyxu.util.complex as pxc
import polyclean.ra_utils as pcrau



TMP_DATA_DIR = 'tmpdir'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simulate a sky and save the ground truth image, as well as a noisy measurement set.')
    parser.add_argument('-r', '--rmax', type=float, required=True,
                        help="Max distance to the center of the configuration to select the stations.")
    parser.add_argument('-s', '--seed',
                        help="Optional seed for reproducibility.")
    args = parser.parse_args()
    rmax = args.rmax
    if args.seed is None:
        seed = np.random.randint(10_000)
    else:
        seed = args.seed
    print(f"Seed: {seed}")

    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    ntimes = config['ra_config']['ntimes']

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

    # side size of the image
    npix = pcrau.get_npixels(vt, config['ra_config']['fov_deg'], phasecentre, config['lasso_params']['nufft_eps'])
    print(f"number of pixels: {npix:d}")

    # simulation of a sky image
    sky_im, sc = pcrau.generate_point_sources(config['ra_config']['nsources'],
                                              config['ra_config']['fov_deg'],
                                              npix,
                                              flux_sigma=.8,
                                              radius_rate=.9,
                                              phasecentre=phasecentre,
                                              frequency=np.r_[config['ra_config']['frequency']],
                                              channel_bandwidth=np.r_[config['ra_config']['channel_bandwidth']],
                                              seed=seed)
    # parametrization of the image
    # directions = SkyCoord(
    #     ra=sky_im.ra_grid.data.ravel() * u.rad,
    #     dec=sky_im.dec_grid.data.ravel() * u.rad,
    #     frame="icrs",
    #     equinox="J2000",
    # )
    # direction_cosines = np.stack(skycoord_to_lmn(directions, phasecentre), axis=-1)

    predicted_visi = predict_visibility(vt, sky_im, context=config['clean_params']['context'])
    real_visi = pxc.view_as_real(predicted_visi.vis.data[:, :, 0, 0] * predicted_visi.weight.data[:, :, 0, 0])
    noise_scale = np.abs(real_visi).max() * 10 ** (-config['ra_config']['psnr_db'] / 20) / np.sqrt(2)
    noise = np.random.normal(0, noise_scale, real_visi.shape)
    predicted_visi.vis.data += pxc.view_as_complex(noise)[:, :, None, None]

    with open(os.path.join(TMP_DATA_DIR, 'gtimage.pkl'), 'wb') as file:
        pickle.dump(sky_im, file)
    with open(os.path.join(TMP_DATA_DIR, 'rmax_npix_seed.pkl'), 'wb') as file:
        pickle.dump({'rmax': rmax, 'npix': npix, 'seed': seed}, file)
    with open(os.path.join(TMP_DATA_DIR, 'ws_args.txt'), 'w') as file:
        file.write(f"{npix:d}" + '\n')
        file.write(f"{config['ra_config']['fov_deg']/npix:.9f}")
    with open(os.path.join(TMP_DATA_DIR, 'data.pkl'), 'wb') as file:
        pickle.dump(predicted_visi, file)
    export_visibility_to_ms(os.path.join(TMP_DATA_DIR, 'data.ms'), [predicted_visi], )

