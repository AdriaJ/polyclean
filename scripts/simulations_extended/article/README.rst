Reconstruction of diffuse emissions
===================================

Files to reproduce the results of section 6.4. We use simulated measurements from an image of the M31
galaxy. PolyCLEAN perform a sparse dictionary reconstruction with a synthesis LASSO optimization problem.

1. How to reproduce the results?
--------------------------------

In order to reproduce the results, you need to run the following scripts::

    > python reconstructions.py
    > python plots.py

The plots of the article are then produced and displayed with `matplotlib`.

All the reconstruction parameters can be modified in the text of the script, including the penalty parameter, the
bias in the choice of the dictionary, the scale of the gaussians, etc. The simulated source image is provided by
RASCIL and loaded with a RASCIL function.

2. Description of the files
---------------------------

Many files are provided in the folder.

- `gauss_pclean.py`
    Provides the optimization algorithm, which is an extension of the base class PolyCLEAN. This algorithm solves the
    dictionary sparse synthesis LASSO problem. The operator that implements the dictionary atoms, namely the
    convolution with some kernels of various scales, are defined in the file `kernels.py` in the main source folder of
    `PollyCLEAN`.
- `reconstructions.py`
    This files loads the source image, simulates the measurements, and performs the reconstruction with the three
    methods considered: PolyCLEAN, WS-CLEAN and CLEAN from RASCIL (with major cycle). The reconstructions are stored in
    a nested folder `reco_pkl/` as pickle files (WS-CLEAN reconstructions first appear in the `wsclean-dir/` before
    being processed and stored as pickle files as well).
- `plots.py`
    Produce the comparative plots of the article.



