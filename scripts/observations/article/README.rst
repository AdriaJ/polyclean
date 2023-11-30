Comparative image reconstruction
================================

Files to reproduce the results of section 6.2 and 6.3.

1. How to reproduce the results?
--------------------------------

1.1 Image comparison
^^^^^^^^^^^^^^^^^^^^

To reproduce the plots of Section 6.2, you need to run the following python scripts::

    > python pclean.py
    > python wsclean.py
    > python plot_reconstructions.py

Three reconstructions are performed with different parameters. PolyCLEAN solves the LASSO problem with different
penalty parameters, relative to the maximum value $\lambda_\mathrm{max}$ (see the article). CLEAN use different
stopping threshold, automatically computed with the argument `auto-threshold` of wsclean. The different values used
are accessible in the files::

    lambda_factors = [0.05, 0.02, 0.005]
    thresholds = [1, 2, 3]


1.2 Dual certificate
^^^^^^^^^^^^^^^^^^^^

As for the dual certificate, in Section 6.3, we use the following scripts::

    > python dual_certificate.py
    > python plot_certif.py

2. Description of the files
---------------------------

The data have already been preprocessed and stored as a `.pkl` and `.ms` files in the `vis/` folder.

- `pclean.py`
    Provide LASSO reconstruction using PolyCLEAN for 3 values of the penalty parameters. The reconstructions are
    stored in the `reco_pkl/` folder.
- `wsclean.py`
    Compute 3 CLEAN reconstructions with different stopping criterion using the `WS-CLEAN` software. The intermediate
    CLEAN reconstructions appear in a `wsclean-dir` as temporary files before being treated with the Python code and
    stored in the `reco_pkl/` folder.
- `plot_reconstructions.py`
    Produce the plots of the article.
- `dual_certificate.py`
    Compute the PolyCLEAN reconstruction and the associated dual certificate. Both are stored as `.pkl` files in the
    folder `reco_pkl/`.
- `plot_certif.py`
    Plot the obtained dual certificate with the colormap of the article.

