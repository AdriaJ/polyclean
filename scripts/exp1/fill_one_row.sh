#!/bin/bash
# Fill one row in the dataframe

read -r rmax <<< "$1"

echo "Simulation for rmax = $rmax"

#conda activate pclean
mkdir -p tmpdir
python simulate_pb.py --rmax "$rmax"
python solve_lasso.py  #todo -----------> to test now
#load ws-clean parameters
IFS=$'\n' read -r -d '' npix cellsize < tmpdir/ws_args.txt
start_time="$(date -u +%s.%N)"
wsclean -auto-threshold 1 -size "$npix" "$npix" -scale "$cellsize" -mgain 0.7 -niter 10000 -weight natural -name ws -quiet -no-dirty tmpdir/data.ms
end_time="$(date -u +%s.%N)"
elapsed="$(bc <<<"$end_time-$start_time")"
echo "Total of $elapsed seconds elapsed for wsclean"
echo $elapsed > tmpdir/wsclean_time.txt
mv ws-* tmpdir
python pp_wsclean.py  #todo -----------
python fill_df.py
rm -r -d tmpdir

#mkdir -p archive
#mv --backup=t props.csv archive/props.csv
#mv --backup=t metrics.csv archive/metrics.csv

#todo update with Pyxu and the new lipschitz api + function scope for lasso reconstructions


  # Simulate the problem and save ms file: Python
  #       -> save noisy ms file and ground truth source image + npix and seed
  #       -> Creates: data.ms, gtimage.pkl, rmax_npix_seed.pkl (dict), ws_args.txt (npix and cellsize on two lines)
  # Run the LASSO solvers: Python
  #       -> Open the ms file and perform the reconstruction, encapsulate everything in functions
  #       -> export output to pickle
  #       -> Creates: lasso_res.pkl (dictionary), lips_t.pkl (dict), clean_beam.pkl (dict), gt_conv.pkl (image)
  # WS-CLEAN: load, run, save and move results + time: bash
  #       -> save time to txt file
  #       -> Creates: ws-model.fits, ws-image.fits, ws-psf.fits, ws-residual.fits
  # post process ws-clean: Python
  #       -> load ws-clean data
  #       -> compute metrics and export to pickle
  #       -> Creates: clean_res.pkl (dict)
  # Fill the dataframe: Python  -----------------------------------------------
  #       -> load the pickle files and fill df
  # remove tmp folders
