#!/bin/bash
# Fill one row in the dataframe

# Simulate the problem and save ms file: Python
#       -> save noisy ms file and ground truth source image + npix and seed + nvis
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

# Initialize default values
save="false"
rmax=""

# Process optional arguments
while [ $# -gt 0 ]; do
    case $1 in
        --save)
            save="true"
            shift
            ;;
        *)
            break  # Exit the loop when the first non-option argument is encountered
            ;;
    esac
done

# Assign remaining arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [--save] <rmax>"
    exit 1
fi
rmax="$1"

echo "Simulation for rmax = $rmax"

#conda activate pclean
mkdir -p tmpdir
python simulate_pb.py --rmax "$rmax"
if [ $save = "true" ]; then
    echo "Saving LASSO reconstructions..."
    save_dir="reconstructions/rmax_$rmax"
    mkdir -p $save_dir
    python solve_lasso.py --save_path "$save_dir"
else
    python solve_lasso.py
fi
# load ws-clean parameters
IFS=$'\n' read -r -d '' npix cellsize < tmpdir/ws_args.txt
start_time="$(date -u +%s.%N)"
wsclean -auto-threshold 1 -size "$npix" "$npix" -scale "$cellsize" -mgain 0.7 -niter 10000 -weight natural -name ws -quiet tmpdir/data.ms
end_time="$(date -u +%s.%N)"
elapsed="$(bc <<<"$end_time-$start_time")"
echo "Total of $elapsed seconds elapsed for wsclean"
echo $elapsed > tmpdir/wsclean_time.txt
mv ws-* tmpdir
if [ $save = "true" ]; then
    echo "Saving WS-CLEAN reconstruction ..."
    save_dir="reconstructions/rmax_$rmax"
    mkdir -p $save_dir
    cp tmpdir/ws-image.fits $save_dir
    cp tmpdir/ws-dirty.fits $save_dir
    python pp_wsclean.py --save_path "$save_dir"
else
    python pp_wsclean.py
fi
python fill_df.py
rm -r -d tmpdir

#mkdir -p archive
#mv --backup=t props.csv archive/props.csv
#mv --backup=t metrics.csv archive/metrics.csv
