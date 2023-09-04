#!/bin/bash
# Fill the dataframes rows
# Repeat each experiment nreps times

rmax=(300) # (300 600 900 1200 1500 2000 3000)
nreps=1
save="true"

for r in "${rmax[@]}"; do
    for ((i=1; i<=nreps; i++)); do
        echo "-----  Simulation for rmax = $r, repetition $i / $nreps  -----"
        if [ $save = "true" ]; then
            ./fill_one_row.sh --save "$r"
        else
            ./fill_one_row.sh "$r"
        fi
    done
done
