#!/bin/bash
# Fill the dataframes rows
# Repeat each experiment nreps times

rmax=(300 600 900 1200 1500 2000 3000)
nreps=2

for r in "${rmax[@]}"; do
    for ((i=1; i<=nreps; i++)); do
        echo "-----  Simulation for rmax = $r, repetition $i / $nreps  -----"
        ./fill_one_row.sh "$r"
    done
done
