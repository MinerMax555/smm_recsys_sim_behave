#!/usr/bin/bash

# Clear the saved folder
rm -rf ./saved/*

ITERATIONS=10
DATASET=smm_demo
MODEL=NeuMF

# Run the training
for i in $(seq 1 $ITERATIONS); do
  python3 run_loop.py $DATASET --control_group --control-group-country DE --only-from-country -cc US -m $MODEL --loop $i --resume | tee ./log/"$MODEL"_"$DATASET"_"$(date +%T)"_stdall.log
done
