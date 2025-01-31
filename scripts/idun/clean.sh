#!/bin/sh

timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p "logs/${timestamp}"
cp srun.out srun.err "logs/${timestamp}"
