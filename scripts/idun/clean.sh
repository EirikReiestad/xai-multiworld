#!/bin/sh

timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p "logs/${timestamp}"
mv srun.out srun.err log.txt "logs/${timestamp}"
