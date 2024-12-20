#!/bin/sh
git stash
git pull

sbatch scripts/idun/idun.sh
