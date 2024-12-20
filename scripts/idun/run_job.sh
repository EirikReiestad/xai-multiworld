#!/bin/sh
git stash
git pull

sh scripts/idun/clean.sh

sbatch idun.sh
