#!/bin/sh
git stash
git pull

sh clean.sh

sbatch idun.sh
