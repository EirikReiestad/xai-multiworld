#!/bin/bash

TEMP=$(getopt -o '' --long help,example,concept-score -- "$@")
eval set -- "$TEMP"

help_flag=false
example_flag=false
concept_score_flag=false

while true; do
    case "$1" in
    --help)
        help_flag=true
        shift
        ;;
    --example)
        example_flag=true
        shift
        ;;
    --concept-score)
        concept_score_flag=true
        shift
        break
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

if [ "$help_flag" = true ]; then
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --help           Show this help message"
    echo "  --example        Run the example script"
    echo "  --xailib         Run the xailib script with additional arguments"
    exit 0
fi

if [ "$example_flag" = true ]; then
    sh scripts/idun/example.sh
    sbatch scripts/idun/example.sh
fi

if [ "$concept_score_flag" = true ]; then
    sh scripts/idun/xailib.sh "--concept-score"
    sbatch scripts/idun/xailib.sh "--concept-score"
fi
