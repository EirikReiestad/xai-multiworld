#!/bin/bash

TEMP=$(getopt -o hecp: --long help,example,concept-score,pipeline -- "$@")
eval set -- "$TEMP"

help_flag=false
example_flag=false
concept_score_flag=false
pipeline_score_flag=false

while true; do
    case "$1" in
    -h | --help)
        help_flag=true
        shift
        ;;
    -e | --example)
        example_flag=true
        shift
        ;;
    -c | --concept-score)
        concept_score_flag=true
        shift
        break
        ;;
    -p | --pipeline)
        pipeline_score_flag=true
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

if [ "$example_flag" = true ]; then
    sbatch scripts/idun/example.sh
    exit 0
fi

if [ "$pipeline_score_flag" = true ]; then
    sbatch scripts/idun/pipeline.sh
    exit 0
fi

if [ "$concept_score_flag" = true ]; then
    sbatch scripts/idun/xailib.sh "--concept-score"
    exit 0
fi

if [ "$help_flag" = true ] || true; then
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --help           Show this help message"
    echo "  --example        Run the example script"
    echo "  --pipeline       Run the pipeline"
    echo "  --xailib         Run the xailib script with additional arguments"
    exit 0
fi
