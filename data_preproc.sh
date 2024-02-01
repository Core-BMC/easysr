#!/bin/bash

# data_preproc.sh

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input) input="$2"; shift ;;
        -t1|--t1) t1_flag="--t1" ;;
        -t2|--t2) t2_flag="--t2" ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Check for required arguments
if [ -z "$input" ]; then
    echo "Input directory is required. Use -i or --input to specify it."
    exit 1
fi

# Setup output directory
output="./train_data"

# Run MRI pre-processing
echo "Running MRI preprocessing..."
python mri_preproc.py --input "$input" $t1_flag $t2_flag --output "$output"
echo "MRI preprocessing completed."

# Split data into training and validation sets
echo "Splitting data into training and validation sets..."
python ./utils/data_splitter.py --input "$output" --train-ratio 0.8 --train-folder "training" --val-folder "validation"
echo "Data splitting completed."
