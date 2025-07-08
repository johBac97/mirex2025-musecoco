#!/bin/bash

# Usage: ./generation.sh input.json output_folder n_sample

input_json="$1"
output_folder="$2"
n_sample="$3"

python3 generate_main.py "$input_json" "$output_folder" "$n_sample"