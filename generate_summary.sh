#!/bin/bash
input_dir_name="$1"

python extract.py --input_dir "$input_dir_name" --model_name "$model_name"
python summarize.py --input_dir "$input_dir_name"


