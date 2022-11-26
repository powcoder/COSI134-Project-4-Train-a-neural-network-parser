https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
#!/bin/bash
set -eo pipefail

DATA_DIR=$1
MODEL_DIR=$2

if [ -z "$1" ] && [ -z "$2" ]; then
  echo "Data dir and Model dir are required arguments"
  exit 1
fi

params=${@:2}

echo ' ### Prepare Data ###' >&2
python src/prepare_data.py \
  --out_dir $DATA_DIR \
  $params

echo ' ### Train ###' >&2
python src/train.py \
  --data_dir $DATA_DIR \
  --model_dir $MODEL_DIR \
  $params

echo ' ### Inference ###' >&2
python src/predict.py \
  --model_dir $MODEL_DIR \
  $params
