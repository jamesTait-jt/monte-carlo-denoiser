MODEL_DIR=$1

python tungsten_eval_full.py $MODEL_DIR 0
python tungsten_eval_full.py $MODEL_DIR 1
python tungsten_eval_full.py $MODEL_DIR 2
