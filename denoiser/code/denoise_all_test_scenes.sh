MODEL_DIR=$1
NET_TYPE=$2

python tungsten_eval_full.py $MODEL_DIR 0 $NET_TYPE
python tungsten_eval_full.py $MODEL_DIR 1 $NET_TYPE
python tungsten_eval_full.py $MODEL_DIR 2 $NET_TYPE
