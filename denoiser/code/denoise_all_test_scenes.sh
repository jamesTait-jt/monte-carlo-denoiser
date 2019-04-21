cd custom_ops
./compile.sh
cd ..


# ========= MAE ========= #
MODEL_DIR="../experiments/models/mae"
NET_TYPE="mae"
#python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 2 $NET_TYPE

MODEL_DIR="../experiments/models/mae_bn"
NET_TYPE="mae_bn"
#python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 2 $NET_TYPE

MODEL_DIR="../experiments/models/mae_albdiv"
NET_TYPE="mae_albdiv"
#python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 2 $NET_TYPE

MODEL_DIR="../experiments/models/mae_albdiv_bn"
NET_TYPE="mae_albdiv_bn"
#python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 2 $NET_TYPE


# ========= VGG ========= #
MODEL_DIR="../experiments/models/vgg"
NET_TYPE="vgg"
python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
python denoise_full_img_at_once.py $MODEL_DIR 2 $NET_TYPE

MODEL_DIR="../experiments/models/vgg_bn"
NET_TYPE="vgg_bn"
python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
python denoise_full_img_at_once.py $MODEL_DIR 2 $NET_TYPE

MODEL_DIR="../experiments/models/vgg_albdiv"
NET_TYPE="vgg_albdiv"
#python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 2 $NET_TYPE

MODEL_DIR="../experiments/models/vgg_albdiv_bn"
NET_TYPE="vgg_albdiv_bn"
#python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 2 $NET_TYPE

