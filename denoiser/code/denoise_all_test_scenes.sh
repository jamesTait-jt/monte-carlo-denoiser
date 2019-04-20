cd custom_ops
./compile.sh
cd ..

# ======= VANILLA MAE, VGG, WGAN-GP NO ALBDIV ====== #
MODEL_DIR="../experiments/models/mae"
NET_TYPE="mae"
python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
python denoise_full_img_at_once.py $MODEL_DIR 2 $NET_TYPE

MODEL_DIR="../experiments/models/vgg"
NET_TYPE="vgg"
python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
python denoise_full_img_at_once.py $MODEL_DIR 2 $NET_TYPE

MODEL_DIR="../experiments/models/wgan-gp"
NET_TYPE="wgan-gp"
python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
python denoise_full_img_at_once.py $MODEL_DIR 2 $NET_TYPE


# ======= VANILLA MAE, VGG WITH ALBDIV AND RMSPROP ====== #
MODEL_DIR="../experiments/models/mae-albdiv-rmsprop"
NET_TYPE="mae-albdiv-rmsprop"
#python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 2 $NET_TYPE

MODEL_DIR="../experiments/models/vgg-albdiv-rmsprop"
NET_TYPE="vgg-albdiv-rmsprop"
#python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE


# ======= VANILLA MAE, VGG, WGAN-GP WITH ALBDIV ====== #
MODEL_DIR="../experiments/models/mae-albdiv"
NET_TYPE="mae-albdiv"
#python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 2 $NET_TYPE

MODEL_DIR="../experiments/models/vgg-albdiv"
NET_TYPE="vgg-albdiv"
#python denoise_full_img_at_once.py $MODEL_DIR 1 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE

MODEL_DIR="../experiments/models/wgan-gp-albdiv"
NET_TYPE="mae-albdiv"
#python denoise_full_img_at_once.py $MODEL_DIR 2 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE
#python denoise_full_img_at_once.py $MODEL_DIR 0 $NET_TYPE

