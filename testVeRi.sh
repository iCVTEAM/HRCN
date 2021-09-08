CUDA_VISIBLE_DEVICES=0 python ./tools/train_net.py \
--config-file ./configs/VeRi/hrcn.yml \
--eval-only \
MODEL.WEIGHTS "model_weight/VeRi.pth" \
MODEL.DEVICE "cuda:0" \
OUTPUT_DIR ./logs/veri/test
