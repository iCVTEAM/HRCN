python ./tools/train_net.py \
--config-file ./configs/VERIWild/hrcn.yml \
--eval-only \
MODEL.WEIGHTS "model_weight/VERiWild.pth" \
MODEL.DEVICE "cuda:0" \
OUTPUT_DIR ./logs/VERIWild/test