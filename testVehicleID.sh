CUDA_VISIBLE_DEVICES=0 python ./tools/train_net.py \
--config-file ./configs/VehicleID/hrcn.yml \
--eval-only \
MODEL.WEIGHTS "model_weight/VehicleID.pth" \
MODEL.DEVICE "cuda:0" \
OUTPUT_DIR ./logs/vehicleid/test