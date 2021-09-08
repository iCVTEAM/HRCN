CUDA_VISIBLE_DEVICES=0 python ./tools/train_net.py \
--config-file ./configs/VehicleID/hrcn.yml \
MODEL.DEVICE "cuda:0" \
OUTPUT_DIR ./logs/vehicleid/hrcn