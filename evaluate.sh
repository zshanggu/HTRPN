#python3.6 tools/test_net.py --num-gpus 0 --config-file configs/COCO/R-101/5shot_baseline.yml --eval-only
# python3.6 tools/test_net.py --num-gpus 4 --config-file configs/PASCAL_VOC/split1/5shot_CL_IoU.yml --eval-only
python3.6 tools/test_net.py --num-gpus 4 --config-file configs/PASCAL_VOC/base-training/R101_FPN_base_training_split1.yml --eval-only
