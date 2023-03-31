TRAIN_FLAGS="--iterations 500000 --anneal_lr True --batch_size 32 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
mpiexec -n 8 python scripts/classifier_train.py --data_dir IMAGENET_PATH $TRAIN_FLAGS $CLASSIFIER_FLAGS
