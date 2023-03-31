CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 32 --num_samples 50000 --diffusion_steps 1000 --timestep_respacing 250"
OPENAI_LOGDIR="LOG_PATH" mpiexec -n 10 python scripts/classifier_sample.py --classifier_scale 1.0 \
    --model_path DOWNLOAD_MODEL_PATH \
    --classifier_path TRAINED_ENERGY_GUIDANCE_PATH \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS