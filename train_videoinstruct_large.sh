#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 TARGET_FLOPS{31.0,34.7,38.5,42.3,46.1}"
    exit 1
fi

# batch-size of 128 results in 22GB memory usage on RTX 3090
python \
    -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29504 \
    main.py \
        --arch-lr 0.01 --arch-min-lr 0.001 \
       	--autoresume \
       	--dist-eval \
        --data-set VIDEOINSTRUCT \
        --data-path /dev/null \
        --output_dir /mnt/ssd3/diffrate/videoinstruct/$1 \
        --batch-size 32 \
        --model vit_large_patch14_clip_224.openai \
        --alpha 1 \
        --train-sampling-rate 0.05 \
        --test-sampling-rate 0.01 \
        --target_flops $1 # {31.0,34.7,38.5,42.3,46.1}
