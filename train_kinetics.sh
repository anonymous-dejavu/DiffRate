python \
    -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29513 \
    main.py \
        --arch-lr 0.01 --arch-min-lr 0.001 \
        --epoch 20 \
        --data-set KINETICS \
        --data-path /mnt/ssd2/dataset \
        --output_dir /workspace/third_parties/DiffRate/output/kinetics \
        --batch-size 64 \
        --model vit_base_patch16_clip_224.openai \
        --alpha 1 \
        --target_flops 8.7 # {8.7,10.0,10.4,11.5}
