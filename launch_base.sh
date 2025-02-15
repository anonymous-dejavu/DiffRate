    # --data-set CHARADES \
# python main.py \
python -m torch.distributed.launch \
    --nproc_per_node=4 --use_env --master_port 29513 \
    main.py \
        --arch-lr 0.01 --arch-min-lr 0.001 \
        --epoch 20 \
        --data-set CHARADES \
        --data-path /mnt/ssd2/dataset \
        --output_dir /tmp \
        --batch-size 128 \
        --model vit_base_patch16_clip_224.openai \
        --alpha 10000
