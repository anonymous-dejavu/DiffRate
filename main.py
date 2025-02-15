# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from dataset import build_dataset
from engine import train_one_epoch, evaluate
from samplers import RASampler
import utils
import shutil
import warnings
from utils import MultiEpochsDataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler

import models_mae
import caformer
import DiffRate


warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('Diffrate training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_clip_224.openai', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--multi-reso', default=False, action='store_true',help='')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.00,
                        help='weight decay (default: 0.00)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--arch-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--arch-min-lr', type=float, default=0.001, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (0.001)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument(
        '--data-set',
        default='IMNET',
        choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'CHARADES', 'HOW2QA', 'KINETICS', 'HMDB51', 'MSRVTT', 'VIDEOINSTRUCT'],
        type=str,
        help='Image Net dataset path'
    )
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./log/temp',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--autoresume', action='store_true', help='auto resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--port', default="15662", type=str,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--target_flops', type=float, default=3.0)
    parser.add_argument('--granularity', type=int, default=4, help='the token number gap between each compression rate candidate')
    parser.add_argument('--load_compression_rate', action='store_true', help='eval by exiting compression rate in compression_rate.json')
    parser.add_argument('--warmup_compression_rate', action='store_true', default=False, help='inactive computational constraint in first epoch')
    parser.add_argument('--alpha', type=int, default=5_000, help='parameter to weight cosine similarity loss')
    parser.add_argument('--train-sampling-rate', type=float, default=0.1, help='sampling rate for training data')
    parser.add_argument('--test-sampling-rate', type=float, default=0.1, help='sampling rate for testing data')
    return parser


def main(args):
    # utils.setup_default_logging()
    utils.init_distributed_mode(args)

    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir,dist_rank=utils.get_rank())
    logger.info(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # Do uniform sampling
    indices = list(range(0, len(dataset_train), int(1/args.train_sampling_rate)))
    dataset_train = torch.utils.data.Subset(dataset_train, indices=indices)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )

        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                logger.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    num_samples = int(args.test_sampling_rate * len(dataset_val))
    sampler_val = torch.utils.data.RandomSampler(dataset_val, replacement=True, num_samples=num_samples)

    # leveraging MultiEpochsDataLoader for faster data loading
    data_loader_train = MultiEpochsDataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,

    )

    data_loader_val = MultiEpochsDataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    # FIXME: Figure out what this is doing
    if 'clip' in args.model:
        mixup_active = False
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)


    logger.info(f"Creating model: {args.model}")
    if args.model.endswith('.openai'):
        class QuickGELU(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x * torch.sigmoid(1.702 * x)
        kwargs = {'act_layer': QuickGELU}
    else:
        kwargs = {}


    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        **kwargs
    )

    if args.data_set =="MSRVTT": 
        def _convert_openai_clip(state_dict, model, prefix='vision_model.'):
            out_dict = {}
            swaps = [
                ('embeddings.patch_embedding', 'patch_embed.proj'),
                ('embeddings.position_embedding.weight', 'pos_embed'),
                ('embeddings.class_embedding', 'cls_token'),
                ('self_attn', 'attn'),
                ('encoder.layers.', 'blocks.'),
                ('pre_layrnorm', 'norm_pre'),
                ('post_layernorm', 'norm'),
                # ('ln_', 'norm'),
                ('layer_norm1', 'norm1'),
                ('layer_norm2', 'norm2'),
                # ('in_proj_', 'qkv.'),
                ('out_proj', 'proj'),
                ('visual_projection', 'head'),
                # ('mlp.c_fc', 'mlp.fc1'),
                # ('mlp.c_proj', 'mlp.fc2'),
            ]
            k_proj_weights = []
            k_proj_biases = []
            v_proj_weights = []
            v_proj_biases = []
            q_proj_weights = []
            q_proj_biases = []
            
            for k, v in state_dict.items():
                if not k.startswith(prefix) and not "visual_projection" in k:
                    continue
                k = k.replace(prefix, '')
                for sp in swaps:
                    k = k.replace(sp[0], sp[1])
                
                if k == "head.weight":
                    out_dict['head.bias'] = torch.zeros(v.shape[0])
                elif k == 'cls_token':
                    v = v.unsqueeze(0).unsqueeze(1)
                elif k == 'pos_embed':
                    v = v.unsqueeze(0)
                    if v.shape[1] != model.pos_embed.shape[1]:
                        # To resize pos embedding when using model at different size from pretrained weights
                        v = resize_pos_embed(
                            v,
                            model.pos_embed,
                            0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
                            model.patch_embed.grid_size
                        )
                elif "attn" in k:
                    if "k_proj.weight" in k: 
                        k_proj_weights.append((k, v))
                    elif "k_proj.bias" in k:
                        k_proj_biases.append((k, v))
                    elif "q_proj.weight" in k:
                        q_proj_weights.append((k, v))
                    elif "q_proj.bias" in k:
                        q_proj_biases.append((k, v))
                    elif "v_proj.weight" in k:
                        v_proj_weights.append((k, v))
                    elif "v_proj.bias" in k:
                        v_proj_biases.append((k, v))
                out_dict[k] = v
            
            for layeridx in range(len(q_proj_weights)):
                q_w = q_proj_weights[layeridx][1]
                q_b = q_proj_biases[layeridx][1]
                k_w = k_proj_weights[layeridx][1]
                k_b = k_proj_biases[layeridx][1]
                v_w = v_proj_weights[layeridx][1]
                v_b = v_proj_biases[layeridx][1]

                weight_name = f"blocks.{layeridx}.attn.qkv.weight"
                weight_value = torch.cat((q_w, k_w, v_w), dim=0)
                bias_name = f"blocks.{layeridx}.attn.qkv.bias"
                bias_value = torch.cat((q_b, k_b, v_b), dim=0)
                out_dict[weight_name] = weight_value
                out_dict[bias_name] = bias_value
            
            return out_dict
    
        clip4clip_checkpoint = torch.load('/mnt/ssd3/CLIP4Clip/ckpts/ckpt_msrvtt_retrieval_looseType/best_hf_model/pytorch_model.bin')
        
        # converted_dict = clip4clip_checkpoint
        converted_dict = _convert_openai_clip(clip4clip_checkpoint, model)
        ret = model.load_state_dict(converted_dict, strict=False)
    
        assert len(ret.missing_keys) == 0
    
    # DiffRate Patch
    if 'deit' in args.model:
        DiffRate.patch.deit(model, prune_granularity=args.granularity, merge_granularity=args.granularity)
    elif 'mae' in args.model:
        DiffRate.patch.mae(model, prune_granularity=args.granularity, merge_granularity=args.granularity)
    elif 'caformer' in args.model:
        DiffRate.patch.caformer(model, prune_granularity=args.granularity, merge_granularity=args.granularity)
    elif 'clip' in args.model:
        DiffRate.patch.clip(model, prune_granularity=args.granularity, merge_granularity=args.granularity)
    else:
        raise ValueError("only support deit, mae, caformer and clip in this codebase")

    model_name_dict = {
        'vit_deit_tiny_patch16_224':'ViT-T-DeiT',
        'vit_deit_small_patch16_224':'ViT-S-DeiT',
        'vit_deit_base_patch16_224': 'ViT-B-DeiT',
        'vit_base_patch16_mae': 'ViT-B-MAE',
        'vit_large_patch16_mae': 'ViT-L-MAE',
        'vit_huge_patch14_mae': 'ViT-H-MAE',
        'caformer_s36':'CAFormer-S36',
    }
    if args.load_compression_rate:
        with open('compression_rate.json', 'r') as f:
            compression_rate = json.load(f)
            model_name = model_name_dict[args.model]
            if not str(args.target_flops) in compression_rate[model_name]:
                raise ValueError(f"compression_rate.json does not contaion {model_name} with {args.target_flops}G flops")
            prune_kept_num = eval(compression_rate[model_name][str(args.target_flops)]['prune_kept_num'])
            merge_kept_num = eval(compression_rate[model_name][str(args.target_flops)]['merge_kept_num'])
            model.set_kept_num(prune_kept_num, merge_kept_num)





    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed
        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr


    if args.eval:
        test_stats = evaluate(data_loader_val, model, device,logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['loss']:.1f}%")
        return


    optimizer = torch.optim.AdamW(model_without_ddp.arch_parameters(), lr=args.arch_lr, weight_decay=0)
    loss_scaler = utils.NativeScalerWithGradNormCount()
    lr_scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=args.arch_min_lr, cycle_decay=args.decay_rate)



    if 'clip' in args.model:
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        alpha = args.alpha
        criterion = lambda x, y: (1 - cosine_similarity(x, y).mean()) * alpha
        # mse_loss = torch.nn.MSELoss()
        # alpha = args.alpha
        # criterion = lambda x, y: mse_loss(x, y) * alpha
    elif mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()


    if args.autoresume and os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])


    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer,device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            logger=logger,
            target_flops=args.target_flops,
            warm_up=args.warmup_compression_rate
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device,logger=logger)
        logger.info(f"Loss of the network on the {len(dataset_val)} test images: {test_stats['loss']:.2f}")
        if utils.is_main_process() and min_loss > test_stats['loss'] :
            shutil.copyfile(checkpoint_path, f'{args.output_dir}/model_best.pth')
        min_loss = min(min_loss, test_stats["loss"])
        logger.info(f'Min loss: {min_loss:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLIP training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
