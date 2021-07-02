"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training VideoCaption Model

copied/modified from HERO
(https://github.com/linjieli222/HERO
"""
import os
from os.path import join
from time import time

import torch
from torch.nn.utils import clip_grad_norm_

from apex import amp
from horovod import torch as hvd
from transformers import RobertaTokenizer

from tqdm import tqdm

from data import (PrefetchLoader, MetaLoader)
from model.videoCap import HeroForVideoCap, VideoCapGenerator
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta, TrainingRestorer
from utils.misc import NoOp, set_dropout, set_random_seed
from utils.const import VFEAT_DIM, MAX_FRM_SEQ_LEN
from utils.basic_utils import save_jsonl

from eval.tvc import TvcEval
from eval.vatex_en_c import Vatex_en_c_Eval
from eval.yc2c import Yc2cEvalMicro
from config.config import shared_configs

from load_data import (
    build_caption_dataloader, VFeatDbGroup,
    SubDbGroup, VideoDbGroup)


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), not opts.no_fp16))

    if hvd.rank() != 0:
        LOGGER.disabled = True

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)
    all_vfeat_dbs = VFeatDbGroup(
        opts.vfeat_interval, opts.vfeat_version,
        opts.compressed_db, opts.max_clip_len)
    all_sub_dbs = SubDbGroup(opts.max_clip_len)
    all_video_dbs = VideoDbGroup(all_vfeat_dbs, all_sub_dbs)

    train_dataloaders = {}
    for train_data in opts.train_datasets:
        name = train_data["name"]
        task = train_data["task"]
        ratio = train_data["ratio"] if "ratio" in train_data else 1
        vfeat_path = train_data["vfeat_db"]
        sub_path = train_data["sub_txt_db"]
        cap_path = train_data["cap_txt_db"]
        batch_size = (
            train_data["batch_size"] if "batch_size" in train_data
            else opts.train_batch_size)
        LOGGER.info(f"Loading train dataset {name}: {sub_path}, "
                    f"{vfeat_path}, {cap_path}")
        video_db = all_video_dbs.__getitem__(name, vfeat_path, sub_path, opts)

        name = f"{task}/{name}"
        train_loader = build_caption_dataloader(
            name, video_db, cap_path, True, opts, batch_size)
        train_dataloaders[name] = (train_loader, ratio)
    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=opts.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)

    # val
    val_dataloaders = {}
    for val_data in opts.val_datasets:
        name = val_data["name"]
        task = val_data["task"]
        vfeat_path = val_data["vfeat_db"]
        sub_path = val_data["sub_txt_db"]
        val_ref = val_data["gt_anno"]
        batch_size = (
            val_data["batch_size"] if "batch_size" in val_data
            else opts.val_batch_size)
        LOGGER.info(f"Loading val dataset {name}: {sub_path}, "
                    f"{vfeat_path}, {val_ref}")
        video_db = all_video_dbs.__getitem__(name, vfeat_path, sub_path, opts)
        name = f"{task}/{name}"

        val_loader = build_caption_dataloader(
            name, video_db, val_ref, False, opts, batch_size)
        val_dataloaders[name] = val_loader

    # Prepare model
    if opts.checkpoint and not opts.load_partial_pretrained:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}

    img_pos_embed_weight_key = "v_encoder.f_encoder.img_embeddings" +\
        ".position_embeddings.weight"
    if img_pos_embed_weight_key in checkpoint:
        max_frm_seq_len = len(checkpoint[img_pos_embed_weight_key])
    else:
        max_frm_seq_len = MAX_FRM_SEQ_LEN

    # QUICK FIX
    one_hot_weight = 'caption_loss_func.one_hot'
    if len(checkpoint) and one_hot_weight in checkpoint:
        new_key = 'loss_func.one_hot'
        checkpoint[new_key] = checkpoint[one_hot_weight]

    model = HeroForVideoCap.from_pretrained(
        opts.model_config,
        state_dict=checkpoint,
        vfeat_dim=VFEAT_DIM[opts.vfeat_version],
        max_frm_seq_len=max_frm_seq_len,
        lsr=opts.lsr)
    if opts.checkpoint and opts.load_partial_pretrained:
        partial_checkpoint = torch.load(opts.checkpoint)
        model.load_partial_pretrained(
            partial_checkpoint, VFEAT_DIM[opts.vfeat_version],
            max_frm_seq_len,
            skip_layers=opts.skip_layer_loading)

    model.to(device)
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}
    model, optimizer = amp.initialize(
        model, optimizer, num_losses=len(task2scaler),
        enabled=not opts.no_fp16, opt_level='O2')
    restorer = TrainingRestorer(opts, model, optimizer)
    global_step = restorer.global_step
    TB_LOGGER.global_step = global_step

    # assumes roberta tokenizer only
    if hvd.local_rank() == 0:
        # quick hack to prevent multi-process download collision
        toker = RobertaTokenizer.from_pretrained('roberta-base')
        all_gather_list(None)
    else:
        all_gather_list(None)
        toker = RobertaTokenizer.from_pretrained('roberta-base')
    bos = toker.convert_tokens_to_ids(['<s>'])[0]
    eos = toker.convert_tokens_to_ids(['</s>'])[0]
    generator = VideoCapGenerator(
        model, opts.max_gen_step, bos, eos, not opts.no_fp16)

    global_step = 0
    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        os.makedirs(join(opts.output_dir, 'results'))  # store val predictions
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()
        restorer = NoOp()

    if global_step > 0:
        pbar.update(global_step)

    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    task2loss = {task: RunningMeter(f'loss/{task}')
                 for task in train_dataloaders.keys()}
    n_vid = 0
    n_cap = 0
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    if global_step == 0:
        optimizer.step()
    model.train()
    while True:
        for step, (task, batch) in enumerate(meta_loader):
            n_vid += opts.train_batch_size
            n_cap += batch['cap_input_ids'].size(0)

            loss = model(batch, compute_loss=True)
            loss = loss.mean()
            task2loss[task](loss.item())

            delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale
                                ) as scaled_loss:
                scaled_loss.backward()
                if not delay_unscale:
                    # gather gradients from every processes
                    # do this before unscaling to make sure every process uses
                    # the same gradient scale
                    grads = [p.grad.data for p in model.parameters()
                             if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))

            if (step + 1) % opts.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, opts)
                for i, param_group in enumerate(optimizer.param_groups):
                    if i == 0 or i == 1:
                        param_group['lr'] = lr_this_step * opts.lr_mul
                    elif i == 2 or i == 3:
                        param_group['lr'] = lr_this_step
                    else:
                        raise ValueError()
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                TB_LOGGER.log_scaler_dict({temp_loss.name: temp_loss.val
                                           for temp_loss in task2loss.values()
                                           if temp_loss.val is not None})
                TB_LOGGER.step()

                # update model params
                if opts.grad_norm != -1:
                    grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

                if global_step % 100 == 0:
                    # monitor training throughput
                    LOGGER.info('-------------------------------------------')
                    LOGGER.info(f'Step {global_step}:')
                    tot_vid = sum(all_gather_list(n_vid))
                    vid_per_sec = int(tot_vid / (time()-start))
                    LOGGER.info(f'{tot_vid} videos trained at '
                                f'{vid_per_sec} vid/s')
                    tot_cap = sum(all_gather_list(n_cap))
                    cap_per_sec = int(tot_cap / (time()-start))
                    TB_LOGGER.add_scalar('perf/vid_per_s', vid_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar('perf/cap_per_s', cap_per_sec,
                                         global_step)

                if global_step % opts.valid_steps == 0:
                    LOGGER.info('===========================================')
                    LOGGER.info(f"Step {global_step}: start validation")
                    validate(
                        val_dataloaders,
                        generator, toker, opts, global_step)
                    LOGGER.info('===========================================')
                    model_saver.save(model, global_step)

                # step restorer in the end
                # to prevent missing validation checkpoint
                restorer.step()
            if global_step >= opts.num_train_steps:
                break

        if global_step >= opts.num_train_steps:
            break

    LOGGER.info('===========================================')
    if global_step % opts.valid_steps != 0 or opts.num_train_steps == 0:
        validate(
            val_dataloaders,
            generator, toker, opts, global_step)
        model_saver.save(model, global_step)


@torch.no_grad()
def validate(val_dataloaders, generator, tokenizer, opts, global_step=0):
    meta_ave = 0
    for name, val_loader in val_dataloaders.items():
        val_ref = val_loader.dataset.gt_anno_path
        LOGGER.info(f"Validation for task/dataset: {name} with {val_ref}")
        dset_name = name.split("/")[-1]
        if hvd.rank() == 0:
            if "tvc" in dset_name:
                evaluator = TvcEval(val_ref)
            elif 'yc2' in dset_name:
                evaluator = Yc2cEvalMicro(val_ref)
            elif 'vatex' in dset_name:
                evaluator = Vatex_en_c_Eval(val_ref)
            else:
                raise ValueError
        else:
            evaluator = NoOp()
        val_log, results = validate_single_dset(
            val_loader, generator, tokenizer, evaluator)
        if hvd.rank() == 0:
            meta_ave += val_log["CIDEr"]
            save_jsonl(results, f"{opts.output_dir}/results/"
                                f"/{dset_name}_results_{global_step}.jsonl")
        TB_LOGGER.log_scaler_dict(
            {f'{dset_name}/{k}': v for k, v in val_log.items()})
    if len(val_dataloaders) > 1:
        meta_ave = meta_ave/len(val_dataloaders)
        LOGGER.info(f'Meta-Ave: {meta_ave:.2f}')
        TB_LOGGER.add_scalar('valid_meta_ave', meta_ave, global_step)


@torch.no_grad()
def validate_single_dset(loader, generator, tokenizer, evaluator):
    st = time()
    generator.model.eval()
    results = []
    for batch in loader:
        vids = batch['vid_names']
        cids = batch['clip_ids']
        if 'all_ts' in batch:
            all_ts = batch['all_ts']
        else:
            all_ts = None
        outputs = generator.greedy_decode(batch)
        for temp_ind, (vid, cid, out_ids) in enumerate(
                zip(vids, cids, outputs)):
            output = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(out_ids))
            if all_ts is not None:
                ts = all_ts[temp_ind]
                results.append({
                    'vid_name': vid, 'clip_id': cid, 'ts': ts,
                    'descs': [{'desc': output}]})
            else:
                results.append({'vid_name': vid, 'clip_id': int(cid),
                                'descs': [{'desc': output}]})

    results = [r for rs in all_gather_list(results) for r in rs]
    LOGGER.info(f'decoding finished in {int(time() - st)} seconds')
    if hvd.rank() == 0:
        val_log = evaluator(results)
        LOGGER.info(f'Validation finished in {int(time() - st)} seconds')
        LOGGER.info(
            f'B@4: {val_log["Bleu_4"]:.2f}, R: {val_log["ROUGE_L"]:.2f}, '
            f'M: {val_log["METEOR"]:.2f}, C: {val_log["CIDEr"]:.2f}')
    else:
        val_log = {}
    generator.model.train()
    return val_log, results


if __name__ == "__main__":
    args = shared_configs.get_videoCap_args()

    main(args)
