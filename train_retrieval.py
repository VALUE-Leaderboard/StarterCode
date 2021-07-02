"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training Retrieval (VCMR/VR) Model

copied/modified from HERO
(https://github.com/linjieli222/HERO
"""
from collections import defaultdict
import os
from os.path import exists, join

from time import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd

from tqdm import tqdm

from data import (
    QueryTokLmdb, MsrvttQueryTokLmdb,
    VrQueryTokLmdb, VrFullEvalDataset, VrVideoOnlyFullEvalDataset,
    VrSubOnlyFullEvalDataset,
    VcmrFullEvalDataset, vcmr_full_eval_collate,
    VcmrVideoOnlyFullEvalDataset, VcmrSubOnlyFullEvalDataset,
    PrefetchLoader, MetaLoader)
from load_data import (
    VideoDbGroup, get_video_ids,
    build_retrieval_qa_dataloader, VFeatDbGroup,
    SubDbGroup)

from model.vcmr import HeroForVcmr
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta, TrainingRestorer
from utils.misc import NoOp, set_dropout, set_random_seed
from utils.const import VFEAT_DIM, MAX_FRM_SEQ_LEN
from utils.basic_utils import save_json
from config.config import shared_configs
from eval_vcmr import validate_full_vcmr
from eval_vr import validate_full_vr


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    opts.n_gpu = n_gpu
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), not opts.no_fp16))

    if hvd.rank() != 0:
        LOGGER.disabled = True

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

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
        query_path = train_data["query_txt_db"]
        batch_size = (
            train_data["batch_size"] if "batch_size" in train_data
            else opts.train_batch_size)
        LOGGER.info(f"Loading train dataset {name}: {sub_path}, "
                    f"{vfeat_path}, {query_path}")

        video_db = all_video_dbs.__getitem__(name, vfeat_path, sub_path, opts)
        name = f"{task}/{name}"
        loader = build_retrieval_qa_dataloader(
            name, video_db, query_path,
            True, opts, batch_size)
        train_dataloaders[name] = (loader, ratio)
    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=opts.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)

    # val
    val_dataloaders = {}
    inf_dataloaders = {}
    for val_data in opts.val_datasets:
        name = val_data["name"]
        task = val_data["task"]
        vfeat_path = val_data["vfeat_db"]
        sub_path = val_data["sub_txt_db"]
        query_path = val_data["query_txt_db"]
        batch_size = (
            val_data["batch_size"] if "batch_size" in val_data
            else opts.val_batch_size)
        LOGGER.info(f"Loading val dataset {name}: {sub_path}, "
                    f"{vfeat_path}, {query_path}")
        video_db = all_video_dbs.__getitem__(name, vfeat_path, sub_path, opts)
        name = f"{task}/{name}"
        loader = build_retrieval_qa_dataloader(
            name, video_db, query_path,
            False, opts, batch_size)
        val_dataloaders[name] = loader

        # load inf data
        if task == "vcmr":
            if "video_only" in name or sub_path is None:
                inf_dataset = VcmrVideoOnlyFullEvalDataset
            elif "sub_only" in name:
                inf_dataset = VcmrSubOnlyFullEvalDataset
            else:
                inf_dataset = VcmrFullEvalDataset
        else:
            if "video_only" in name or sub_path is None:
                inf_dataset = VrVideoOnlyFullEvalDataset
            elif "sub_only" in name:
                inf_dataset = VrSubOnlyFullEvalDataset
            else:
                inf_dataset = VrFullEvalDataset

        video_ids = get_video_ids(query_path)

        if "msrvtt" in name:
            queryLmdb = MsrvttQueryTokLmdb
        elif "yc2r" in name or "vatex_en_r" in name:
            queryLmdb = VrQueryTokLmdb
        else:
            queryLmdb = QueryTokLmdb
        val_q_txt_db = queryLmdb(query_path, -1)
        LOGGER.info(f"Loading Inference Dataset {query_path} (val)")
        val_dset = inf_dataset(
            video_ids, video_db, val_q_txt_db,
            distributed=opts.distributed_eval)
        inf_loader_val = DataLoader(val_dset,
                                    batch_size=opts.vcmr_eval_q_batch_size,
                                    num_workers=opts.n_workers,
                                    pin_memory=not opts.no_pin_mem,
                                    collate_fn=vcmr_full_eval_collate)
        inf_loader_val = PrefetchLoader(inf_loader_val)
        inf_dataloaders[name] = inf_loader_val

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}
    img_pos_embed_weight_key = "v_encoder.f_encoder.img_embeddings" +\
        ".position_embeddings.weight"
    if img_pos_embed_weight_key in checkpoint:
        max_frm_seq_len = len(checkpoint[img_pos_embed_weight_key])
    else:
        max_frm_seq_len = MAX_FRM_SEQ_LEN

    model = HeroForVcmr.from_pretrained(
            opts.model_config,
            state_dict=checkpoint,
            vfeat_dim=VFEAT_DIM[opts.vfeat_version],
            max_frm_seq_len=max_frm_seq_len,
            lw_neg_ctx=opts.lw_neg_ctx,
            lw_neg_q=opts.lw_neg_q, lw_st_ed=0,
            ranking_loss_type=opts.ranking_loss_type,
            use_hard_negative=False,
            hard_pool_size=opts.hard_pool_size,
            margin=opts.margin,
            use_all_neg=opts.use_all_neg,
            drop_svmr_prob=opts.drop_svmr_prob)
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
    if hvd.rank() == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        if not exists(join(opts.output_dir, 'results')):
            # store tvr predictions
            os.makedirs(join(opts.output_dir, 'results'))
        if opts.nms_thd != -1:
            # store tvr-nms predictions
            if not exists(join(opts.output_dir, 'results_nms')):
                os.makedirs(join(opts.output_dir, 'results_nms'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
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
    for key in train_dataloaders.keys():
        for obj in (f'{key}_st_ed', f'{key}_neg_ctx',
                    f'{key}_neg_q'):
            task2loss[obj] = RunningMeter(f'loss/{obj}')

    model.train()
    n_examples = defaultdict(int)
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    if global_step == 0:
        optimizer.step()
    for step, (task, batch) in enumerate(meta_loader):
        if len(opts.hard_negative_start_step) > 0:
            for i, hn_step in enumerate(opts.hard_negative_start_step):
                if global_step >= hn_step and hn_step != -1:
                    model.set_hard_negative(
                        True, opts.hard_pool_size[i], opts.hard_neg_weights[i])
        if opts.train_span_start_step != -1 and\
                global_step >= opts.train_span_start_step:
            model.set_train_st_ed(opts.lw_st_ed)

        n_examples[task] += opts.train_batch_size
        train_task = task.split("/")[0]
        loss = model(batch, task=train_task, compute_loss=True)

        loss_st_ed, loss_neg_ctx, loss_neg_q = loss
        loss = loss_st_ed + loss_neg_ctx + loss_neg_q
        for n, ls, w in (('st_ed', loss_st_ed, opts.lw_st_ed),
                         ('neg_ctx', loss_neg_ctx, opts.lw_neg_ctx),
                         ('neg_q', loss_neg_q, opts.lw_neg_q)):
            ls = ls.item()
            if w:
                ls /= w
            task2loss[f'{task}_{n}'](ls)

        loss = loss.mean()
        task2loss[task](loss.item())

        delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
        with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale,
                            loss_id=task2scaler[task]) as scaled_loss:
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
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
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
                for t in train_dataloaders.keys():
                    tot_ex = sum(all_gather_list(n_examples[t]))
                    ex_per_sec = int(tot_ex / (time()-start))
                    LOGGER.info(f'{t}: {tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar(f'perf/{t}_ex_per_s', ex_per_sec,
                                         global_step)

            if global_step % opts.valid_steps == 0:
                LOGGER.info('===========================================')
                LOGGER.info(f"Step {global_step}: start running validation")
                validate(model, val_dataloaders, opts)
                if hvd.rank() == 0 or opts.distributed_eval:
                    full_validate(model, inf_dataloaders, opts, global_step)
                LOGGER.info('===========================================')
                model_saver.save(model, global_step)

            # step restorer in the end to prevent missing validation checkpoint
            restorer.step()
        if global_step >= opts.num_train_steps:
            break

    LOGGER.info('===========================================')
    if global_step % opts.valid_steps != 0 or opts.num_train_steps == 0:
        if hvd.rank() == 0 or opts.distributed_eval:
            full_validate(model, inf_dataloaders, opts, global_step)
    model_saver.save(model, f'{global_step}')


@torch.no_grad()
def validate(model, val_dataloaders, opts):
    for name, loader in val_dataloaders.items():
        model.eval()
        LOGGER.info(f"validate on {name} task")
        dset_name = name.split("/")[-1]
        if "vcmr/" in name:
            val_log = validate_vcmr(model, loader, opts)
        else:
            val_log = validate_vr(model, loader, opts)
        TB_LOGGER.log_scaler_dict(
            {f'valid_{dset_name}/{k}': v for k, v in val_log.items()})
    model.train()


@torch.no_grad()
def full_validate(model, inf_dataloaders, opts, global_step):
    meta_ave = 0
    for name, inf_loader_val in inf_dataloaders.items():
        LOGGER.info(f"full inference on {name} task")
        model.eval()
        dset_name = name.split("/")[-1]
        if "vcmr" in name:
            log, results = validate_full_vcmr(
                    model, inf_loader_val,
                    "val", opts, model_opts=opts,
                    task=dset_name)
            aveR = (
                log["valid_val_VCMR/VCMR_0.7-r1"] +
                log["valid_val_VCMR/VCMR_0.7-r10"] +
                log["valid_val_VCMR/VCMR_0.7-r5"]) / 3.0
        else:
            log, results = validate_full_vr(
                model, inf_loader_val,
                'val', opts, model_opts=opts,
                task=dset_name)
            aveR = (
                log["valid_val_VR/VR_r1"] +
                log["valid_val_VR/VR_r5"] +
                log["valid_val_VR/VR_r10"]) / 3.0
        save_json(
            results, f'{opts.output_dir}/results/'
            f'{dset_name}_val_results_{global_step}'
            f'_rank{hvd.rank()}.json')
        log = {f'{dset_name}_{k}': v for k, v in log.items()}
        log[f'{dset_name}_aveR'] = aveR

        LOGGER.info(f'{dset_name}_aveR: {aveR:.2f}')
        TB_LOGGER.log_scaler_dict(log)
        meta_ave += aveR

    if len(inf_dataloaders) > 1:
        meta_ave = meta_ave/len(inf_dataloaders)
        LOGGER.info(f'Meta-Ave: {meta_ave:.2f}')
        TB_LOGGER.log_scaler_dict({'valid_meta_ave': meta_ave})
    model.train()


@torch.no_grad()
def validate_vr(model, val_loader, opts):
    LOGGER.info(
        "start running validation (easy version with loss computed)...")
    val_loss = 0
    val_loss_neg_ctx = 0
    val_loss_neg_q = 0
    n_ex = 0
    n_ex_pos = 0
    st = time()
    model.eval()

    for i, batch in enumerate(val_loader):
        if 'qids' in batch:
            # qids = batch['qids']
            del batch['qids']
        n_ex += len(batch['q_vidx'])

        _, loss_neg_ctx, loss_neg_q =\
            model(batch, "vr", compute_loss=True)

        if opts.lw_neg_ctx != 0 or opts.lw_neg_q != 0:
            n_pos = len(loss_neg_ctx)
            val_loss_neg_ctx += loss_neg_ctx.sum().item()
            val_loss_neg_q += loss_neg_q.sum().item()
            n_ex_pos += n_pos

    val_loss_neg_ctx = sum(all_gather_list(val_loss_neg_ctx))
    val_loss_neg_q = sum(all_gather_list(val_loss_neg_q))
    n_ex = sum(all_gather_list(n_ex))
    n_ex_pos = sum(all_gather_list(n_ex_pos))
    tot_time = time()-st
    if n_ex_pos > 0 and opts.lw_neg_q > 0 and\
            opts.lw_neg_ctx > 0:
        val_loss_neg_ctx /= n_ex_pos
        val_loss_neg_q /= n_ex_pos
        val_loss_neg_ctx /= opts.lw_neg_ctx
        val_loss_neg_q /= opts.lw_neg_q

    val_loss = opts.lw_neg_ctx * val_loss_neg_ctx +\
        opts.lw_neg_q * val_loss_neg_q
    val_log = {
        'valid/loss_overall': val_loss,
        'valid/loss_neg_ctx': val_loss_neg_ctx,
        'valid/loss_neg_q': val_loss_neg_q,
        'valid/ex_per_s': n_ex/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"loss: {val_loss:.2f}")
    model.train()
    return val_log


@torch.no_grad()
def validate_vcmr(model, val_loader, opts):
    LOGGER.info(
        "start running validation (easy version with loss computed)...")
    val_loss = 0
    val_loss_st_ed = 0
    val_loss_neg_ctx = 0
    val_loss_neg_q = 0
    n_ex = 0
    n_ex_pos = 0
    st = time()
    model.eval()

    for i, batch in enumerate(val_loader):
        if 'qids' in batch:
            # qids = batch['qids']
            del batch['qids']
        n_ex += len(batch['q_vidx'])

        loss_st_ed, loss_neg_ctx, loss_neg_q =\
            model(batch, "vcmr", compute_loss=True)

        val_loss_st_ed += loss_st_ed.item()
        if opts.lw_neg_ctx != 0 or opts.lw_neg_q != 0:
            assert loss_neg_ctx.dim(), f"loss_neg_ctx with dim {loss_neg_ctx.dim()}"
            n_pos = len(loss_neg_ctx)
            val_loss_neg_ctx += loss_neg_ctx.sum().item()
            val_loss_neg_q += loss_neg_q.sum().item()
            n_ex_pos += n_pos

    val_loss_st_ed = sum(all_gather_list(val_loss_st_ed))
    val_loss_neg_ctx = sum(all_gather_list(val_loss_neg_ctx))
    val_loss_neg_q = sum(all_gather_list(val_loss_neg_q))
    n_ex = sum(all_gather_list(n_ex))
    n_ex_pos = sum(all_gather_list(n_ex_pos))
    tot_time = time()-st
    if opts.lw_st_ed:
        val_loss_st_ed /= n_ex
        val_loss_st_ed /= opts.lw_st_ed
    if n_ex_pos > 0 and opts.lw_neg_q > 0 and\
            opts.lw_neg_ctx > 0:
        val_loss_neg_ctx /= n_ex_pos
        val_loss_neg_q /= n_ex_pos
        val_loss_neg_ctx /= opts.lw_neg_ctx
        val_loss_neg_q /= opts.lw_neg_q

    val_loss = opts.lw_st_ed * val_loss_st_ed +\
        opts.lw_neg_ctx * val_loss_neg_ctx +\
        opts.lw_neg_q * val_loss_neg_q
    val_log = {
        'valid/loss_overall': val_loss,
        'valid/loss_st_ed': val_loss_st_ed,
        'valid/loss_neg_ctx': val_loss_neg_ctx,
        'valid/loss_neg_q': val_loss_neg_q,
        'valid/ex_per_s': n_ex/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"loss: {val_loss:.2f}")
    model.train()
    return val_log


if __name__ == "__main__":
    args = shared_configs.get_vcmr_args()
    main(args)
