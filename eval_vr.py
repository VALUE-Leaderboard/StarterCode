"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run evaluation of VR or infenrece for submission
1. VATEX_EN_R
2. YC2R

copied/modified from HERO
(https://github.com/linjieli222/HERO
"""
import argparse
import os
from os.path import exists
from time import time

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pprint
from apex import amp
from horovod import torch as hvd

from data import (VrFullEvalDataset, vr_full_eval_collate,
                  VrVideoOnlyFullEvalDataset,
                  VrSubOnlyFullEvalDataset,
                  PrefetchLoader, MsrvttQueryTokLmdb,
                  VrQueryTokLmdb, video_collate)
from load_data import (
    get_video_ids, load_video_sub_dataset,
    load_video_only_dataset, load_sub_only_dataset)
from data.loader import move_to_cuda
from model.vcmr import HeroForVcmr

from utils.logger import LOGGER
from utils.const import VFEAT_DIM, VCMR_IOU_THDS
from utils.tvr_standalone_eval import eval_retrieval
from utils.distributed import all_gather_list
from utils.misc import Struct
from utils.basic_utils import (
    load_json, save_json)
from utils.tvr_eval_utils import get_results_top_n, format_submission_file


def load_model(model_dir, checkpoint, opts, device):
    hps_file = f'{model_dir}/log/hps.json'
    model_opts = Struct(load_json(hps_file))
    model_config = f'{model_dir}/log/model_config.json'
    # Prepare model
    if exists(checkpoint):
        ckpt_file = checkpoint
    else:
        ckpt_file = f'{model_dir}/ckpt/model_step_{checkpoint}.pt'
    checkpoint = torch.load(ckpt_file)
    img_pos_embed_weight_key = (
        "v_encoder.f_encoder.img_embeddings" +
        ".position_embeddings.weight")
    assert img_pos_embed_weight_key in checkpoint
    max_frm_seq_len = len(checkpoint[img_pos_embed_weight_key])

    model = HeroForVcmr.from_pretrained(
        model_config,
        state_dict=checkpoint,
        vfeat_dim=VFEAT_DIM[model_opts.vfeat_version],
        max_frm_seq_len=max_frm_seq_len,
        lw_neg_ctx=model_opts.lw_neg_ctx,
        lw_neg_q=model_opts.lw_neg_q,
        lw_st_ed=0,
        ranking_loss_type=model_opts.ranking_loss_type,
        use_hard_negative=False,
        hard_pool_size=model_opts.hard_pool_size,
        margin=model_opts.margin,
        use_all_neg=model_opts.use_all_neg)
    model.to(device)
    if not opts.no_fp16:
        model = amp.initialize(model, enabled=True, opt_level='O2')
    return model_opts, model


def load_inf_data(opts, model_opts, mode="video_sub"):
    # load DBs and image dirs
    video_ids = get_video_ids(opts.query_txt_db, opts.vfeat_db)
    if opts.vfeat_interval <= 0:
        vfeat_interval = model_opts.vfeat_interval
    else:
        vfeat_interval = opts.vfeat_interval
    if mode == "video_sub":
        video_db = load_video_sub_dataset(
            opts.vfeat_db, opts.sub_txt_db, model_opts.vfeat_interval,
            model_opts)
    elif mode == "video_only":
        txt_meta = load_json(
            os.path.join(opts.query_txt_db, "meta.json"))
        video_db = load_video_only_dataset(
            opts.vfeat_db, txt_meta,
            vfeat_interval,
            model_opts)
    else:  # sub_only
        video_db = load_sub_only_dataset(
            opts.vfeat_db, opts.sub_txt_db,
            vfeat_interval, model_opts)
    assert opts.split in opts.query_txt_db
    if "msrvtt" in opts.task:
        queryLmdb = MsrvttQueryTokLmdb
    elif "yc2r" in opts.task or "vatex" in opts.task:
        queryLmdb = VrQueryTokLmdb
    else:
        raise ValueError(f"{opts.task} not defined.")
    q_txt_db = queryLmdb(opts.query_txt_db, -1)
    if mode == "video_sub":
        inf_dataset = VrFullEvalDataset
    elif mode == "video_only":
        inf_dataset = VrVideoOnlyFullEvalDataset
    else:  # sub_only
        inf_dataset = VrSubOnlyFullEvalDataset
    eval_dataset = inf_dataset(
        video_ids, video_db, q_txt_db,
        distributed=model_opts.distributed_eval)

    eval_dataloader = DataLoader(eval_dataset, batch_size=opts.batch_size,
                                 num_workers=opts.n_workers,
                                 pin_memory=not opts.no_pin_mem,
                                 collate_fn=vr_full_eval_collate)
    eval_dataloader = PrefetchLoader(eval_dataloader)
    return eval_dataloader


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), not opts.no_fp16))
    if hvd.rank() != 0:
        LOGGER.disabled = True
    model_opts, model = load_model(
        opts.output_dir, opts.checkpoint, opts, device)
    eval_dataloader = load_inf_data(opts, model_opts, mode="video_sub")

    valid_log, results = validate_full_vr(
        model, eval_dataloader, opts.split, opts, model_opts)
    result_dir = f'{opts.output_dir}/results_{opts.task}_{opts.split}'

    if not exists(result_dir) and rank == 0:
        os.makedirs(result_dir)

    all_results_list = all_gather_list(results)
    if hvd.rank() == 0:
        all_results = {
            # quick hack: deleted video2idx due to large prediction file size
            # "video2idx": all_results_list[0]["video2idx"]
            }
        for rank_id in range(hvd.size()):
            for key, val in all_results_list[rank_id].items():
                if key == "video2idx":
                    continue
                if key not in all_results:
                    all_results[key] = []
                all_results[key].extend(all_results_list[rank_id][key])
        LOGGER.info('All results joined......')
        all_results = format_submission_file(all_results)
        save_json(
            all_results,
            f'{result_dir}/results_{opts.checkpoint}_all.json')
        save_json(
            valid_log,
            f'{result_dir}/scores_{opts.checkpoint}.json')
        LOGGER.info('All results written......')


@torch.no_grad()
def validate_full_vr(model, val_loader, split, opts, model_opts, task=None):
    if task is None:
        task = opts.task
    LOGGER.info("start running  full VR evaluation"
                f"on {task} {split} split...")
    if hasattr(model_opts, "max_vr_video"):
        max_vr_video = model_opts.max_vr_video
    else:
        max_vr_video = 100

    model.eval()
    n_ex = 0
    st = time()
    val_log = {}
    has_gt_target = True  # MSRVTT test set has annotations

    video2idx_global = {}
    if hasattr(opts, "vid2dur_idx") and exists(opts.vid2dur_idx):
        # loading from given path
        video_data = load_json(opts.vid2dur_idx)
        video2idx_global = {k: v[1] for k, v in video_data[opts.split].items()}
    else:
        vid2idx = val_loader.dataset.vid2idx
        if split in vid2idx:
            video2idx_global = vid2idx[split]
        else:
            video2idx_global = vid2idx
    assert len(video2idx_global)

    video_ids = sorted(list(video2idx_global.keys()))
    video2idx_local = {e: i for i, e in enumerate(video_ids)}
    query_data = val_loader.dataset.query_data

    partial_query_data = []
    total_frame_embeddings = None
    video_batch, video_idx = [], []
    max_clip_len = 0
    for video_i, (vid, vidx) in tqdm(enumerate(video2idx_local.items()),
                                     desc="Computing Video Embeddings",
                                     total=len(video2idx_local)):
        video_item = val_loader.dataset.video_db[vid]
        video_batch.append(video_item)
        video_idx.append(vidx)
        if len(video_batch) == opts.vr_eval_video_batch_size or\
                video_i == len(video2idx_local) - 1:
            video_batch = move_to_cuda(video_collate(video_batch))
            # Safeguard fp16
            for k, item in video_batch.items():
                if isinstance(item, torch.Tensor) and\
                        item.dtype == torch.float32:
                    video_batch[k] = video_batch[k].to(
                        dtype=next(model.parameters()).dtype)
            curr_frame_embeddings = model.v_encoder(video_batch, 'repr')
            curr_c_attn_masks = video_batch['c_attn_masks']
            curr_clip_len = curr_frame_embeddings.size(-2)
            assert curr_clip_len <= model_opts.max_clip_len

            if total_frame_embeddings is None:
                feat_dim = curr_frame_embeddings.size(-1)
                total_frame_embeddings = torch.zeros(
                    (len(video2idx_local), model_opts.max_clip_len, feat_dim),
                    dtype=curr_frame_embeddings.dtype,
                    device=curr_frame_embeddings.device)
                total_c_attn_masks = torch.zeros(
                    (len(video2idx_local), model_opts.max_clip_len),
                    dtype=curr_c_attn_masks.dtype,
                    device=curr_frame_embeddings.device)
            indices = torch.LongTensor(video_idx)
            total_frame_embeddings[indices, :curr_clip_len] =\
                curr_frame_embeddings
            total_c_attn_masks[indices, :curr_clip_len] =\
                curr_c_attn_masks
            max_clip_len = max(max_clip_len, curr_clip_len)
            video_batch, video_idx = [], []
    total_frame_embeddings = total_frame_embeddings[:, :max_clip_len, :]
    total_c_attn_masks = total_c_attn_masks[:, :max_clip_len]

    sorted_q2c_indices, sorted_q2c_scores = None, None
    total_qids, total_vids = [], []
    for batch in tqdm(val_loader, desc="Computing q2vScores"):
        qids = batch['qids']
        vids = batch['vids']
        if has_gt_target and vids[0] == -1:
            has_gt_target = False
            LOGGER.info(
                "No GT annotations provided, only generate predictions")

        del batch['targets']
        del batch['qids']
        del batch['vids']

        total_qids.extend(qids)
        total_vids.extend(vids)
        for qid in qids:
            # fix msrvtt query data to have tvr format
            gt = query_data[qid]
            gt["desc_id"] = qid
            if "vid_name" not in gt and has_gt_target:
                # FIXME: quick hack
                gt["vid_name"] = gt["clip_name"]
            partial_query_data.append(gt)
        # Safeguard fp16
        for k, item in batch.items():
            if isinstance(item, torch.Tensor) and item.dtype == torch.float32:
                batch[k] = batch[k].to(
                    dtype=next(model.parameters()).dtype)

        # FIXME
        _q2video_scores, _, _ = model.get_pred_from_raw_query(
                total_frame_embeddings, total_c_attn_masks, **batch,
                cross=True, val_gather_gpus=False)
        n_ex += len(qids)

        _q2video_scores = _q2video_scores.float()

        q2video_scores = _q2video_scores
        _sorted_q2c_scores, _sorted_q2c_indices = \
            torch.topk(q2video_scores, max_vr_video,
                       dim=1, largest=True)
        if sorted_q2c_indices is None:
            sorted_q2c_indices = _sorted_q2c_indices.cpu().numpy()
            sorted_q2c_scores = _sorted_q2c_scores.cpu().numpy()
        else:
            sorted_q2c_indices = np.concatenate(
                (sorted_q2c_indices, _sorted_q2c_indices.cpu().numpy()),
                axis=0)
            sorted_q2c_scores = np.concatenate(
                (sorted_q2c_scores, _sorted_q2c_scores.cpu().numpy()),
                axis=0)

    vr_res = []
    for vr_i, (_sorted_q2c_scores_row, _sorted_q2c_indices_row) in tqdm(
                enumerate(
                    zip(sorted_q2c_scores[:, :100],
                        sorted_q2c_indices[:, :100])),
                desc="[VR] Loop over queries to generate predictions",
                total=len(total_qids)):
        cur_vr_redictions = []
        for v_score, v_meta_idx in zip(_sorted_q2c_scores_row,
                                       _sorted_q2c_indices_row):
            video_idx = video2idx_global[video_ids[v_meta_idx]]
            cur_vr_redictions.append([video_idx, 0, 0, float(v_score)])
        cur_query_pred = dict(desc_id=total_qids[vr_i],
                              desc="",
                              predictions=cur_vr_redictions)
        vr_res.append(cur_query_pred)
    eval_res = dict(VR=vr_res)
    eval_res = {k: v for k, v in eval_res.items() if len(v) != 0}
    eval_res["video2idx"] = video2idx_global

    eval_results_top_n = get_results_top_n(
        eval_res, top_n=max_vr_video)

    if has_gt_target:
        metrics = eval_retrieval(eval_results_top_n, partial_query_data,
                                 iou_thds=VCMR_IOU_THDS,
                                 match_number=True,
                                 verbose=False,
                                 use_desc_type=False)

        if model_opts.distributed_eval:
            n_ex_per_rank = all_gather_list(n_ex)
            metrics_per_rank = all_gather_list(metrics)
        else:
            n_ex_per_rank = [n_ex]
            metrics_per_rank = [metrics]
        n_ex = sum(n_ex_per_rank)
        val_log = {}
        gathered_metrics = {}
        for task_type, task_metric in metrics.items():
            gathered_metrics[task_type] = {}
            for k in task_metric.keys():
                if k == "desc_type_ratio":
                    continue
                gathered_v = 0
                for idx, n in enumerate(n_ex_per_rank):
                    gathered_v += n*metrics_per_rank[idx][task_type][k]
                gathered_v = gathered_v / n_ex
                gathered_metrics[task_type][k] = gathered_v
                val_log[
                    f'valid_{split}_{task_type}/{task_type}_{k}'] = gathered_v

        LOGGER.info("metrics_VR \n{}".format(pprint.pformat(
                gathered_metrics["VR"], indent=4)))

        tot_time = time()-st
        val_log.update(
            {f'valid/vr_{split}_ex_per_s': n_ex/tot_time})
        LOGGER.info(f"validation finished in {int(tot_time)} seconds")
    model.train()
    return val_log, eval_results_top_n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--sub_txt_db",
                        default="/txt/yc2_subtitles.db",
                        type=str,
                        help="The input video subtitle corpus. (LMDB)")
    parser.add_argument("--vfeat_db",
                        default="/video/yc2", type=str,
                        help="The input video frame features.")
    parser.add_argument("--query_txt_db",
                        default="/txt/yc2r_val.db",
                        type=str,
                        help="The input test query corpus. (LMDB)")
    parser.add_argument("--split",
                        default="val", type=str,
                        help="The input query split")
    parser.add_argument("--task", choices=["vatex_en_r",
                                           "yc2r"],
                        default="yc2r", type=str,
                        help="The evaluation vr task")
    parser.add_argument("--vid2dur_idx",
                        default="/txt/yc2_subtitles.db/vid2dur_idx.jsonl",
                        type=str, help="vid2dur_idx used for vr evaluation")
    parser.add_argument("--checkpoint",
                        default="", type=str,
                        help="pretrained model checkpoint steps")
    parser.add_argument("--batch_size",
                        default=80, type=int,
                        help="number of queries in a batch")
    parser.add_argument("--vr_eval_video_batch_size",
                        default=50, type=int,
                        help="number of videos in a batch")
    parser.add_argument("--vfeat_interval",
                        default=-1, type=float,
                        help="vfeat_interval in evaluation")

    parser.add_argument(
        "--output_dir", default="", type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # device parameters
    parser.add_argument('--no_fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--no_pin_mem', action='store_true',
                        help="pin memory")

    args = parser.parse_args()

    # options safe guard
    # TODO

    main(args)
