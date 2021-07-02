"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Two-stream evaluation for
1. VATEX_EN_R
2. YC2R

copied/modified from HERO
(https://github.com/linjieli222/HERO)
"""
import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import argparse
from os.path import exists
from time import time

import torch
import numpy as np
from tqdm import tqdm
import pprint
from horovod import torch as hvd

from data import video_collate
from data.loader import move_to_cuda

from utils.logger import LOGGER
from utils.const import VCMR_IOU_THDS
from utils.tvr_standalone_eval import eval_retrieval
from utils.distributed import all_gather_list
from utils.basic_utils import save_json
from utils.tvr_eval_utils import get_results_top_n, format_submission_file
from eval_vr import load_model, load_inf_data


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
    video_only_model_opts, video_only_model = load_model(
        opts.video_only_model_dir, opts.video_only_checkpoint, opts, device)
    video_only_loader = load_inf_data(
        opts, video_only_model_opts, mode="video_only")

    if exists(opts.sub_only_model_dir):
        sub_only_model_opts, sub_only_model = load_model(
            opts.sub_only_model_dir, opts.sub_only_checkpoint,
            opts, device)
        sub_only_loader = load_inf_data(
            opts, sub_only_model_opts, mode="sub_only")
    else:
        sub_only_model, sub_only_loader = None, None

    valid_log, results = validate_full_vr(
        video_only_model, video_only_loader,
        opts.split, opts, video_only_model_opts,
        model2=sub_only_model, val_loader2=sub_only_loader)

    result_dir = os.path.join(
        f'{opts.video_only_model_dir}',
        f'results_{opts.task}_{opts.split}_two_stream')
    if not exists(result_dir) and rank == 0:
        os.makedirs(result_dir)

    all_results_list = all_gather_list(results)
    if hvd.rank() == 0:
        all_results = {"video2idx": all_results_list[0]["video2idx"]}
        for rank_id in range(hvd.size()):
            for key, _ in all_results_list[rank_id].items():
                if key == "video2idx":
                    continue
                if key not in all_results:
                    all_results[key] = []
                all_results[key].extend(all_results_list[rank_id][key])
        LOGGER.info('All results joined......')
        all_results = format_submission_file(all_results)
        save_json(
            all_results,
            f'{result_dir}/results_{opts.video_only_checkpoint}_all.json')
        save_json(
            valid_log,
            f'{result_dir}/scores_{opts.video_only_checkpoint}.json')
        LOGGER.info('All results written......')


@torch.no_grad()
def validate_full_vr(
        model1, val_loader1, split, opts,
        model_opts, task=None, model2=None,
        val_loader2=None):
    if task is None:
        task = opts.task
    LOGGER.info("start running  full VR evaluation"
                f"on {task} {split} split...")
    if hasattr(model_opts, "max_vr_video"):
        max_vr_video = model_opts.max_vr_video
    else:
        max_vr_video = 100

    model1.eval()
    n_ex = 0
    st = time()
    val_log = {}
    has_gt_target = True
    val_vid2idx = val_loader1.dataset.vid2idx
    if val_loader2 is not None:
        assert model2 is not None
        model2.eval()
        val_vid2idx = val_loader2.dataset.vid2idx  # sub_only
    if split in val_vid2idx:
        video2idx_global = val_vid2idx[split]
    else:
        video2idx_global = val_vid2idx

    video_ids = sorted(list(video2idx_global.keys()))
    video2idx_local = {e: i for i, e in enumerate(video_ids)}
    query_data = val_loader1.dataset.query_data

    partial_query_data = []

    def get_video_embeddings(val_loader1, model1):
        total_frame_embeddings = None
        video_batch, video_idx = [], []
        max_clip_len = 0
        for video_i, (vid, vidx) in tqdm(enumerate(video2idx_local.items()),
                                         desc="Computing Video Embeddings",
                                         total=len(video2idx_local)):
            video_item = val_loader1.dataset.video_db[vid]
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
                            dtype=next(model1.parameters()).dtype)
                curr_frame_embeddings = model1.v_encoder(video_batch, 'repr')
                curr_c_attn_masks = video_batch['c_attn_masks']
                curr_clip_len = curr_frame_embeddings.size(-2)
                assert curr_clip_len <= model_opts.max_clip_len

                if total_frame_embeddings is None:
                    feat_dim = curr_frame_embeddings.size(-1)
                    total_frame_embeddings = torch.zeros(
                        (len(video2idx_local), model_opts.max_clip_len,
                         feat_dim),
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
        return total_frame_embeddings, total_c_attn_masks
    total_frame_embeddings, total_c_attn_masks = get_video_embeddings(
                                                    val_loader1, model1)

    if val_loader2 is not None:
        total_frame_embeddings2, total_c_attn_masks2 = get_video_embeddings(
            val_loader2, model2)

    sorted_q2c_indices, sorted_q2c_scores = None, None
    total_qids, total_vids = [], []
    for batch in tqdm(val_loader1, desc="Computing q2vScores"):
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
                    dtype=next(model1.parameters()).dtype)

        # FIXME
        _q2video_scores, _, _ =\
            model1.get_pred_from_raw_query(
                total_frame_embeddings, total_c_attn_masks, **batch,
                cross=True, val_gather_gpus=False)
        if val_loader2 is not None:
            _q2video_scores2, _, _ =\
                model2.get_pred_from_raw_query(
                    total_frame_embeddings2, total_c_attn_masks2, **batch,
                    cross=True, val_gather_gpus=False)
            _q2video_scores = _q2video_scores/2.0 + _q2video_scores2/2.0
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
    model1.train()
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
    parser.add_argument("--task", choices=["vatex_video_sub",
                                           "vatex_video_only",
                                           "yc2r_video_sub",
                                           "yc2r_video_only"],
                        default="yc2r_video_sub", type=str,
                        help="The evaluation vr task")
    parser.add_argument("--video_only_checkpoint",
                        default="", type=str,
                        help="pretrained model checkpoint steps")
    parser.add_argument("--sub_only_checkpoint",
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
        "--video_only_model_dir", default="", type=str,
        help="The output directory where the model checkpoints will be "
             "written (video_only_model).")

    parser.add_argument(
        "--sub_only_model_dir", default="", type=str,
        help="sub_only model")

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
