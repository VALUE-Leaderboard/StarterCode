"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run evaluation of VIOLIN or infenrece for submission

copied/modified from HERO
(https://github.com/linjieli222/HERO
"""
import argparse
import os
from os.path import exists
from time import time

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from horovod import torch as hvd

from data import (ViolinEvalDataset, violin_eval_collate,
                  QaQueryTokLmdb, PrefetchLoader,
                  ViolinSubOnlyEvalDataset, ViolinVideoOnlyEvalDataset)
from load_data import (
    get_video_ids, load_video_sub_dataset,
    load_video_only_dataset, load_sub_only_dataset)

from utils.basic_utils import save_json, save_pickle, load_json
from utils.distributed import all_gather_list
from utils.logger import LOGGER
from eval_videoQA import load_model


def load_inf_data(opts, model_opts, mode="video_sub"):
    # load DBs and image dirs
    video_ids = get_video_ids(opts.query_txt_db, opts.vfeat_db)
    if mode == "video_sub":
        video_db = load_video_sub_dataset(
            opts.vfeat_db, opts.sub_txt_db, model_opts.vfeat_interval,
            model_opts)
    elif mode == "video_only":
        txt_meta = load_json(
            os.path.join(opts.query_txt_db, "meta.json"))
        video_db = load_video_only_dataset(
            opts.vfeat_db, txt_meta,
            model_opts.vfeat_interval,
            model_opts)
    else:  # sub_only
        video_db = load_sub_only_dataset(
            opts.vfeat_db, opts.sub_txt_db,
            model_opts.vfeat_interval, model_opts)
    assert opts.split in opts.query_txt_db
    q_txt_db = QaQueryTokLmdb(opts.query_txt_db, -1)

    if mode == "video_sub":
        dataCLS = ViolinEvalDataset
    elif mode == "video_only":
        dataCLS = ViolinVideoOnlyEvalDataset
    else:  # sub_only
        dataCLS = ViolinSubOnlyEvalDataset
    eval_dataset = dataCLS(
        video_ids, video_db, q_txt_db)
    collate_fn = violin_eval_collate
    eval_dataloader = DataLoader(eval_dataset, batch_size=opts.batch_size,
                                 num_workers=opts.n_workers,
                                 pin_memory=not opts.no_pin_mem,
                                 collate_fn=collate_fn)
    eval_dataloader = PrefetchLoader(eval_dataloader)
    return eval_dataloader


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), not opts.no_fp16))
    if hvd.rank() != 0:
        LOGGER.disabled = True
    model_opts, model = load_model(
        opts.output_dir, opts.checkpoint, opts, device)
    eval_dataloader = load_inf_data(opts, model_opts, mode="video_sub")

    valid_log, results, logits = validate_violin(
        model, eval_dataloader, opts.split, opts.save_logits)
    result_dir = f'{opts.output_dir}/results_violin_{opts.split}'
    if opts.save_logits:
        result_dir += '_w_logit'
    if not exists(result_dir) and hvd.rank() == 0:
        os.makedirs(result_dir)

    all_results = {}
    for id2res in all_gather_list(results):
        all_results.update(id2res)
    if opts.save_logits:
        all_logits = {}
        for id2logit in all_gather_list(logits):
            all_logits.update(id2logit)
    if hvd.rank() == 0:
        save_json(
            all_results,
            f'{result_dir}/results_{opts.checkpoint}_all.json')
        save_json(
            valid_log,
            f'{result_dir}/scores_{opts.checkpoint}.json')
        LOGGER.info('All results written......')
        if opts.save_logits:
            save_pickle(
                all_logits,
                f'{result_dir}/logits_{opts.checkpoint}_all.pkl')
            LOGGER.info('All logits written......')


def compute_accuracies(predictions, labels):
    matched_qa = predictions.squeeze() == labels.squeeze()
    n_correct_qa = matched_qa.sum().item()
    return n_correct_qa


@torch.no_grad()
def validate_violin(model, val_loader, split, save_logits=False):
    LOGGER.info(f"start running validation on VIOLIN {split} split...")
    model.eval()
    val_loss = 0
    n_ex = 0
    tot_score = 0
    results = {}
    logits = {}
    val_log = {}
    st = time()

    has_gt_target = True

    for _, batch in enumerate(val_loader):
        targets = batch['targets']
        if has_gt_target and targets.min() < 0:
            has_gt_target = False
            LOGGER.info(
                "No GT annotations provided, only generate predictions")
        if 'qids' in batch:
            qids = batch['qids']
            del batch['qids']

        scores = model(batch, "violin", compute_loss=False)
        predictions = (torch.sigmoid(scores) > 0.5).long()
        answers = predictions.squeeze().cpu().tolist()
        for qid, answer in zip(qids, answers):
            results[str(qid)] = answer
        if save_logits:
            scores = scores.cpu().tolist()
            for qid, logit in zip(qids, scores):
                logits[str(qid)] = logit

        if has_gt_target:
            loss = F.binary_cross_entropy(
                torch.sigmoid(scores), targets.to(dtype=scores.dtype),
                reduction='sum')
            val_loss += loss.item()
            tot_score += compute_accuracies(predictions, targets)
            n_ex += len(qids)

    if has_gt_target:
        val_loss = sum(all_gather_list(val_loss))
        tot_score = sum(all_gather_list(tot_score))
        n_ex = sum(all_gather_list(n_ex))
        tot_time = time()-st
        val_loss /= n_ex
        val_acc = tot_score / n_ex
        val_log = {
            f'valid/{split}_loss': val_loss,
            f'valid/{split}_acc': val_acc,
            f'valid/{split}_ex_per_s': n_ex/tot_time}
        LOGGER.info(
            f"validation of {split} split of {n_ex} examples "
            f"finished in {int(tot_time)}s, "
            f"loss:{val_loss:.2f}, score: {val_acc*100:.2f}")
    model.train()
    return val_log, results, logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--sub_txt_db",
                        default="/txt/violin_subtitles.db",
                        type=str,
                        help="The input video subtitle corpus. (LMDB)")
    parser.add_argument("--vfeat_db",
                        default="/video/violin", type=str,
                        help="The input video frame features.")
    parser.add_argument("--query_txt_db",
                        default="/txt/violin_test_private.db",
                        type=str,
                        help="The input test query corpus. (LMDB)")
    parser.add_argument("--split", choices=["val", "test", "test_private"],
                        default="test_private", type=str,
                        help="The input query split")
    parser.add_argument("--checkpoint",
                        default="", type=str,
                        help="pretrained model checkpoint steps")
    parser.add_argument("--batch_size",
                        default=10, type=int,
                        help="number of queries in a batch")

    parser.add_argument(
        "--output_dir", default="", type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # Prepro parameters

    # device parameters
    parser.add_argument('--no_fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--no_pin_mem', action='store_true',
                        help="pin memory")
    parser.add_argument("--save_logits", action='store_true',
                        help="Whether to save logits")

    args = parser.parse_args()

    # options safe guard
    # TODO

    main(args)
