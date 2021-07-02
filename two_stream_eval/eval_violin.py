"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Two-stream evaluation for VIOLIN

copied/modified from HERO
(https://github.com/linjieli222/HERO)
"""
import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import argparse
import os
from os.path import exists
from time import time

import torch
from torch.nn import functional as F

from horovod import torch as hvd

from utils.basic_utils import save_json, save_pickle
from utils.distributed import all_gather_list
from utils.logger import LOGGER
from eval_videoQA import load_model
from eval_violin import compute_accuracies, load_inf_data


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
    video_only_model_opts, video_only_model = load_model(
        opts.video_only_model_dir, opts.video_only_checkpoint,
        opts, device)
    video_only_loader = load_inf_data(
        opts, video_only_model_opts, mode="video_only")

    if exists(opts.sub_only_model_dir):
        sub_only_model_opts, sub_only_model = load_model(
            opts.sub_only_model_dir, opts.sub_only_checkpoint, opts, device)
        sub_only_loader = load_inf_data(
            opts, sub_only_model_opts, mode="sub_only")
    else:
        sub_only_model, sub_only_loader = None, None

    valid_log, results, logits = validate_violin(
        video_only_model, video_only_loader, opts.split,
        save_logits=opts.save_logits,
        model2=sub_only_model, val_loader2=sub_only_loader)
    result_dir = os.path.join(
        f'{opts.video_only_model_dir}',
        f'results_violin_{opts.split}_two_stream')
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
            f'{result_dir}/results_{opts.video_only_checkpoint}_all.json')
        save_json(
            valid_log,
            f'{result_dir}/scores_{opts.video_only_checkpoint}.json')
        LOGGER.info('All results written......')
        if opts.save_logits:
            save_pickle(
                all_logits,
                f'{result_dir}/logits_{opts.video_only_checkpoint}_all.pkl')
            LOGGER.info('All logits written......')


@torch.no_grad()
def validate_violin(model1, val_loader1, split,
                    save_logits=False, model2=None, val_loader2=None):
    LOGGER.info(f"start running validation on VIOLIN {split} split...")
    model1.eval()
    val_loss = 0
    n_ex = 0
    tot_score = 0
    results = {}
    logits = {}
    st = time()

    has_gt_target = True
    if val_loader2 is not None:
        assert model2 is not None
        model2.eval()
        dataloader = zip(val_loader1, val_loader2)
    else:
        dataloader = val_loader1
    for i, data in enumerate(dataloader):
        if val_loader2 is not None:
            batch1, batch_2 = data
            assert batch1["qids"] == batch_2["qids"]
        else:
            batch1 = data
        targets = batch1['targets']
        if has_gt_target and targets.min() < 0:
            has_gt_target = False
            LOGGER.info("No GT annotations provided, only generate predictions")
        if 'qids' in batch1:
            qids = batch1['qids']
            del batch1['qids']

        scores = model1(batch1, "violin", compute_loss=False)
        scores = torch.sigmoid(scores) 
        if val_loader2 is not None:
            scores2 = model2(batch_2, "violin", compute_loss=False)
            scores2 = torch.sigmoid(scores2) 
            scores = scores/2. + scores2/2.
        predictions = (scores > 0.5).long()
        answers = predictions.squeeze().cpu().tolist()
        for qid, answer in zip(qids, answers):
            results[str(qid)] = answer
        if save_logits:
            scores = scores.cpu().tolist()
            for qid, logit in zip(qids, scores):
                logits[str(qid)] = logit

        if has_gt_target:
            loss = F.binary_cross_entropy(
                scores, targets.to(dtype=scores.dtype),
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
            'valid/loss': val_loss,
            'valid/acc': val_acc,
            'valid/ex_per_s': n_ex/tot_time}
        LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                    f"loss:{val_loss:.2f}, score: {val_acc*100:.2f}")
    model1.train()
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
                        default="/txt/violin_test.db",
                        type=str,
                        help="The input test query corpus. (LMDB)")
    parser.add_argument("--split", choices=["val", "test", "test_private"],
                        default="test", type=str,
                        help="The input query split")
    parser.add_argument("--video_only_checkpoint",
                        default="", type=str,
                        help="pretrained model checkpoint steps")
    parser.add_argument("--sub_only_checkpoint",
                        default="", type=str,
                        help="pretrained model checkpoint steps")
    parser.add_argument("--batch_size",
                        default=10, type=int,
                        help="number of queries in a batch")

    parser.add_argument(
        "--video_only_model_dir", default="", type=str,
        help="The output directory where the model checkpoints will be "
             "written (video_only_model).")

    parser.add_argument(
        "--sub_only_model_dir", default="", type=str,
        help="sub_only model")

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
    parser.add_argument(
        "--save_logits", action='store_true',
        help="Whether to save logits")

    args = parser.parse_args()

    # options safe guard
    # TODO

    main(args)
