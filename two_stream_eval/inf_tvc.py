"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Two-stream evaluation for TVC

copied/modified from HERO
(https://github.com/linjieli222/HERO)
"""
import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import argparse
import json
from time import time

from horovod import torch as hvd
from transformers import RobertaTokenizer
from tqdm import tqdm
from eval.tvc import TvcEval
from utils.distributed import all_gather_list
from utils.basic_utils import save_jsonl

from os.path import exists
from pred_agg_eval.videocap_generator import VideoCapGenerator
from inf_tvc import load_model, load_inf_data


def main(opts):
    hvd.init()
    if hvd.rank() == 0:
        toker = RobertaTokenizer.from_pretrained('roberta-base')
        all_gather_list(None)
    else:
        all_gather_list(None)
        toker = RobertaTokenizer.from_pretrained('roberta-base')
    bos = toker.convert_tokens_to_ids(['<s>'])[0]
    eos = toker.convert_tokens_to_ids(['</s>'])[0]

    video_only_model_opts, video_only_model = load_model(
        opts.video_only_model_dir, opts.video_only_ckpt_step,
        opts)
    video_only_dataloader = load_inf_data(
        opts, video_only_model_opts, mode="video_only")

    if exists(opts.sub_only_model_dir):
        sub_only_model_opts, sub_only_model = load_model(
            opts.sub_only_model_dir,
            opts.sub_only_ckpt_step, opts)
        sub_only_dataloader = load_inf_data(
            opts, sub_only_model_opts, mode="sub_only")
    else:
        sub_only_model, sub_only_dataloader = None, None

    generator = VideoCapGenerator(
        video_only_model, opts.max_gen_step,
        bos, eos, not opts.no_fp16,
        model2=sub_only_model)

    results = decode(
        video_only_dataloader, sub_only_dataloader,
        generator, toker)
    output_path = os.path.join(
        opts.video_only_model_dir, opts.output)
    save_jsonl(results, output_path)

    # evaluate score if possible
    if (hvd.rank() == 0
            and 'descs' in json.loads(next(iter(open(opts.target_clip))))):
        evaluator = TvcEval(opts.target_clip)
        score = evaluator(results)
        print(score)


def decode(loader1, loader2, generator, tokenizer):
    st = time()
    results = []
    if loader2 is not None:
        dataloader = zip(loader1, loader2)
    else:
        dataloader = loader1
    for i, data in tqdm(enumerate(dataloader), desc='decoding...'):
        if loader2 is not None:
            batch1, batch2 = data
            assert batch1["vid_names"] == batch2["vid_names"]
            assert batch1["clip_ids"] == batch2["clip_ids"]
            assert batch1["all_ts"] == batch2["all_ts"]
        else:
            batch1 = data
            batch2 = None
        vids = batch1['vid_names']
        cids = batch1['clip_ids']
        all_ts = batch1['all_ts']
        outputs = generator.greedy_decode(batch1, batch2)
        for vid, cid, ts, out_ids in zip(vids, cids, all_ts, outputs):
            output = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(out_ids))
            results.append({'vid_name': vid, 'clip_id': cid, 'ts': ts,
                            'descs': [{'desc': output}]})
    results = [r for rs in all_gather_list(results) for r in rs]
    print(f'decoding finished in {int(time() - st)} seconds')
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_txt_db",
                        default="/txt/tv_subtitles.db",
                        type=str,
                        help="The input video subtitle corpus. (LMDB)")
    parser.add_argument("--vfeat_db",
                        default="/video/tv", type=str,
                        help="The input video frame features.")
    parser.add_argument("--video_only_model_dir", required=True, type=str,
                        help="dir root to trained model")
    parser.add_argument("--video_only_ckpt_step", required=True, type=int,
                        help="checkpoint step")
    parser.add_argument("--sub_only_model_dir", default="", type=str,
                        help="dir root to trained model")
    parser.add_argument("--sub_only_ckpt_step", default=-1, type=int,
                        help="checkpoint step")
    parser.add_argument("--output", type=str, required=True,
                        help="output file name")

    parser.add_argument("--batch_size", default=8, type=int,
                        help="validation batch size (per GPU)")
    parser.add_argument("--max_gen_step", default=30, type=int,
                        help="max generation steps")

    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--no_pin_mem', action='store_true',
                        help="disable pin memory")
    parser.add_argument("--no_fp16", action='store_true',
                        help="disable fp16")

    parser.add_argument("--target_clip", required=True, type=str,
                        help="jsonl annotation")

    args = parser.parse_args()

    main(args)
