"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run evaluation of VATEX_EN_C or infenrece for submission
generate prediction from JSON file

copied/modified from HERO
(https://github.com/linjieli222/HERO
"""
import argparse
import json
from time import time
import os

from horovod import torch as hvd
from transformers import RobertaTokenizer
from tqdm import tqdm

from model.videoCap import VideoCapGenerator
from eval.vatex_en_c import Vatex_en_c_Eval
from utils.distributed import all_gather_list
from utils.basic_utils import save_jsonl, load_json

from data.videoCap import (
    VideoCapEvalDataset, VideoCapSubOnlyEvalDataset,
    VideoCapVideoOnlyEvalDataset)
from load_data import (
    load_sub_only_dataset, load_video_only_dataset,
    load_video_sub_dataset)
from inf_tvc import load_model
from torch.utils.data import DataLoader
from data.loader import PrefetchLoader


def load_inf_data(opts, model_opts, mode="video_sub"):
    # load DBs and image dirs
    vfeat_interval = model_opts.vfeat_interval
    if mode == "video_sub":
        video_db = load_video_sub_dataset(
            opts.vfeat_db, opts.sub_txt_db, vfeat_interval,
            model_opts)
    elif mode == "video_only":
        txt_meta = load_json(
            os.path.join(opts.sub_txt_db, "meta.json"))
        video_db = load_video_only_dataset(
            opts.vfeat_db, txt_meta,
            vfeat_interval,
            model_opts)
    else:  # sub_only
        video_db = load_sub_only_dataset(
            opts.vfeat_db, opts.sub_txt_db,
            vfeat_interval, model_opts)
    if mode == "video_sub":
        inf_dataset = VideoCapEvalDataset
    elif mode == "video_only":
        inf_dataset = VideoCapVideoOnlyEvalDataset
    else:  # sub_only
        inf_dataset = VideoCapSubOnlyEvalDataset
    eval_dataset = inf_dataset(
        video_db, opts.target_clip)

    loader = DataLoader(
        eval_dataset, batch_size=opts.batch_size,
        num_workers=opts.n_workers,
        pin_memory=not opts.no_pin_mem,
        collate_fn=VideoCapEvalDataset.collate,
        shuffle=False)
    return PrefetchLoader(loader)


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

    model_opts, model = load_model(opts.model_dir, opts.ckpt_step, opts)
    loader = load_inf_data(opts, model_opts, mode="video_sub")

    generator = VideoCapGenerator(
        model, opts.max_gen_step, bos, eos, not opts.no_fp16)
    results = decode(loader, generator, toker)
    import os
    output_path = os.path.join(opts.model_dir, opts.output)
    save_jsonl(results, output_path)

    # evaluate score if possible
    if (hvd.rank() == 0
            and 'descs' in json.loads(next(iter(open(opts.target_clip))))):
        evaluator = Vatex_en_c_Eval(opts.target_clip)
        score = evaluator(results)
        print(score)


def decode(loader, generator, tokenizer):
    st = time()
    results = []
    for batch in tqdm(loader, desc='decoding...'):
        vids = batch['vid_names']
        cids = batch['clip_ids']
        outputs = generator.greedy_decode(batch)
        for vid, cid, out_ids in zip(vids, cids, outputs):
            output = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(out_ids))
            results.append({'vid_name': vid, 'clip_id': int(cid),
                            'descs': [{'desc': output}]})
    results = [r for rs in all_gather_list(results) for r in rs]
    print(f'decoding finished in {int(time() - st)} seconds')
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_txt_db",
                        default="/txt/vatex_subtitles.db",
                        type=str,
                        help="The input video subtitle corpus. (LMDB)")
    parser.add_argument("--vfeat_db",
                        default="/video/vatex", type=str,
                        help="The input video frame features.")
    parser.add_argument("--model_dir", required=True, type=str,
                        help="dir root to trained model")
    parser.add_argument("--ckpt_step", required=True, type=int,
                        help="checkpoint step")
    parser.add_argument("--output", type=str, required=True,
                        help="output file name")

    parser.add_argument("--batch_size", default=96, type=int,
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
