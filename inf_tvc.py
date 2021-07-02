"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run evaluation of TVC or infenrece for submission
generate prediction from JSON file

copied/modified from HERO
(https://github.com/linjieli222/HERO
"""
import argparse
import json
from time import time

import os
from os.path import exists
import torch
from horovod import torch as hvd
from transformers import RobertaTokenizer
from apex import amp
from tqdm import tqdm

from data.tvc import (
    TvcEvalDataset, TvcSubOnlyEvalDataset,
    TvcVideoOnlyEvalDataset)
from load_data import (
    load_sub_only_dataset, load_video_only_dataset,
    load_video_sub_dataset)
from model.videoCap import HeroForVideoCap, VideoCapGenerator
from eval.tvc import TvcEval
from utils.misc import Struct
from utils.distributed import all_gather_list
from utils.const import VFEAT_DIM
from utils.basic_utils import save_jsonl, load_json

from torch.utils.data import DataLoader
from data.loader import PrefetchLoader


def load_model(model_dir, checkpoint, opts):
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

    model = HeroForVideoCap.from_pretrained(
        model_config,
        state_dict=checkpoint,
        vfeat_dim=VFEAT_DIM[model_opts.vfeat_version],
        max_frm_seq_len=max_frm_seq_len,
        lsr=model_opts.lsr)
    model.cuda()
    if not opts.no_fp16:
        model = amp.initialize(model, enabled=True, opt_level='O2')
    model.eval()
    return model_opts, model


def load_inf_data(opts, model_opts, mode="video_sub"):
    # load DBs and image dirs
    vfeat_interval = model_opts.vfeat_interval
    if mode == "video_sub":
        video_db = load_video_sub_dataset(
            opts.vfeat_db, opts.sub_txt_db, model_opts.vfeat_interval,
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
        inf_dataset = TvcEvalDataset
    elif mode == "video_only":
        inf_dataset = TvcVideoOnlyEvalDataset
    else:  # sub_only
        inf_dataset = TvcSubOnlyEvalDataset
    eval_dataset = inf_dataset(
        video_db, opts.target_clip)

    loader = DataLoader(eval_dataset, batch_size=opts.batch_size,
                        num_workers=opts.n_workers,
                        pin_memory=not opts.no_pin_mem,
                        collate_fn=TvcEvalDataset.collate,
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
    if len(opts.output):
        output_path = os.path.join(opts.model_dir, opts.output)
    else:
        output_path = os.path.join(
            opts.model_dir, f"tvc_test_step{opts.ckpt_step}.jsonl")
    save_jsonl(results, output_path)

    # evaluate score if possible
    if (hvd.rank() == 0
            and 'descs' in json.loads(next(iter(open(opts.target_clip))))):
        evaluator = TvcEval(opts.target_clip)
        score = evaluator(results)
        print(score)


def decode(loader, generator, tokenizer):
    st = time()
    results = []
    for batch in tqdm(loader, desc='decoding...'):
        vids = batch['vid_names']
        cids = batch['clip_ids']
        all_ts = batch['all_ts']
        outputs = generator.greedy_decode(batch)
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
    parser.add_argument("--model_dir", required=True, type=str,
                        help="dir root to trained model")
    parser.add_argument("--ckpt_step", required=True, type=int,
                        help="checkpoint step")
    parser.add_argument("--output", type=str, default="",
                        help="output file name")

    parser.add_argument("--batch_size", default=30, type=int,
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
