"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run evaluation of YC2C or infenrece for submission
generate prediction from JSON file

copied/modified from HERO
(https://github.com/linjieli222/HERO
"""
import argparse
import json

from horovod import torch as hvd
from transformers import RobertaTokenizer

from model.videoCap import VideoCapGenerator
from eval.yc2c import Yc2cEval
from utils.distributed import all_gather_list
from utils.basic_utils import save_jsonl

from inf_tvc import load_model
from inf_vatex_en_c import load_inf_data, decode


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
    model.eval()
    generator = VideoCapGenerator(
        model, opts.max_gen_step, bos, eos, not opts.no_fp16)
    results = decode(loader, generator, toker)
    import os
    output_path = os.path.join(opts.model_dir, opts.output)
    save_jsonl(results, output_path)

    # evaluate score if possible
    if (hvd.rank() == 0
            and 'descs' in json.loads(next(iter(open(opts.target_clip))))):
        evaluator = Yc2cEval(opts.target_clip)
        score = evaluator(results)
        print(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_txt_db",
                        default="/txt/yc2_subtitles.db",
                        type=str,
                        help="The input video subtitle corpus. (LMDB)")
    parser.add_argument("--vfeat_db",
                        default="/video/yc2", type=str,
                        help="The input video frame features.")
    parser.add_argument("--model_dir", required=True, type=str,
                        help="dir root to trained model")
    parser.add_argument("--ckpt_step", required=True, type=int,
                        help="checkpoint step")
    parser.add_argument("--output", type=str, required=True,
                        help="output file name")

    parser.add_argument("--batch_size", default=16, type=int,
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
