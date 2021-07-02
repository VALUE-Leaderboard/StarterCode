"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess annotations into LMDB

copied/modified from HERO
(https://github.com/linjieli222/HERO)
"""
import argparse
import json
import os
from os.path import exists
from cytoolz import curry
from tqdm import tqdm
from transformers import RobertaTokenizer
import copy

# quick hack for import
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.basic_utils import save_jsonl, save_json
from data.data import open_lmdb


@curry
def roberta_tokenize(tokenizer, text):
    if text.isupper():
        text = text.lower()
    words = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(words)
    return ids


def process_retrieval(jsonl, db, tokenizer):
    id2len = {}
    query2video = {}  # not sure if useful
    query_data = []
    for line in tqdm(
            jsonl,
            desc='processing retrieval with raw query text'):
        example = json.loads(line)
        query_data.append(copy.copy(example))
        id_ = example['desc_id']
        input_ids = tokenizer(example["desc"])
        if 'vid_name' in example:
            vid = example['vid_name']
        else:
            vid = None
        if 'ts' in example:
            target = example['ts']
        else:
            target = None
        if vid is not None:
            query2video[id_] = vid
            example['vid'] = vid
        id2len[id_] = len(input_ids)
        example['input_ids'] = input_ids
        example['target'] = target
        example['qid'] = str(id_)
        db[str(id_)] = example
    return id2len, query2video, query_data


def process_tvqa_or_how2qa(jsonl, db, tokenizer):
    id2len = {}
    query2video = {}  # not sure if useful
    query_data = []
    for line in tqdm(jsonl, desc='processing raw QA text (TVQA/How2QA)'):
        example = json.loads(line)
        query_data.append(copy.copy(example))
        id_ = example['qid']
        input_ids = [tokenizer(example["q"]), tokenizer(example["a0"]),
                     tokenizer(example["a1"]), tokenizer(example["a2"]),
                     tokenizer(example["a3"])]
        if "a4" in example:
            input_ids.append(tokenizer(example["a4"]))
        vid = example['vid_name']
        if 'ts' in example:
            ts = example['ts']
        else:
            ts = None

        if 'answer_idx' in example:
            target = example['answer_idx']
        else:
            target = None

        query2video[id_] = vid
        id2len[id_] = [len(input_id) for input_id in input_ids]
        example['input_ids'] = input_ids
        example['vid'] = vid
        example['ts'] = ts
        example['target'] = target
        example['qid'] = str(id_)
        db[str(id_)] = example
    return id2len, query2video, query_data


def process_vlep(jsonl, db, tokenizer):
    id2len = {}
    query2video = {}  # not sure if useful
    query_data = []
    for line in tqdm(jsonl, desc='processing VLEP with raw annotations'):
        example = json.loads(line)
        query_data.append(copy.copy(example))
        id_ = example['example_id']
        input_ids = [tokenizer(example["events"][0]),
                     tokenizer(example["events"][1])]
        vid = example['vid_name']
        if 'ts' in example:
            ts = example['ts']
        else:
            ts = None

        if 'answer' in example:
            target = example['answer']
        else:
            target = None

        query2video[id_] = vid
        id2len[id_] = [len(input_ids[0]), len(input_ids[1])]
        example['input_ids'] = input_ids
        example['vid'] = vid
        example['ts'] = ts
        example['target'] = target
        example['qid'] = str(id_)
        db[str(id_)] = example
    return id2len, query2video, query_data


def process_violin(jsonl, db, tokenizer):
    id2len = {}
    query2video = {}  # not sure if useful
    query_data = []
    for line in tqdm(
            jsonl,
            desc='processing Violin with raw statement text'):
        example = json.loads(line)
        query_data.append(copy.copy(example))
        id_ = example['example_id']
        input_ids = tokenizer(example["statement"])
        vid = example['vid_name']
        if 'answer' in example:
            target = example['answer']
        else:
            target = None
        query2video[id_] = vid
        example['vid'] = vid
        id2len[id_] = len(input_ids)
        example['input_ids'] = input_ids
        example['target'] = target
        example['qid'] = str(id_)
        db[str(id_)] = example
    return id2len, query2video, query_data


def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                         'for re-processing')
    meta = vars(opts)
    meta['tokenizer'] = opts.toker
    toker = RobertaTokenizer.from_pretrained(
        opts.toker)
    tokenizer = roberta_tokenize(toker)
    meta['BOS'] = toker.convert_tokens_to_ids(['<s>'])[0]
    meta['EOS'] = toker.convert_tokens_to_ids(['</s>'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['</s>'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['<s>'])[0]
    meta['PAD'] = toker.convert_tokens_to_ids(['<pad>'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['<mask>'])[0]
    meta['UNK'] = toker.convert_tokens_to_ids(['<unk>'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids(['.'])[0],
                       toker.convert_tokens_to_ids(['<|endoftext|>'])[0]+1)
    save_json(vars(opts), f'{opts.output}/meta.json', save_pretty=True)

    open_db = curry(open_lmdb, opts.output, readonly=False)
    with open_db() as db:
        with open(opts.annotation, "r") as ann:
            if opts.task in ["tvr", "how2r", "yc2r", "vatex_en_r"]:
                id2lens, query2video, query_data = process_retrieval(
                    ann, db, tokenizer)
            elif opts.task in ["tvqa", "how2qa"]:
                id2lens, query2video, query_data = process_tvqa_or_how2qa(
                    ann, db, tokenizer)

            elif opts.task == "vlep":
                id2lens, query2video, query_data = process_vlep(
                    ann, db, tokenizer)

            elif opts.task == "violin":
                id2lens, query2video, query_data = process_violin(
                    ann, db, tokenizer)
            else:
                raise NotImplementedError(
                    f"prepro for {opts.task} not implemented")

    save_json(id2lens, f'{opts.output}/id2len.json')
    if len(query2video):
        save_json(query2video, f'{opts.output}/query2video.json')
    else:
        print("No query to video mappings, need to get video ids from raw annotations (for example, vid2dur_idx.json)...")
    save_jsonl(query_data, f'{opts.output}/query_data.jsonl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True,
                        help='annotation JSON')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--toker', default='roberta-base',
                        help='which RoBerTa tokenizer to used')
    parser.add_argument('--task', default='tvr',
                        choices=["tvr", "tvqa", "how2r", "how2qa", "vlep",
                                 "violin", "vatex_en_r", "yc2r"],
                        help='which RoBerTa tokenizer to used')
    args = parser.parse_args()
    main(args)
