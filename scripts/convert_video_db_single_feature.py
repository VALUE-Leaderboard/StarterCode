"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

convert feature npz file to lmdb

copied/modified from HERO
(https://github.com/linjieli222/HERO)
"""
import argparse
import glob
import io
import json
import multiprocessing as mp
import os
from os.path import exists

from cytoolz import curry
import numpy as np
from tqdm import tqdm
import lmdb

import msgpack
import msgpack_numpy
msgpack_numpy.patch()
# released feature .tar filename: 'resnet' 'slowfast' 'mil-nce-s3d' 'clip-vit'
FEAT_DIR = {
    "resnet": "resnet",
    "slowfast": "slowfast",
    "mil-nce": "mil-nce-s3d",
    "clip-vit": "clip-vit"}


@curry
def load_npz(fname):
    try:
        feature_data = np.load(
            fname, allow_pickle=True)["features"]
        if feature_data.dtype == np.float16:
            feature_data = feature_data.astype(np.float32)
        nframes = len(feature_data)
        dump = {"features": feature_data}
    except Exception as e:
        # corrupted file
        print(f'corrupted file {fname}', e)
        dump = {}
        nframes = 0

    vid = os.path.basename(fname).split(".npz")[0]
    return vid, dump, nframes


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)


def main(opts):
    db_name = f'{opts.feat_version}_{opts.frame_length:g}'
    if opts.compress:
        db_name += '_compressed'
    if not exists(f'{opts.output}/{opts.dataset}'):
        os.makedirs(f'{opts.output}/{opts.dataset}')
    env = lmdb.open(
        f'{opts.output}/{opts.dataset}/{db_name}',
        map_size=1024**4)
    txn = env.begin(write=True)
    clip_interval = int(opts.clip_interval/opts.frame_length)
    files = glob.glob(
        f'{opts.feature_dir}/{FEAT_DIR[opts.feat_version]}/*.npz')
    load = load_npz()
    name2nframes = {}
    corrupted_files = set()
    with mp.Pool(opts.nproc) as pool, tqdm(total=len(files)) as pbar:
        for i, (fname, features, nframes) in enumerate(
                pool.imap_unordered(load, files, chunksize=128)):
            if not features or nframes == 0:
                pbar.update(1)
                corrupted_files.add(fname)
                continue  # corrupted feature
            if opts.clip_interval != -1:
                feature_values = features["features"]
                clip_id = 0
                for st_ind in range(0, nframes, clip_interval):
                    clip_name = fname+f".{clip_id}"
                    ed_ind = min(st_ind + clip_interval, nframes)
                    clip_features = {
                        "features": feature_values[st_ind: ed_ind]}
                    clip_id += 1
                    if opts.compress:
                        clip_dump = dumps_npz(clip_features, compress=True)
                    else:
                        clip_dump = dumps_msgpack(clip_features)
                    txn.put(key=clip_name.encode('utf-8'), value=clip_dump)
                    name2nframes[clip_name] = ed_ind - st_ind
            else:
                if opts.compress:
                    dump = dumps_npz(features, compress=True)
                else:
                    dump = dumps_msgpack(features)
                txn.put(key=fname.encode('utf-8'), value=dump)
                name2nframes[fname] = nframes
            if i % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
            pbar.update(1)
        txn.commit()
        env.close()
    id2frame_len_file = os.path.join(
        f'{opts.output}/{opts.dataset}',
        f'id2nframe_{opts.frame_length:g}.json')
    if os.path.exists(id2frame_len_file):
        id2frame = json.load(open(id2frame_len_file, "r"))
        for key, val in id2frame.items():
            if key not in name2nframes:
                print(f"Missing: {key}")
                continue
            if val != name2nframes[key]:
                print(f"Mismatch: {val} vs. {name2nframes[key]}")
                id2frame[key] = min(val, name2nframes[key])
            assert id2frame[key] > 0
            assert name2nframes[key] >= val
    else:
        id2frame = name2nframes
        with open(id2frame_len_file, 'w') as f:
            json.dump(id2frame, f)
    corrupted_files = list(corrupted_files)
    if len(corrupted_files) > 0:
        corrupted_output_file = f'{opts.output}/{opts.dataset}/corrupted.json'
        with open(corrupted_output_file, 'w') as f:
            json.dump(corrupted_files, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", default=None, type=str,
                        help="where are the features saved")
    parser.add_argument("--output", default=None, type=str,
                        help="output lmdb")
    parser.add_argument(
        '--frame_length', type=float, default=1.5,
        help='1 feature per "frame_length" seconds used in feature extraction,'
             'in seconds (1.5/2)')
    parser.add_argument('--dataset', type=str,
                        default="")
    parser.add_argument('--feat_version', type=str,
                        choices=["resnet", "mil-nce", "slowfast", "clip-vit"],
                        default="resnet")
    parser.add_argument('--nproc', type=int, default=4,
                        help='number of cores used')
    parser.add_argument('--compress', action='store_true',
                        help='compress the tensors')
    parser.add_argument(
        '--clip_interval', type=int, default=-1,
        help="cut the whole video into small clips, in seconds"
             "set to 60 for HowTo100M videos, set to -1 otherwise")
    args = parser.parse_args()
    main(args)
