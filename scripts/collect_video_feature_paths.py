"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

gather feature paths

copied/modified from HERO
(https://github.com/linjieli222/HERO)
"""
import os
import numpy as np
import pickle as pkl
import argparse
from tqdm import tqdm
from cytoolz import curry
import multiprocessing as mp
# released feature .tar filename: 'resnet' 'slowfast' 'mil-nce-s3d' 'clip-vit'
FEAT_DIR = {
    "resnet": "resnet",
    "slowfast": "slowfast",
    "mil-nce": "mil-nce-s3d",
    "clip-vit": "clip-vit"}


@curry
def load_npz(dir_3d, dir_2d, f_3d):
    vid = f_3d.split("/")[-1].split(".npz")[0]
    folder_name = f_3d.split("/")[-2]
    f_2d = f_3d.replace(dir_3d, dir_2d)
    try:
        feature_3d = np.load(f_3d, allow_pickle=True)
        feat_len_3d = max(0, len(feature_3d["features"]))
    except Exception:
        feat_len_3d = 0
    feat_len_2d = 0
    if feat_len_3d == 0:
        f_3d = ""
        print(f"Corrupted {dir_3d.split('/')[-1]} feature for {vid}")
    # print(f_2d)
    if not os.path.exists(f_2d):
        f_2d = ""
        print(f"{dir_2d.split('/')[-1]} files for {vid} does not exists")
    else:
        try:
            feature_2d = np.load(f_2d, allow_pickle=True)
            feat_len_2d = len(feature_2d["features"])
        except Exception:
            feat_len_2d = 0
            f_2d = ""
            print(f"Corrupted {dir_2d.split('/')[-1]} files for {vid}")
    frame_len = min(feat_len_3d, feat_len_2d)
    return vid, frame_len, f_3d, f_2d, folder_name


def main(opts):
    name_2d, name_3d = opts.feat_version.split("_")
    dir_3d = os.path.join(opts.feature_dir, FEAT_DIR[name_3d])
    dir_2d = os.path.join(opts.feature_dir, FEAT_DIR[name_3d])
    failed_2d_files = []
    failed_3d_files = []
    loaded_file = []
    for root, dirs, curr_files in os.walk(f'{dir_3d}/'):
        for f in curr_files:
            if f.endswith('.npz'):
                f_3d = os.path.join(root, f)
                loaded_file.append(f_3d)
    print(f"Found {len(loaded_file)} {name_3d} files....")
    print(f"sample loaded_file: {loaded_file[:3]}")
    failed_2d_files, failed_3d_files = [], []
    files = {}
    load = load_npz(dir_3d, dir_2d)
    with mp.Pool(opts.nproc) as pool, tqdm(total=len(loaded_file)) as pbar:
        for i, (vid, frame_len, f_3d,
                f_2d, folder_name) in enumerate(
                pool.imap_unordered(load, loaded_file, chunksize=128)):
            files[vid] = (frame_len, f_3d, f_2d, folder_name)
            if f_2d == "":
                video_file = os.path.join(folder_name, vid)
                failed_2d_files.append(video_file)
            if f_3d == "":
                video_file = os.path.join(folder_name, vid)
                failed_3d_files.append(video_file)
            pbar.update(1)
    output_dir = os.path.join(opts.output, opts.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    pkl.dump(files, open(os.path.join(
        output_dir, f"{opts.feat_version}_info.pkl"), "wb"))
    if len(failed_3d_files):
        pkl.dump(failed_3d_files, open(os.path.join(
            output_dir, f"failed_{name_3d}_files.pkl"), "wb"))
    if len(failed_2d_files):
        pkl.dump(failed_2d_files, open(os.path.join(
            output_dir, f"failed_{name_2d}_files.pkl"), "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir",
                        default="",
                        type=str, help="The input video feature dir.")
    parser.add_argument("--output", default=None, type=str,
                        help="output dir")
    parser.add_argument('--dataset', type=str,
                        default="")
    parser.add_argument('--feat_version', type=str,
                        choices=[
                            "resnet_slowfast", "resnet_mil-nce",
                            "clip-vit_slowfast", "clip-vit_mil-nce"],
                        default="resnet_slowfast")
    parser.add_argument('--nproc', type=int, default=10,
                        help='number of cores used')
    args = parser.parse_args()
    main(args)
