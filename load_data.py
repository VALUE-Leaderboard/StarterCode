"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Data loading functions

copied/modified from HERO
(https://github.com/linjieli222/HERO
"""

from torch.utils.data import DataLoader, ConcatDataset
from utils.basic_utils import load_json
from data import *
from utils.logger import LOGGER
from utils.distributed import all_gather_list
from os.path import exists, join


def get_video_ids(query_txt_db, video_db=None):
    all_video_ids = {}
    if video_db is not None and exists(f'{video_db}/id2nframe.json'):
        all_video_ids = load_json(f'{video_db}/id2nframe.json')
    if exists(f'{query_txt_db}/query2video.json'):
        q2v = load_json(f'{query_txt_db}/query2video.json')
        qids = load_json(f'{query_txt_db}/id2len.json').keys()
        video_ids = set([q2v[qid] for qid in qids])
    else:
        video_ids = set(load_json(f'{query_txt_db}/video_ids.json'))
    if len(all_video_ids):
        all_video_ids = set(all_video_ids.keys())
        difference = list(video_ids.difference(all_video_ids))
        if len(difference):
            LOGGER.info(
                f"Ignoring {len(difference)} missing video features, "
                f"for example: {difference[:3]}")
        video_ids = list(video_ids.intersection(all_video_ids))
    return video_ids


class VFeatDbGroup(object):
    def __init__(self, vfeat_interval, vfeat_version,
                 compressed_db, max_clip_len):
        self.path2vFeatdb = {}
        self.vfeat_version = vfeat_version
        self.max_clip_len = max_clip_len
        self.vfeat_interval = vfeat_interval
        self.compressed_db = compressed_db

    def __getitem__(self, vfeat_path):
        vfeat_db = self.path2vFeatdb.get(vfeat_path, None)
        if vfeat_db is None:
            vfeat_db = VideoFeatLmdb(
                vfeat_path, self.vfeat_version,
                self.vfeat_interval,  self.compressed_db,
                self.max_clip_len)
            self.path2vFeatdb[vfeat_path] = vfeat_db
        return vfeat_db


class SubDbGroup(object):
    def __init__(self, max_clip_len):
        self.path2Subdb = {}
        self.max_clip_len = max_clip_len

    def __getitem__(self, sub_path):
        sub_db = self.path2Subdb.get(sub_path, None)
        if sub_db is None:
            sub_db = SubTokLmdb(sub_path, self.max_clip_len)
            self.path2Subdb[sub_path] = sub_db
        return sub_db


def load_video_sub_dataset(vfeat_path, sub_txt_db, vfeat_interval, opts):
    if isinstance(vfeat_path, str):
        vfeat_db = VideoFeatLmdb(
            vfeat_path, opts.vfeat_version,
            vfeat_interval,  opts.compressed_db,
            opts.max_clip_len)
    elif isinstance(vfeat_path, VideoFeatLmdb):
        vfeat_db = vfeat_path
    else:
        raise ValueError(f"Invalid vfeat_path {vfeat_path}")
    if not isinstance(sub_txt_db, SubTokLmdb):
        sub_txt_db = SubTokLmdb(sub_txt_db, opts.max_clip_len)
    video_db = VideoFeatSubTokDataset(
        sub_txt_db, vfeat_db,
        sub_ctx_len=opts.sub_ctx_len)
    return video_db


def load_video_only_dataset(vfeat_path, txt_meta, vfeat_interval, opts):
    if isinstance(vfeat_path, str):
        vfeat_db = VideoFeatLmdb(
            vfeat_path, opts.vfeat_version,
            vfeat_interval,  opts.compressed_db,
            opts.max_clip_len)
    elif isinstance(vfeat_path, VideoFeatLmdb):
        vfeat_db = vfeat_path
    else:
        raise ValueError(f"Invalid vfeat_path {vfeat_path}")
    video_db = VideoFeatDataset(
        txt_meta, vfeat_db)
    return video_db


def load_sub_only_dataset(vfeat_path, sub_txt_db, vfeat_interval, opts):
    if isinstance(vfeat_path, str):
        vfeat_db = VideoFeatLmdb(
            vfeat_path, opts.vfeat_version,
            vfeat_interval,  opts.compressed_db,
            opts.max_clip_len)
    elif isinstance(vfeat_path, VideoFeatLmdb):
        vfeat_db = vfeat_path
    else:
        raise ValueError(f"Invalid vfeat_path {vfeat_path}")
    if not isinstance(sub_txt_db, SubTokLmdb):
        sub_txt_db = SubTokLmdb(sub_txt_db, opts.max_clip_len)
    video_db = SubOnlyDataset(
        sub_txt_db, vfeat_db,
        sub_ctx_len=opts.sub_ctx_len)
    return video_db


class VideoDbGroup(object):
    def __init__(self, all_vfeat_dbs, all_sub_dbs):
        self.all_vfeat_dbs = all_vfeat_dbs
        self.all_sub_dbs = all_sub_dbs
        self.name2Videodb = {}

    def __getitem__(self, dset_name, vfeat_path, sub_path, opts):
        video_db_name = vfeat_path
        if "video_only" in dset_name:
            video_db_name += "_video_only"
        elif "sub_only" in dset_name:
            video_db_name += "_sub_only"
        else:
            video_db_name = video_db_name
        video_db = self.name2Videodb.get(video_db_name, None)
        if video_db is None:
            vfeat_db = self.all_vfeat_dbs[vfeat_path]
            if "video_only" in dset_name:
                sub_db = load_json(
                    join(sub_path, "meta.json"))
                video_db = load_video_only_dataset(
                    vfeat_db, sub_db,
                    opts.vfeat_interval, opts)
            elif "sub_only" in dset_name:
                sub_db = self.all_sub_dbs[sub_path]
                video_db = load_sub_only_dataset(
                    vfeat_db, sub_db,
                    opts.vfeat_interval, opts)
            else:
                sub_db = self.all_sub_dbs[sub_path]
                video_db = load_video_sub_dataset(
                    vfeat_db, sub_db,
                    opts.vfeat_interval, opts)
            self.name2Videodb[video_db_name] = video_db
        return video_db


def build_retrieval_qa_dataloader(
        task_dset, video_db, q_txt_db_path, is_train, opts,
        batch_size=None):

    kwargs = {'num_workers': opts.n_workers,
              'pin_memory': not opts.no_pin_mem}

    task, dset_name = task_dset.split("/")
    if is_train:
        LOGGER.info(f"Loading {task_dset} train dataset "
                    f"{video_db.img_db.img_dir}, {q_txt_db_path}")
        if batch_size is None:
            batch_size = opts.train_batch_size
    else:
        if batch_size is None:
            batch_size = opts.val_batch_size
        LOGGER.info(f"Loading {task_dset} validation dataset "
                    f"{video_db.img_db.img_dir}, {q_txt_db_path}")
    if isinstance(video_db, VideoFeatDataset):
        videoOnly = True
        subOnly = False
    elif isinstance(video_db, SubOnlyDataset):
        videoOnly = False
        subOnly = True
    else:
        videoOnly = False
        subOnly = False
    video_ids = get_video_ids(q_txt_db_path)

    if task == "videoQA":
        if is_train:
            q_txt_db = QaQueryTokLmdb(q_txt_db_path, opts.max_txt_len)
            if "vlep" in dset_name:
                DatasetCLS = VlepDataset
                if videoOnly:
                    DatasetCLS = VlepVideoOnlyDataset
                if subOnly:
                    DatasetCLS = VlepSubOnlyDataset
                collate_fn = vlep_collate
            else:
                DatasetCLS = VideoQaDataset
                if videoOnly:
                    DatasetCLS = VideoQaVideoOnlyDataset
                if subOnly:
                    DatasetCLS = VideoQaSubOnlyDataset
                collate_fn = video_qa_collate
        else:
            q_txt_db = QaQueryTokLmdb(q_txt_db_path, -1)
            if "vlep" in dset_name:
                DatasetCLS = VlepEvalDataset
                if videoOnly:
                    DatasetCLS = VlepVideoOnlyEvalDataset
                if subOnly:
                    DatasetCLS = VlepSubOnlyEvalDataset
                collate_fn = vlep_eval_collate
            else:
                DatasetCLS = VideoQaEvalDataset
                if videoOnly:
                    DatasetCLS = VideoQaVideoOnlyEvalDataset
                if subOnly:
                    DatasetCLS = VideoQaSubOnlyEvalDataset
                collate_fn = video_qa_eval_collate
    elif task == "vcmr":
        if is_train:
            q_txt_db = QueryTokLmdb(q_txt_db_path, opts.max_txt_len)
            DatasetCLS = VcmrDataset
            if videoOnly:
                DatasetCLS = VcmrVideoOnlyDataset
            if subOnly:
                DatasetCLS = VcmrSubOnlyDataset
            collate_fn = vcmr_collate
        else:
            q_txt_db = QueryTokLmdb(q_txt_db_path, -1)
            DatasetCLS = VcmrEvalDataset
            if videoOnly:
                DatasetCLS = VcmrVideoOnlyEvalDataset
            if subOnly:
                DatasetCLS = VcmrSubOnlyEvalDataset
            collate_fn = vcmr_eval_collate
    elif task == "vr":
        if "msrvtt" in dset_name:
            queryLmdb = MsrvttQueryTokLmdb
        else:
            queryLmdb = VrQueryTokLmdb
        if is_train:
            q_txt_db = queryLmdb(q_txt_db_path, opts.max_txt_len)
            DatasetCLS = VrDataset
            if videoOnly:
                DatasetCLS = VrVideoOnlyDataset
            if subOnly:
                DatasetCLS = VrSubOnlyDataset
            collate_fn = vr_collate
        else:
            q_txt_db = queryLmdb(q_txt_db_path, -1)
            DatasetCLS = VrEvalDataset
            if videoOnly:
                DatasetCLS = VrVideoOnlyEvalDataset
            if subOnly:
                DatasetCLS = VrSubOnlyEvalDataset
            collate_fn = vr_eval_collate
    elif task == "violin":
        if is_train:
            q_txt_db = QaQueryTokLmdb(q_txt_db_path, opts.max_txt_len)
            DatasetCLS = ViolinDataset
            if videoOnly:
                DatasetCLS = ViolinVideoOnlyDataset
            if subOnly:
                DatasetCLS = ViolinSubOnlyDataset
            collate_fn = violin_collate
        else:
            q_txt_db = QaQueryTokLmdb(q_txt_db_path, -1)
            DatasetCLS = ViolinEvalDataset
            if videoOnly:
                DatasetCLS = ViolinVideoOnlyEvalDataset
            if subOnly:
                DatasetCLS = ViolinSubOnlyEvalDataset
            collate_fn = violin_eval_collate
    else:
        raise ValueError(f'Undefined task {task}')
    dataset = DatasetCLS(video_ids, video_db, q_txt_db)
    LOGGER.info(f"{sum(all_gather_list(len(dataset)))} samples loaded")
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=collate_fn,
                        shuffle=is_train,
                        **kwargs)
    if is_train:
        return loader
    else:
        return PrefetchLoader(loader)


def build_caption_dataloader(
        task_dset, video_db, cap_path, is_train, opts,
        batch_size=None):
    _, dset_name = task_dset.split("/")
    if is_train:
        LOGGER.info(f"Loading {task_dset} train dataset "
                    f"{video_db.img_db.img_dir}, {cap_path}")
        if batch_size is None:
            batch_size = opts.train_batch_size
    else:
        if batch_size is None:
            batch_size = opts.val_batch_size
        LOGGER.info(f"Loading {task_dset} validation dataset "
                    f"{video_db.img_db.img_dir}, {cap_path}")
    if isinstance(video_db, VideoFeatDataset):
        videoOnly = True
        subOnly = False
    elif isinstance(video_db, SubOnlyDataset):
        videoOnly = False
        subOnly = True
    else:
        videoOnly = False
        subOnly = False
    if is_train:
        dsets = []
        for db_path in cap_path:
            if "tvc" in dset_name:
                CapTokCLS = TvcTokLmdb
            else:
                CapTokCLS = CaptionTokLmdb
            cap = CapTokCLS(db_path, opts.max_txt_len)
            if "tvc" in dset_name:
                DatasetCLS = TvcTrainDataset
                if videoOnly:
                    DatasetCLS = TvcVideoOnlyTrainDataset
                elif subOnly:
                    DatasetCLS = TvcSubOnlyTrainDataset
                dset = DatasetCLS(
                    video_db, cap, opts.max_cap_per_vid)
            else:
                DatasetCLS = VideoCapTrainDataset
                if videoOnly:
                    DatasetCLS = VideoCapVideoOnlyTrainDataset
                elif subOnly:
                    DatasetCLS = VideoCapSubOnlyTrainDataset
                dset = DatasetCLS(video_db, cap)
            dsets.append(dset)
        dataset = ConcatDataset(dsets)
    else:
        if "tvc" in dset_name:
            DatasetCLS = TvcEvalDataset
            if isinstance(video_db, VideoFeatDataset):
                DatasetCLS = TvcVideoOnlyEvalDataset
            elif isinstance(video_db, SubOnlyDataset):
                DatasetCLS = TvcSubOnlyEvalDataset
        else:
            DatasetCLS = VideoCapEvalDataset
            if isinstance(video_db, VideoFeatDataset):
                DatasetCLS = VideoCapVideoOnlyEvalDataset
            elif isinstance(video_db, SubOnlyDataset):
                DatasetCLS = VideoCapSubOnlyEvalDataset
        dataset = DatasetCLS(video_db, cap_path)
    LOGGER.info(f"{sum(all_gather_list(len(dataset)))} samples loaded")
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=opts.n_workers,
                        pin_memory=not opts.no_pin_mem,
                        collate_fn=DatasetCLS.collate,
                        shuffle=is_train)
    if is_train:
        return loader
    else:
        return PrefetchLoader(loader)
