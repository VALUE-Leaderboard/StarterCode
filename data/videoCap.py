"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Video Captioning dataset

copied/modified from HERO
(https://github.com/linjieli222/HERO)
"""
import json
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import horovod.torch as hvd
from toolz.sandbox import unzip

from utils.basic_utils import load_jsonl, load_json

from .data import (SubOnlyDataset, TxtTokLmdb, VideoFeatSubTokDataset,
                   VideoFeatDataset,
                   video_collate, _check_ngpu)


class CaptionTokLmdb(TxtTokLmdb):
    """
    NOTE: this shares yc2/vatex retrieval db
    """
    def __init__(self, db_dir, max_txt_len=-1):
        super().__init__(db_dir, max_txt_len)
        if os.path.exists(f'{self.db_dir}/query2video.json'):
            self.query2video = load_json(f'{self.db_dir}/query2video.json')
            self.video2query = {}
            for k, v in self.query2video.items():
                if v not in self.video2query:
                    self.video2query[v] = [k]
                else:
                    self.video2query[v].append(k)
        else:
            self.query2video = {}
            self.video2query = {}
        self.query_data_f = load_jsonl(f'{self.db_dir}/query_data.jsonl')

    def __getitem__(self, id_):
        return self.get_caption(id_)

    def get_caption(self, id_):
        txt_dump = self.db[id_]
        cap_input_ids = txt_dump['input_ids']
        input_ids = [self.bos] + cap_input_ids
        tgt_ids = cap_input_ids + [self.eos]
        txt_dump['input_ids'] = torch.tensor(input_ids)
        txt_dump['tgt_ids'] = torch.tensor(tgt_ids)
        return txt_dump


class VideoCapTrainDataset(Dataset):
    def __validate_input_db__(self):
        assert isinstance(self.caption_db, CaptionTokLmdb)
        assert isinstance(self.video_db, VideoFeatSubTokDataset)

    def __init__(self, video_db, caption_db):
        self.video_db = video_db
        self.caption_db = caption_db
        self.__validate_input_db__()

        self.ids = list(caption_db.id2len.keys())
        self.query2video = caption_db.query2video
        if _check_ngpu() > 1:
            # partition data by rank
            self.ids = self.ids[hvd.rank()::hvd.size()]

    def __getitem__(self, i):
        id_ = self.ids[i]
        ex = self.caption_db[id_]
        vid = self.query2video[id_]

        video_inputs = self.video_db.__getitem__(vid)

        return video_inputs, ex['input_ids'], ex['tgt_ids']

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def collate(inputs):
        video_inputs, input_ids, tgt_ids = map(list, unzip(inputs))

        input_ids = pad_sequence(input_ids,
                                 batch_first=True, padding_value=1)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)
        tgt_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=-1)
        batch = {'cap_input_ids': input_ids,
                 'cap_pos_ids': position_ids,
                 'cap_tgt_ids': tgt_ids}

        vid_batch = video_collate(video_inputs)
        batch.update(vid_batch)
        return batch


class VideoCapValDataset(VideoCapTrainDataset):
    """ NOTE: indexed by video id, video and clip is equivalent """
    def __init__(self, video_db, caption_db):
        self.video_db = video_db
        self.caption_db = caption_db
        self.__validate_input_db__()

        self.vids = list(caption_db.video2query.keys())
        if _check_ngpu() > 1:
            # partition data by rank
            self.vids = self.vids[hvd.rank()::hvd.size()]

    def __getitem__(self, i):
        vid = self.vids[i]
        video_inputs = self.video_db.__getitem__(vid)

        # quick work around
        return vid, vid, video_inputs

    def __len__(self):
        return len(self.vids)

    @staticmethod
    def collate(inputs):
        vids, clip_ids, video_inputs = map(list, unzip(inputs))

        batch = {'vid_names': vids,
                 'clip_ids': clip_ids}

        vid_batch = video_collate(video_inputs)
        batch.update(vid_batch)
        return batch


class VideoCapEvalDataset(VideoCapTrainDataset):
    """ from jsonl input """
    def __validate_input_db__(self):
        assert isinstance(self.video_db, VideoFeatSubTokDataset)

    def __init__(self, video_db, data_jsonl):
        self.video_db = video_db
        self.gt_anno_path = data_jsonl
        self.__validate_input_db__()

        self.clip2vid = {}
        self.clip2ex = {}
        for line in open(data_jsonl):
            example = json.loads(line)
            vid = example['vid_name']
            clip_id = str(example['clip_id'])
            self.clip2vid[clip_id] = vid
            self.clip2ex[clip_id] = example
        self.clip_ids = list(self.clip2ex.keys())

        if _check_ngpu() > 1:
            # partition data by rank
            self.clip_ids = self.clip_ids[hvd.rank()::hvd.size()]

        self.vid2dur = video_db.vid2dur
        self.vid2idx = video_db.vid2idx
        self.max_clip_len = video_db.max_clip_len
        self.frame_interval = video_db.img_db.frame_interval

    def __getitem__(self, i):
        clip_id = self.clip_ids[i]
        vid = self.clip2vid[clip_id]

        video_inputs = self.video_db.__getitem__(vid)
        return vid, clip_id, video_inputs

    def __len__(self):
        return len(self.clip_ids)

    @staticmethod
    def collate(inputs):
        vids, clip_ids, video_inputs = map(list, unzip(inputs))

        batch = {'vid_names': vids,
                 'clip_ids': clip_ids}

        vid_batch = video_collate(video_inputs)
        batch.update(vid_batch)
        return batch


class VideoCapVideoOnlyTrainDataset(VideoCapTrainDataset):
    def __validate_input_db__(self):
        assert isinstance(self.caption_db, CaptionTokLmdb)
        assert isinstance(self.video_db, VideoFeatDataset)


class VideoCapVideoOnlyValDataset(VideoCapValDataset):
    """ for validation """
    def __validate_input_db__(self):
        assert isinstance(self.caption_db, CaptionTokLmdb)
        assert isinstance(self.video_db, VideoFeatDataset)


class VideoCapVideoOnlyEvalDataset(VideoCapEvalDataset):
    """ for generating submission from JSON input
    """
    def __validate_input_db__(self):
        assert isinstance(self.video_db, VideoFeatDataset)


class VideoCapSubOnlyTrainDataset(VideoCapTrainDataset):
    def __validate_input_db__(self):
        assert isinstance(self.caption_db, CaptionTokLmdb)
        assert isinstance(self.video_db, SubOnlyDataset)


class VideoCapSubOnlyValDataset(VideoCapValDataset):
    """ for validation """
    def __validate_input_db__(self):
        assert isinstance(self.caption_db, CaptionTokLmdb)
        assert isinstance(self.video_db, SubOnlyDataset)


class VideoCapSubOnlyEvalDataset(VideoCapEvalDataset):
    """ for generating submission from JSON input
    """
    def __validate_input_db__(self):
        assert isinstance(self.video_db, SubOnlyDataset)
