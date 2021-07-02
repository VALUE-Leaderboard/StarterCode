"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Violin dataset

copied/modified from HERO
(https://github.com/linjieli222/HERO)
"""
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import (QaQueryTokLmdb, SubOnlyDataset, video_collate,
                   txt_input_collate, VideoFeatDataset)
from .videoQA import VideoQaDataset


def get_paired_statement_id(qid):
    parsed_qid = qid.split("-")
    label = int(parsed_qid[-1])
    paired_qid = "-".join(parsed_qid[:-1]+[str(1 - label)])
    return paired_qid


class ViolinDataset(VideoQaDataset):
    def __init__(self, video_ids, video_db, query_db, max_num_query=6,
                 sampled_by_q=True):
        super().__init__(
            video_ids, video_db, query_db, max_num_query,
            sampled_by_q)

    def __getitem__(self, i):
        vid, qids = self.getids(i)
        video_inputs = self.video_db.__getitem__(vid)
        (frame_level_input_ids, frame_level_v_feats,
         frame_level_attn_masks, frame_level_sub_attn_masks,
         clip_level_v_feats, clip_level_attn_masks, num_subs,
         sub_idx2frame_idx) = video_inputs

        all_vids = []
        all_targets = []
        all_q_input_ids = []
        all_q_attn_masks = []
        all_video_q_inputs = []
        for qid in qids:
            example = self.query_db[qid]
            if example['target'] is not None:
                if example['target']:
                    target = torch.LongTensor([1])
                else:
                    target = torch.LongTensor([0])
            else:
                target = torch.LongTensor([-1])

            curr_q_input_ids = torch.tensor(
                [self.query_db.sep] + example["input_ids"])
            curr_q_attn_masks = torch.tensor([1]*len(curr_q_input_ids))
            all_q_input_ids.append(curr_q_input_ids)
            all_q_attn_masks.append(curr_q_attn_masks)
            f_sub_q_input_ids, f_sub_q_attn_masks = [], []
            sub_qa_attn_masks = []
            for f_sub_input_ids, f_attn_masks, sub_attn_masks in zip(
                    frame_level_input_ids, frame_level_attn_masks,
                    frame_level_sub_attn_masks):
                curr_f_sub_q_input_ids = torch.cat((
                    f_sub_input_ids, curr_q_input_ids))
                curr_f_sub_q_attn_masks = torch.cat((
                    f_attn_masks, curr_q_attn_masks))
                curr_sub_qa_attn_masks = torch.cat(
                    (sub_attn_masks, curr_q_attn_masks))
                f_sub_q_input_ids.append(curr_f_sub_q_input_ids)
                f_sub_q_attn_masks.append(curr_f_sub_q_attn_masks)
                sub_qa_attn_masks.append(curr_sub_qa_attn_masks)
            curr_video_q_inputs = (
                f_sub_q_input_ids, frame_level_v_feats,
                f_sub_q_attn_masks, sub_qa_attn_masks,
                clip_level_v_feats, clip_level_attn_masks, num_subs,
                sub_idx2frame_idx)
            all_video_q_inputs.append(curr_video_q_inputs)
            all_vids.append(vid)
            all_targets.append(target)
        out = (all_video_q_inputs, all_q_input_ids, all_q_attn_masks,
               all_vids, all_targets)
        return out


def violin_collate(inputs):
    (video_q_inputs, q_input_ids, q_attn_masks,
     vids, target) = map(
        list, unzip(inputs))
    all_video_qa_inputs = []
    all_target = []
    all_q_input_ids, all_q_attn_masks = [], []
    for i in range(len(video_q_inputs)):
        all_video_qa_inputs.extend(video_q_inputs[i])
        all_q_input_ids.extend(q_input_ids[i])
        all_q_attn_masks.extend(q_attn_masks[i])
    for j in range(len(vids)):
        all_target.extend(target[j])
    batch = video_collate(all_video_qa_inputs)

    targets = pad_sequence(
        all_target, batch_first=True, padding_value=-1)
    input_ids, pos_ids, attn_masks =\
        txt_input_collate(all_q_input_ids, all_q_attn_masks)
    batch["targets"] = targets
    batch['qa_input_ids'] = input_ids
    batch['qa_pos_ids'] = pos_ids
    batch['qa_attn_masks'] = attn_masks
    return batch


class ViolinEvalDataset(ViolinDataset):
    def getids(self, i):
        if not self.sampled_by_q:
            vid = self.vids[i]
            # TVR video loss assumes fix number of queries
            qids = self.query_db.video2query[vid][:self.max_num_query]
            if len(qids) < self.max_num_query:
                qids += random.sample(qids, self.max_num_query - len(qids))
        else:
            qids = [self.qids[i]]
            vid = self.query_db.query2video[qids[0]]
        return vid, qids

    def __getitem__(self, i):
        vid, qids = self.getids(i)
        outs = super().__getitem__(i)
        return qids, outs


def violin_eval_collate(inputs):
    qids, batch = [], []
    for id_, tensors in inputs:
        qids.extend(id_)
        batch.append(tensors)
    batch = violin_collate(batch)
    batch['qids'] = qids
    return batch


class ViolinVideoOnlyDataset(ViolinDataset):
    def __validate_input_db__(self):
        assert isinstance(self.query_db, QaQueryTokLmdb)
        assert isinstance(self.video_db, VideoFeatDataset)


class ViolinVideoOnlyEvalDataset(ViolinEvalDataset):
    def __validate_input_db__(self):
        assert isinstance(self.query_db, QaQueryTokLmdb)
        assert isinstance(self.video_db, VideoFeatDataset)


class ViolinSubOnlyDataset(ViolinDataset):
    def __validate_input_db__(self):
        assert isinstance(self.query_db, QaQueryTokLmdb)
        assert isinstance(self.video_db, SubOnlyDataset)


class ViolinSubOnlyEvalDataset(ViolinEvalDataset):
    def __validate_input_db__(self):
        assert isinstance(self.query_db, QaQueryTokLmdb)
        assert isinstance(self.video_db, SubOnlyDataset)
