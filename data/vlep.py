"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VLEP dataset

copied/modified from HERO
(https://github.com/linjieli222/HERO)
"""
import torch
from .videoQA import (
    VideoQaDataset, video_qa_eval_collate, video_qa_collate)
from .data import (SubOnlyDataset, VideoFeatDataset, QaQueryTokLmdb)


class VlepDataset(VideoQaDataset):

    def __getitem__(self, i):
        vid, qids = self.getids(i)
        video_inputs = self.video_db.__getitem__(vid)
        (frame_level_input_ids, frame_level_v_feats,
         frame_level_attn_masks, frame_level_sub_attn_masks,
         clip_level_v_feats, clip_level_attn_masks, num_subs,
         sub_idx2frame_idx) = video_inputs
        nframes = len(clip_level_v_feats)

        all_vids = []
        all_targets = []
        all_ts_targets = []
        all_qa_input_ids = []
        all_qa_attn_masks = []
        all_video_qa_inputs = []
        for qid in qids:
            example = self.query_db[qid]
            if example['target'] is not None:
                target = torch.LongTensor([example['target']])
            else:
                target = torch.LongTensor([-1])
            if example['ts'] is not None:
                st_idx, ed_idx = self.get_st_ed_label(
                    example['ts'], max_idx=nframes-1)
                ts_target = torch.LongTensor(
                    [st_idx, ed_idx])
            else:
                ts_target = torch.LongTensor([-1, -1])

            input_ids = example["input_ids"]
            for a_input_ids in input_ids:
                f_sub_qa_input_ids = []
                f_sub_qa_attn_masks = []
                sub_qa_attn_masks = []
                curr_qa_input_id = torch.tensor(
                    [self.query_db.sep] + a_input_ids)
                curr_qa_attn_masks = torch.tensor([1]*len(curr_qa_input_id))
                all_qa_input_ids.append(curr_qa_input_id)
                all_qa_attn_masks.append(curr_qa_attn_masks)
                for f_sub_input_ids, f_attn_masks, sub_attn_masks in zip(
                        frame_level_input_ids, frame_level_attn_masks,
                        frame_level_sub_attn_masks):
                    curr_f_sub_qa_input_ids = torch.cat((
                        f_sub_input_ids, curr_qa_input_id))
                    curr_f_sub_qa_attn_masks = torch.cat((
                        f_attn_masks, curr_qa_attn_masks))
                    curr_sub_qa_attn_masks = torch.cat(
                        (sub_attn_masks, curr_qa_attn_masks))
                    f_sub_qa_input_ids.append(curr_f_sub_qa_input_ids)
                    f_sub_qa_attn_masks.append(curr_f_sub_qa_attn_masks)
                    sub_qa_attn_masks.append(curr_sub_qa_attn_masks)
                curr_video_qa_inputs = (
                    f_sub_qa_input_ids, frame_level_v_feats,
                    f_sub_qa_attn_masks, sub_qa_attn_masks,
                    clip_level_v_feats, clip_level_attn_masks, num_subs,
                    sub_idx2frame_idx)
                all_video_qa_inputs.append(curr_video_qa_inputs)
            all_vids.append(vid)
            all_targets.append(target)
            all_ts_targets.append(ts_target)
        out = (all_video_qa_inputs, all_qa_input_ids, all_qa_attn_masks,
               all_vids, all_targets, all_ts_targets)
        return out


vlep_collate = video_qa_collate


class VlepEvalDataset(VlepDataset):
    def __getitem__(self, i):
        vid, qids = self.getids(i)
        outs = super().__getitem__(i)
        return qids, outs


vlep_eval_collate = video_qa_eval_collate


class VlepVideoOnlyDataset(VlepDataset):
    def __validate_input_db__(self):
        assert isinstance(self.query_db, QaQueryTokLmdb)
        assert isinstance(self.video_db, VideoFeatDataset)


class VlepVideoOnlyEvalDataset(VlepVideoOnlyDataset):
    def __getitem__(self, i):
        vid, qids = self.getids(i)
        outs = super().__getitem__(i)
        return qids, outs


class VlepSubOnlyDataset(VlepDataset):
    def __validate_input_db__(self):
        assert isinstance(self.query_db, QaQueryTokLmdb)
        assert isinstance(self.video_db, SubOnlyDataset)


class VlepSubOnlyEvalDataset(VlepSubOnlyDataset):
    def __getitem__(self, i):
        vid, qids = self.getids(i)
        outs = super().__getitem__(i)
        return qids, outs
