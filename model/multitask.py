"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

HERO for Multi-Task
"""
from .vcmr import HeroForVcmr

from collections import defaultdict
import copy

import torch
from torch import nn
from torch.nn import functional as F
from .layers import MLPLayer
from .modeling_utils import mask_logits

from .videoCap import LabelSmoothingLoss, BertDecoder
from .layers import BertLayerNorm


class HeroForMultiTask(HeroForVcmr):
    def __init__(self, config, vfeat_dim, max_frm_seq_len,
                 conv_stride=1, conv_kernel_size=5,
                 ranking_loss_type="hinge", margin=0.1,
                 lw_neg_ctx=0, lw_neg_q=0, lw_st_ed=0.01, drop_svmr_prob=0,
                 use_hard_negative=False, hard_pool_size=20,
                 hard_neg_weight=10, use_all_neg=True, lsr=0.1):
        super(HeroForMultiTask, self).__init__(
            config, vfeat_dim, max_frm_seq_len,
            conv_stride, conv_kernel_size,
            ranking_loss_type, margin,
            lw_neg_ctx, lw_neg_q, lw_st_ed, drop_svmr_prob,
            use_hard_negative, hard_pool_size,
            hard_neg_weight, use_all_neg)

        # QA head
        hsz = config.c_config.hidden_size

        self.qa_pool = nn.Linear(
            in_features=hsz, out_features=1, bias=False)
        self.qa_pred_head = MLPLayer(hsz, 1)

        # in tvqa/how2qa, we also have annotations for st and ed frame idx
        self.st_ed_pool = copy.deepcopy(self.qa_pool)
        self.st_ed_pred_head = MLPLayer(hsz, 2)

        # Caption head
        self.position_embeddings = nn.Embedding(
            config.d_config.max_position_embeddings,
            config.d_config.hidden_size)
        self.emb_LayerNorm = BertLayerNorm(
            config.d_config.hidden_size, eps=1e-5)
        self.decoder = BertDecoder(config.d_config)

        if lsr > 0:
            self.caption_loss_func = LabelSmoothingLoss(
                lsr, config.f_config.vocab_size,
                ignore_index=-1, reduction='none')
        else:
            self.caption_loss_func = nn.CrossEntropyLoss(
                ignore_index=-1, reduction='none')

    def encode(self, batch):
        if 'cap_attn_mask' in batch:
            frame_embeddings = self.v_encoder(batch, 'repr')
            # pick video segments with associated captions
            segment_embeddings = [
                frame_embeddings[i, st:ed, :]
                for i, segs in enumerate(batch['clip_ranges'])
                for st, ed in segs]

            def pad_tensors(ts):
                """ pad segmet embeddings """
                bs = len(ts)
                max_l = max(t.size(0) for t in ts)
                hid = ts[0].size(1)
                output = torch.zeros(bs, max_l, hid).to(ts[0])
                for i, t in enumerate(ts):
                    len_ = t.size(0)
                    output[i, :len_, :] = t
                return output

            encoder_outputs = pad_tensors(segment_embeddings)

            attn_mask = batch['cap_attn_mask']
        else:
            # captioning on the whole video
            encoder_outputs = self.v_encoder(batch, 'repr')
            attn_mask = batch['c_attn_masks']
        return encoder_outputs, attn_mask

    def decode(self, encoder_outputs, encoder_masks,
               caption_ids, pos_ids, label_ids, compute_loss=True):
        """
        Args:
            text_input_ids: (N, Lt)
            text_masks: (N, Lt)  with 1 indicates valid bits
            text_input_labels: (N, Lt)  with `-1` on ignored positions
            encoder_outputs: (N, Lctx, D)
            encoder_masks: (N, Lctx)
        """
        # shared embedding layer
        text_embeddings = self.v_encoder.f_encoder.embeddings.word_embeddings(
            caption_ids)
        pos_embeddings = self.position_embeddings(pos_ids)
        embeddings = self.emb_LayerNorm(text_embeddings + pos_embeddings)
        decoder_outputs = self.decoder(
            embeddings, encoder_outputs, encoder_masks)[-1]  # (N, Lt, D)
        # shared projection layer
        prediction_scores = self.v_encoder.f_encoder.lm_head(
            decoder_outputs)  # (N, Lt, vocab_size)
        if compute_loss:
            caption_loss = self.caption_loss_func(
                prediction_scores.view(-1, self.config.f_config.vocab_size),
                label_ids.view(-1))
            return caption_loss
        else:
            return prediction_scores

    def get_modularized_video_for_qa(
            self, frame_embeddings, frame_mask, violin=False):
        """
        Args:
            frame_embeddings: (Nv, Nq, L, D)
            frame_mask: (Nv, Nq, L)
        """
        if not violin:
            st_ed_attn_scores = self.st_ed_pool(
                frame_embeddings)  # (Nv, Nq, L, 1)
            qa_attn_scores = self.qa_pool(frame_embeddings)

            st_ed_attn_scores = F.softmax(
                mask_logits(st_ed_attn_scores,
                            frame_mask.unsqueeze(-1)), dim=1)
            qa_attn_scores = F.softmax(
                mask_logits(qa_attn_scores,
                            frame_mask.unsqueeze(-1)), dim=2)
            # TODO check whether it is the same
            st_ed_pooled_video = torch.einsum(
                "vqlm,vqld->vlmd", st_ed_attn_scores,
                frame_embeddings)  # (Nv, L, 1, D)
            qa_pooled_video = torch.einsum(
                "vqlm,vqld->vqmd", qa_attn_scores,
                frame_embeddings)  # (Nv, Nq, 1, D)
            return st_ed_pooled_video.squeeze(2), qa_pooled_video.squeeze(2)
        else:
            violin_attn_scores = self.qa_pool(
                frame_embeddings)  # (Nv, L, 1)

            violin_attn_scores = F.softmax(
                mask_logits(violin_attn_scores,
                            frame_mask.unsqueeze(-1)), dim=1)

            # TODO check whether it is the same
            violin_pooled_video = torch.einsum(
                "vlm,vld->vmd", violin_attn_scores,
                frame_embeddings)  # (Nv, 1, D)
            return violin_pooled_video.squeeze(1)

    def forward_qa(self, batch, task='videoQA', compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task == "videoQA" or task == "violin":
            targets = batch['targets'].squeeze(-1)
            c_attn_masks = batch["c_attn_masks"]

            # (num_video * 5, num_frames, hid_size)
            frame_embeddings = self.v_encoder.forward_repr(
                batch, encode_clip=False)
            frame_embeddings = self.v_encoder.c_encoder.embeddings(
                frame_embeddings,
                position_ids=None)
            qa_embeddings = self.v_encoder.f_encoder._compute_txt_embeddings(
                batch["qa_input_ids"], batch["qa_pos_ids"], txt_type_ids=None)
            frame_qa_embeddings = torch.cat(
                (frame_embeddings, qa_embeddings), dim=1)
            frame_qa_attn_mask = torch.cat(
                (c_attn_masks, batch["qa_attn_masks"]), dim=1)
            fused_video_qa = self.v_encoder.c_encoder.forward_encoder(
                frame_qa_embeddings, frame_qa_attn_mask)
            num_frames = c_attn_masks.shape[1]
            video_embeddings = fused_video_qa[:, :num_frames, :]

            if task == "videoQA":
                num_videos = len(targets)
                num_frames, hid_size = video_embeddings.shape[1:3]
                video_embeddings = video_embeddings.view(
                    num_videos, -1, num_frames, hid_size)
                video_masks = c_attn_masks.view(num_videos, -1, num_frames)
                video_masks = video_masks.to(dtype=video_embeddings.dtype)
                (st_ed_pooled_video, qa_pooled_video
                 ) = self.get_modularized_video_for_qa(
                    video_embeddings, video_masks)
                pred_st_ed = self.st_ed_pred_head(st_ed_pooled_video)
                st_prob = mask_logits(pred_st_ed[:, :, 0], video_masks[:, 0])
                ed_prob = mask_logits(pred_st_ed[:, :, 1], video_masks[:, 0])
                logits = self.qa_pred_head(qa_pooled_video).squeeze(-1)
            else:
                video_masks = c_attn_masks.to(dtype=video_embeddings.dtype)
                qa_pooled_video = self.get_modularized_video_for_qa(
                    video_embeddings, video_masks, violin=True)
                logits = self.qa_pred_head(qa_pooled_video)

            if compute_loss:
                if task == "videoQA":
                    ts_targets = batch["ts_targets"]
                    st_target, ed_target = ts_targets[:, 0], ts_targets[:, 1]
                    st_loss = F.cross_entropy(
                        st_prob, st_target, reduction="mean",
                        ignore_index=-1)
                    ed_loss = F.cross_entropy(
                        ed_prob, ed_target, reduction="mean",
                        ignore_index=-1)
                    temporal_loss = (st_loss + ed_loss)/2.
                    qa_loss = F.cross_entropy(logits, targets,
                                              reduction='mean',
                                              ignore_index=-1)
                    return qa_loss, temporal_loss
                else:
                    targets = batch['targets']
                    scores = torch.sigmoid(logits).squeeze(-1)
                    targets = targets.squeeze(-1).to(dtype=scores.dtype)
                    qa_loss = F.binary_cross_entropy(
                        scores, targets, reduction='mean')
                    return qa_loss
            else:
                return logits
        else:
            raise ValueError(f'Unrecognized task: {task}')

    def forward(self, batch, task='vcmr', compute_loss=True):
        if task in ["vcmr", "vr"]:
            return super().forward(
                batch, task=task, compute_loss=compute_loss)
        elif task in ["videoQA", "violin"]:
            return self.forward_qa(batch, task, compute_loss)
        elif task == "videoCap":
            encoder_outputs, attn_mask = self.encode(batch)  # (N, Lv, D)
            caption_ids = batch['cap_input_ids']
            pos_ids = batch['cap_pos_ids']
            label_ids = batch['cap_tgt_ids']
            res = self.decode(encoder_outputs, attn_mask,
                              caption_ids, pos_ids, label_ids, compute_loss)
            return res
        else:
            raise ValueError(f'Unrecognized task {task}')
