"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

HERO for Video Corpus Moment Retrieval Tasks, shared by:
1. TVR
2. How2R

copied/modified from HERO
(https://github.com/linjieli222/HERO)
"""
from .pretrain import HeroForPretraining
import torch


class HeroForVcmr(HeroForPretraining):
    def __init__(self, config, vfeat_dim, max_frm_seq_len,
                 conv_stride=1, conv_kernel_size=5,
                 ranking_loss_type="hinge", margin=0.1,
                 lw_neg_ctx=0, lw_neg_q=0, lw_st_ed=0.01, drop_svmr_prob=0,
                 use_hard_negative=False, hard_pool_size=20,
                 hard_neg_weight=10, use_all_neg=True):
        super(HeroForVcmr, self).__init__(
            config, vfeat_dim, max_frm_seq_len,
            conv_stride, conv_kernel_size,
            ranking_loss_type, margin,
            lw_neg_ctx, lw_neg_q, lw_st_ed, drop_svmr_prob,
            use_hard_negative, hard_pool_size,
            hard_neg_weight, use_all_neg)
        assert lw_st_ed > 0 or lw_neg_ctx > 0 or lw_neg_q > 0

    def forward(self, batch, task='vcmr', compute_loss=True):
        if task == "vcmr":
            return super(HeroForVcmr, self).forward(
                batch, task='vsm', compute_loss=compute_loss)
        elif task == "vr":
            if compute_loss:
                _, loss_neg_ctx, loss_neg_q = super(HeroForVcmr, self).forward(
                    batch, task='vsm', compute_loss=True)
                return torch.zeros_like(loss_neg_ctx), loss_neg_ctx, loss_neg_q
            else:
                q2video_scores, _, _ = super(HeroForVcmr, self).forward(
                    batch, task='vsm', compute_loss=False)
                return q2video_scores
        else:
            raise ValueError(f'Unrecognized task {task}')

    def get_pred_from_raw_query(self, frame_embeddings, c_attn_masks,
                                query_input_ids, query_pos_ids,
                                query_attn_masks, cross=False,
                                val_gather_gpus=False):
        modularized_query = self.encode_txt_inputs(
                    query_input_ids, query_pos_ids,
                    query_attn_masks, attn_layer=self.q_feat_attn,
                    normalized=False)
        if self.lw_st_ed != 0:
            st_prob, ed_prob = self.get_pred_from_mod_query(
                frame_embeddings, c_attn_masks,
                modularized_query, cross=cross)
        else:
            st_prob, ed_prob = None, None

        if self.lw_neg_ctx != 0 or self.lw_neg_q != 0:
            q2video_scores = self.get_video_level_scores(
                modularized_query, frame_embeddings, c_attn_masks,
                val_gather_gpus)
        else:
            q2video_scores = None
        return q2video_scores, st_prob, ed_prob
