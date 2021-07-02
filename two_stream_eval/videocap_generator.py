"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

copied/modified from HERO
(https://github.com/linjieli222/HERO)
"""
import torch
from model.videoCap import _to_fp16


class VideoCapGenerator(object):
    def __init__(self, model1, max_step, bos, eos, fp16, model2=None):
        self.model1 = model1
        self.model2 = model2
        self.max_step = max_step
        self.bos = bos
        self.eos = eos
        self.fp16 = fp16

    def greedy_decode(self, batch1, batch2=None):
        """
        run greedy decoding
        NOTE: Speed can potentially be improved by keeping past
              decoder hidden states and only run `step-wise` forward.
              Also, maybe can add early stop when all sequences reaches eos
              instead of running until max_step.
        """
        if self.fp16:
            batch1 = _to_fp16(batch1)
            if batch2 is not None:
                batch2 = _to_fp16(batch2)
        encoder_outputs1, enc_mask1 = self.model1.encode(batch1)  # (N, Lv, D)
        if self.fp16:
            encoder_outputs1 = encoder_outputs1.half()
        batch_size = enc_mask1.size(0)
        bos = torch.tensor([self.bos]).expand(batch_size).cuda()
        input_ids = torch.zeros(batch_size, self.max_step).to(bos)
        pos_ids = torch.arange(0, self.max_step+1).unsqueeze(0).cuda()
        last_out = bos

        if batch2 is not None:
            encoder_outputs2, enc_mask2 = self.model2.encode(
                batch2)  # (N, Lv, D)
            if self.fp16:
                encoder_outputs2 = encoder_outputs2.half()
        for step in range(self.max_step):
            input_ids[:, step] = last_out
            score = self.model1.decode(encoder_outputs1, enc_mask1,
                                       input_ids[:, :step+1],
                                       pos_ids[:, :step+1],
                                       None, compute_loss=False)
            if batch2 is not None:
                score2 = self.model2.decode(
                    encoder_outputs2, enc_mask2,
                    input_ids[:, :step+1],
                    pos_ids[:, :step+1],
                    None, compute_loss=False)
                score = score/2. + score2/2.
            output_ids = score.max(dim=-1)[1]
            last_out = output_ids[:, -1]

        outputs = [self.cut_eos(ids) for ids in output_ids.tolist()]
        return outputs

    def cut_eos(self, ids):
        out_ids = []
        for i in ids:
            if i == self.eos:
                break
            out_ids.append(i)
        return out_ids
