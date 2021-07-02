"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Youcook2 Captioning evaluation
"""
from collections import defaultdict
import json

from .pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.cider.cider import Cider
from .pycocoevalcap.meteor.meteor import Meteor
from .pycocoevalcap.rouge.rouge import Rouge


def _remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


class Yc2cEval(object):
    """ preload evaluation tools and references for repeated evaluation
        include Micro, Macro, and Paragraph-level evaluation
    """
    def __init__(self, ref_path):
        self.tokenizer = PTBTokenizer()

        id2refs = {ex['clip_id']: [_remove_nonascii(ex['descs'][0]['desc'].strip())]
                   for ex in map(json.loads, open(ref_path))}
        self.cid2refs = self.tokenizer.tokenize(id2refs)

        # group by video
        self.cid2cname = {ex['clip_id']: ex['vid_name']
                        for ex in map(json.loads, open(ref_path))}
        self.cid2vid = {ex['clip_id']: "_".join(ex['vid_name'].split("_")[:-1])
                        for ex in map(json.loads, open(ref_path))}
        self.vid2id2refs = defaultdict(dict)
        for cid, refs in self.cid2refs.items():
            vid = self.cid2vid[cid]
            self.vid2id2refs[vid][cid] = refs
        # refs to compute paragraph-level scores
        self.vid2refs = {
            vid: [' '.join(refs[0] for _, refs in sorted(id2refs.items(),
                                                         key=self._sort_fn))]
            for vid, id2refs in self.vid2id2refs.items()}

        self.scorers = []
        self.scorers.append((Bleu(4),
                             ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        self.scorers.append((Meteor(), "METEOR"))
        self.scorers.append((Rouge(), "ROUGE_L"))
        self.scorers.append((Cider(), "CIDEr"))

    def _sort_fn(self, id_ref):
        clip_id, ref = id_ref
        return int(self.cid2cname[clip_id].split('_')[-1])

    def __call__(self, json_res):
        """ corpus level metrics, take list of results """
        # micro-level scores
        cid2hyps = {
            res['clip_id']: [_remove_nonascii(res['descs'][0]['desc'].strip())]
            for res in json_res
        }
        cid2hyps = self.tokenizer.tokenize(cid2hyps)
        assert len(cid2hyps) == len(self.cid2refs)

        micro_scores = {}
        for scorer, method in self.scorers:
            print(f"Computing paragraph {method} score...")
            score, scores = scorer.compute_score(self.cid2refs, cid2hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    micro_scores[m] = sc * 100
            else:
                micro_scores[method] = score * 100

        # macro-level scores

        vid2id2hyps = defaultdict(dict)
        for cid, refs in cid2hyps.items():
            vid = self.cid2vid[cid]
            vid2id2hyps[vid][cid] = refs

        assert len(vid2id2hyps) == len(self.vid2id2refs)

        macro_scores = {}
        for scorer, method in self.scorers:
            print(f"Computing macro {method} score...")

            all_scores = []
            for vid, id2refs in self.vid2id2refs.items():
                id2hyps = vid2id2hyps[vid]
                assert len(id2hyps) == len(id2refs)
                score, _ = scorer.compute_score(id2refs, id2hyps)
                all_scores.append(score)

                if isinstance(method, list):
                    for i, m in enumerate(method):
                        sc = sum(s[i] for s in all_scores) / len(all_scores)
                        macro_scores[m] = sc * 100
                else:
                    score = sum(all_scores) / len(all_scores)
                    macro_scores[method] = score * 100

        # compute paragraph-level scores
        vid2hyps = {
            vid: [' '.join(hyps[0] for _, hyps in sorted(id2hyps.items(),
                                                         key=self._sort_fn))]
            for vid, id2hyps in vid2id2hyps.items()}
        assert len(vid2hyps) == len(self.vid2refs)

        par_scores = {}
        for scorer, method in self.scorers:
            print(f"Computing paragraph {method} score...")
            score, scores = scorer.compute_score(self.vid2refs, vid2hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    par_scores[m] = sc * 100
            else:
                par_scores[method] = score * 100

        return {'micro': micro_scores,
                'macro': macro_scores,
                'paragraph': par_scores}


# Micro Evaluation
class Yc2cEvalMicro(object):
    """ preload evaluation tools and references for repeated evaluation """
    def __init__(self, ref_path):
        self.tokenizer = PTBTokenizer()
        id2refs = {ex['clip_id']: [_remove_nonascii(ex['descs'][0]['desc'].strip())]
                   for ex in map(json.loads, open(ref_path))}
        self.id2refs = self.tokenizer.tokenize(id2refs)
        self.scorers = []
        self.scorers.append((Bleu(4),
                             ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        self.scorers.append((Meteor(), "METEOR"))
        self.scorers.append((Rouge(), "ROUGE_L"))
        self.scorers.append((Cider(), "CIDEr"))

    def __call__(self, json_res):
        """ corpus level metrics, take list of results """
        id2hyps = {
            res['clip_id']:
            [_remove_nonascii(res['descs'][0]['desc'].strip())]
            for res in json_res}
        id2hyps = self.tokenizer.tokenize(id2hyps)
        assert len(id2hyps) == len(self.id2refs)

        ret_scores = {}
        for scorer, method in self.scorers:
            print(f"Computing {method} score...")
            score, scores = scorer.compute_score(self.id2refs, id2hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    ret_scores[m] = sc * 100
            else:
                ret_scores[method] = score * 100

        return ret_scores


# Macro Evaluation
class Yc2cEvalMacro(object):
    """ preload evaluation tools and references for repeated evaluation """
    def __init__(self, ref_path):
        self.tokenizer = PTBTokenizer()

        id2refs = {ex['clip_id']: [_remove_nonascii(ex['descs'][0]['desc'].strip())]
                   for ex in map(json.loads, open(ref_path))}
        cid2refs = self.tokenizer.tokenize(id2refs)

        # group by video
        self.cid2vid = {ex['clip_id']: "_".join(ex['vid_name'].split("_")[:-1])
                        for ex in map(json.loads, open(ref_path))}
        self.vid2id2refs = defaultdict(dict)
        for cid, refs in cid2refs.items():
            vid = self.cid2vid[cid]
            self.vid2id2refs[vid][cid] = refs

        self.scorers = []
        self.scorers.append((Bleu(4),
                             ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        self.scorers.append((Meteor(), "METEOR"))
        self.scorers.append((Rouge(), "ROUGE_L"))
        self.scorers.append((Cider(), "CIDEr"))

    def __call__(self, json_res):
        """ corpus level metrics, take list of results """

        cid2hyps = {
            res['clip_id']: [_remove_nonascii(res['descs'][0]['desc'].strip())]
            for res in json_res
        }
        cid2hyps = self.tokenizer.tokenize(cid2hyps)

        vid2id2hyps = defaultdict(dict)
        for cid, refs in cid2hyps.items():
            vid = self.cid2vid[cid]
            vid2id2hyps[vid][cid] = refs

        assert len(vid2id2hyps) == len(self.vid2id2refs)

        ret_scores = {}
        for scorer, method in self.scorers:
            print(f"Computing {method} score...")

            all_scores = []
            for vid, id2refs in self.vid2id2refs.items():
                id2hyps = vid2id2hyps[vid]
                assert len(id2hyps) == len(id2refs)
                score, _ = scorer.compute_score(id2refs, id2hyps)
                all_scores.append(score)

            if isinstance(method, list):
                for i, m in enumerate(method):
                    sc = sum(s[i] for s in all_scores) / len(all_scores)
                    ret_scores[m] = sc * 100
            else:
                score = sum(all_scores) / len(all_scores)
                ret_scores[method] = score * 100

        return ret_scores


# paragraph Evaluation
# TODO Repetition@4
class Yc2cEvalParagraph(object):
    """ preload evaluation tools and references for repeated evaluation """
    def __init__(self, ref_path):
        self.tokenizer = PTBTokenizer()
        vid2exs = defaultdict(list)
        for ex in map(json.loads, open(ref_path)):
            curr_video_name = "_".join(ex['vid_name'].split("_")[:-1])
            vid2exs[ex[curr_video_name]].append(ex)
        for exs in vid2exs.values():
            exs.sort(key=lambda ex: int(ex['vid_name'].split('_')[-1]))

        id2refs = {vid: [_remove_nonascii(ex['descs'][0]['desc'].strip()) for ex in exs]
                   for vid, exs in vid2exs.items()}
        # concat all captions
        self.id2refs = {
            i: [' '.join(refs)]  # only 1 ref
            for i, refs in self.tokenizer.tokenize(id2refs).items()}

        self.scorers = []
        self.scorers.append((Bleu(4),
                             ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        self.scorers.append((Meteor(), "METEOR"))
        self.scorers.append((Rouge(), "ROUGE_L"))
        self.scorers.append((Cider(), "CIDEr"))

    def __call__(self, json_res):
        """ corpus level metrics, take list of results """
        vid2results = defaultdict(list)
        for res in json_res:
            vid2results[res['video_id']].append(res)
        for results in vid2results.values():
            results.sort(key=lambda res: int(res['video_id'].split('_')[-1]))

        id2hyps = {
            vid: [_remove_nonascii(res['descs'][0]['desc'].strip())
                  for res in results]
            for vid, results in vid2results.items()
        }
        # concat all captions
        id2hyps = {i: [' '.join(hyps)]
                   for i, hyps in self.tokenizer.tokenize(id2hyps).items()}
        assert len(id2hyps) == len(self.id2refs)

        ret_scores = {}
        for scorer, method in self.scorers:
            print(f"Computing {method} score...")
            score, scores = scorer.compute_score(self.id2refs, id2hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    ret_scores[m] = sc * 100
            else:
                ret_scores[method] = score * 100

        return ret_scores
