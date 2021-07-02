"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

constants

copied/modified from HERO
(https://github.com/linjieli222/HERO)
"""
VFEAT_DIM = {"resnet_slowfast": 4352, "resnet_mil-nce": 3072,
             "clip-vit_slowfast": 2816,
             "clip-vit_mil-nce": 1536,
             "resnet": 2048,
             "slowfast": 2304,
             "clip-vit": 512,
             "mil-nce": 1024}
# VFEAT_DIM = 4352
MAX_FRM_SEQ_LEN = 100
VCMR_IOU_THDS = (0.5, 0.7)
