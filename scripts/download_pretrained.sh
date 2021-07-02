# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


DOWNLOAD=$1

if [ ! -d $DOWNLOAD/pretrained ] ; then
    mkdir -p $DOWNLOAD/pretrained
fi

HEROBLOB='https://convaisharables.blob.core.windows.net/hero'

# This will overwrite models
wget $HEROBLOB/pretrained/hero-tv-ht100.pt -O $DOWNLOAD/pretrained/hero-tv-ht100.pt

# converted RoBERTa
if [ ! -f $DOWNLOAD/pretrained/pretrain-tv-init.bin ] ; then
    wget $HEROBLOB/pretrained/pretrain-tv-init.bin -O $DOWNLOAD/pretrained/pretrain-tv-init.bin
fi