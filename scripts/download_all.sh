# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

# checkpoint
bash ./scripts/download_pretrained.sh $DOWNLOAD

# data
bash ./scripts/download_tvr.sh $DOWNLOAD
bash ./scripts/download_tvqa.sh $DOWNLOAD
bash ./scripts/download_tvc.sh $DOWNLOAD
bash ./scripts/download_how2.sh $DOWNLOAD
bash ./scripts/download_violin.sh $DOWNLOAD
bash ./scripts/download_vlep.sh $DOWNLOAD
bash ./scripts/download_yc2.sh $DOWNLOAD
bash ./scripts/download_vatex.sh $DOWNLOAD