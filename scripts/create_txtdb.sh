# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


TXT_DB=$1
ANN_DIR=$2
VIDEO_DB=$3

set -e

# annotations
DataBLOB='https://datarelease.blob.core.windows.net/value-leaderboard/tv_tasks'
TVR='https://raw.githubusercontent.com/jayleicn/TVRetrieval/master/data/'

if [ ! -d $TXT_DB ]; then
    mkdir -p $TXT_DB
fi
if [ ! -d $ANN_DIR ]; then
    mkdir -p $ANN_DIR
fi


for SPLIT in 'train' 'val'; do
    if [ ! -f $ANN_DIR/tvr_${SPLIT}_release.jsonl ]; then
        echo "downloading ${SPLIT} annotations..."
        wget $TVR/tvr_${SPLIT}_release.jsonl -O $ANN_DIR/tvr_${SPLIT}_release.jsonl
    fi
done
if [ ! -f $ANN_DIR/tvr_test_release.jsonl ]; then
    echo "downloading test annotations..."
    wget $DataBLOB/tvr_test_release.jsonl -O $ANN_DIR/tvr_test_release.jsonl
fi

for SPLIT in 'train' 'val' 'test'; do
    if [ ! -d $TXT_DB/tvr_${SPLIT}.db ]; then
        echo "preprocessing tvr ${SPLIT} annotations..."
        docker run --ipc=host --rm -it \
            --mount src=$(pwd),dst=/src,type=bind \
            --mount src=$TXT_DB,dst=/txt_db,type=bind \
            --mount src=$ANN_DIR,dst=/ann,type=bind,readonly \
            -w /src linjieli222/hero \
            python scripts/prepro_query.py --annotation /ann/tvr_${SPLIT}_release.jsonl \
                            --output /txt_db/tvr_${SPLIT}.db \
                            --task tvr
    fi
done



if [ ! -d $VIDEO_DB ]; then
    echo "Make sure you have constructed/downloaded the video dbs before processing the subtitles..."
else
    if [ ! -f $ANN_DIR/tv_subtitles.jsonl ]; then
        echo "downloading raw subtitle and additional annotations..."
        wget  $DataBLOB/tvr_video2dur_idx.json -O $ANN_DIR/vid2dur_idx.json

        wget $TVR/tvqa_preprocessed_subtitles.jsonl -O $ANN_DIR/tv_subtitles.jsonl
    fi

    if [ ! -d $TXT_DB/tv_subtitles.db ]; then
        echo "preprocessing tv subtitles..."
        docker run --ipc=host --rm -it \
            --mount src=$(pwd),dst=/src,type=bind \
            --mount src=$TXT_DB,dst=/txt_db,type=bind \
            --mount src=$ANN_DIR,dst=/ann,type=bind,readonly \
            --mount src=$VIDEO_DB,dst=/video_db,type=bind,readonly \
            -w /src linjieli222/hero \
            /bin/bash -c "python scripts/prepro_sub.py --annotation /ann/tv_subtitles.jsonl --output /txt_db/tv_subtitles.db --vid2nframe /video_db/tv/id2nframe_1.5.json --frame_length 1.5; cp /ann/vid2dur_idx.json /txt_db/tv_subtitles.db/"
        echo "done"
    fi
fi