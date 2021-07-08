# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


DOWNLOAD=$1

for FOLDER in 'video_db' 'txt_db' 'pretrained' 'finetune'; do
    if [ ! -d $DOWNLOAD/$FOLDER ] ; then
        mkdir -p $DOWNLOAD/$FOLDER
    fi
done

BLOB='https://datarelease.blob.core.windows.net/value-leaderboard/starter_code_data'

# Use azcopy for video db downloading
if [ -f ~/azcopy/azcopy ]; then
    echo "azcopy exists, skip downloading"
else 
    echo "azcopy does not exist, start downloading"
    wget -P ~/azcopy/ https://convaisharables.blob.core.windows.net/azcopy/azcopy
fi
chmod +x ~/azcopy/azcopy

# video dbs
if [ ! -d $DOWNLOAD/video_db/vatex/ ] ; then
    ~/azcopy/azcopy cp $BLOB/video_db/vatex.tar $DOWNLOAD/video_db/vatex.tar
    tar -xvf $DOWNLOAD/video_db/vatex.tar -C $DOWNLOAD/video_db
    rm $DOWNLOAD/video_db/vatex.tar
fi

# text dbs
if [ ! -d $DOWNLOAD/txt_db/vatex_subtitles.db/ ] ; then
    wget $BLOB/txt_db/vatex_subtitles.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/vatex_subtitles.db.tar -C $DOWNLOAD/txt_db
    rm $DOWNLOAD/txt_db/vatex_subtitles.db.tar
fi
# vatex_en_r
for SPLIT in 'train' 'val' 'test_public' ; do
    if [ ! -d $DOWNLOAD/txt_db/vatex_en_r_$SPLIT.db/ ] ; then
        wget $BLOB/txt_db/vatex_en_r_$SPLIT.db.tar -P $DOWNLOAD/txt_db/
        tar -xvf $DOWNLOAD/txt_db/vatex_en_r_$SPLIT.db.tar -C $DOWNLOAD/txt_db
        rm $DOWNLOAD/txt_db/vatex_en_r_$SPLIT.db.tar
    fi
done

HEROBLOB='https://convaisharables.blob.core.windows.net/hero'
# pretrained
if [ ! -f $DOWNLOAD/pretrained/hero-tv-ht100.pt ] ; then
    wget $HEROBLOB/pretrained/hero-tv-ht100.pt -P $DOWNLOAD/pretrained/
fi

VATEXCBLOB='https://datarelease.blob.core.windows.net/value-leaderboard/vatex_en_c'
# vatex_en_c raw data (evaluation and inference)
for SPLIT in 'test_public' 'test_private'; do
    wget -nc $VATEXCBLOB/vatex_en_c_${SPLIT}_release.jsonl -O $DOWNLOAD/txt_db/vatex_en_c_${SPLIT}_release.jsonl
done
