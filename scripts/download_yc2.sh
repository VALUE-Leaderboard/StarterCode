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
if [ ! -d $DOWNLOAD/video_db/yc2/ ] ; then
    ~/azcopy/azcopy cp $BLOB/video_db/yc2.tar $DOWNLOAD/video_db/yc2.tar
    tar -xvf $DOWNLOAD/video_db/yc2.tar -C $DOWNLOAD/video_db 
    rm $DOWNLOAD/video_db/yc2.tar
fi

# text dbs
if [ ! -d $DOWNLOAD/txt_db/yc2_subtitles.db/ ] ; then
    wget $BLOB/txt_db/yc2_subtitles.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/yc2_subtitles.db.tar -C $DOWNLOAD/txt_db
    rm $DOWNLOAD/txt_db/yc2_subtitles.db.tar
fi
# yc2r
for SPLIT in 'train' 'val' 'test' ; do
    if [ ! -d $DOWNLOAD/txt_db/yc2r_$SPLIT.db/ ] ; then
        wget $BLOB/txt_db/yc2r_$SPLIT.db.tar -P $DOWNLOAD/txt_db/
        tar -xvf $DOWNLOAD/txt_db/yc2r_$SPLIT.db.tar -C $DOWNLOAD/txt_db
        rm $DOWNLOAD/txt_db/yc2r_$SPLIT.db.tar
    fi
done

# pretrained
if [ ! -f $DOWNLOAD/pretrained/hero-tv-ht100.pt ] ; then
    wget $BLOB/pretrained/hero-tv-ht100.pt -P $DOWNLOAD/pretrained/
fi

BLOB='https://datarelease.blob.core.windows.net/value-leaderboard'
YC2C=$BLOB/'yc2c'
# yc2c raw data (evaluation and inference)
for SPLIT in 'val' 'test' ; do
    wget -nc $YC2C/yc2c_${SPLIT}_release.jsonl -O $DOWNLOAD/txt_db/yc2c_${SPLIT}_release.jsonl
done
