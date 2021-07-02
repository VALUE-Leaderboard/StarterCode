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
if [ ! -d $DOWNLOAD/video_db/how2/ ] ; then
    ~/azcopy/azcopy cp $BLOB/video_db/how2.tar $DOWNLOAD/video_db/how2.tar
    tar -xvf $DOWNLOAD/video_db/how2.tar -C $DOWNLOAD/video_db 
    rm $DOWNLOAD/video_db/how2.tar
fi

# text dbs
if [ ! -d $DOWNLOAD/txt_db/how2_subtitles.db/ ] ; then
    wget $BLOB/txt_db/how2_subtitles.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/how2_subtitles.db.tar -C $DOWNLOAD/txt_db
    rm $DOWNLOAD/txt_db/how2_subtitles.db.tar
fi
# how2r
for SPLIT in 'train' 'val_1k' 'test_public_1k' ; do
    if [ ! -d $DOWNLOAD/txt_db/how2r_$SPLIT.db/ ] ; then
        wget $BLOB/txt_db/how2r_$SPLIT.db.tar -P $DOWNLOAD/txt_db/
        tar -xvf $DOWNLOAD/txt_db/how2r_$SPLIT.db.tar -C $DOWNLOAD/txt_db
        rm $DOWNLOAD/txt_db/how2r_$SPLIT.db.tar
    fi
done
# how2qa
for SPLIT in 'train' 'val' 'test_public' ; do
    if [ ! -d $DOWNLOAD/txt_db/how2qa_$SPLIT.db/ ] ; then
        wget $BLOB/txt_db/how2qa_$SPLIT.db.tar -P $DOWNLOAD/txt_db/
        tar -xvf $DOWNLOAD/txt_db/how2qa_$SPLIT.db.tar -C $DOWNLOAD/txt_db
        rm $DOWNLOAD/txt_db/how2qa_$SPLIT.db.tar
    fi
done

HEROBLOB='https://convaisharables.blob.core.windows.net/hero'
# pretrained
if [ ! -f $DOWNLOAD/pretrained/hero-tv-ht100.pt ] ; then
    wget $HEROBLOB/pretrained/hero-tv-ht100.pt -P $DOWNLOAD/pretrained/
fi
