#! /bin/bash

for filename in raw/*; do
    echo $(basename $filename)
    ffmpeg -i $filename -vcodec libx264 -pix_fmt yuv420p -an -f h264 -b 4096k $(basename $filename)".h264"
done
