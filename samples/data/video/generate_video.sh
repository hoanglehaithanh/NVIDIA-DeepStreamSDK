#! /bin/bash
iv=10

rm -f val_224x224.txt
touch val_224x224.txt
iv=10
#first download images
for cropline in $(cat image_urls.txt)
do
   echo $cropline
   wnid=$(echo $cropline | cut -f2 -d',')
   imgurl=$(echo $cropline | cut -f1 -d',')
   echo $wnid
   wget --tries=1 --timeout=10  $imgurl -O $iv".jpg"
   ft=$(file $iv".jpg" | cut -f2 -d' ')
   echo "file type"
   echo $ft
   if [ "$ft" = "JPEG" ]
    then
      #for valid images create an entry in validation file
      echo $wnid >> "val_224x224.txt"
    fi
   iv=$((iv+1))
done


rm -rf rescale
mkdir rescale
mv *.jpg rescale
mv *.JPG rescale
cd rescale



for filename in *; do
    ft=$(file $filename | cut -f2 -d' ')
    if [ "$ft" = "JPEG" ]
    then
      #rescale to 224x224
      ffmpeg  -i $filename -vf scale=224:224 "rescale_"$filename
    fi
done

#stitch to create mp4
ffmpeg -framerate 25 -pattern_type glob -i "rescale*.jpg"  -vf scale=224x224,setdar=1:1  out.mp4

#create raw h264 video from mp4
ffmpeg -i out.mp4 -vcodec libx264 -pix_fmt yuv420p -an -f h264 -b 4096k sample_224x224.h264
mv sample_224x224.h264 ..
cd ..
#rm -rf rescale
