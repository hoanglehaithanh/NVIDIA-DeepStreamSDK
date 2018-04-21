#!/bin/bash

LABEL=../data/model/resnet10/labels.txt
MODEL=../data/model/resnet10/resnet10.caffemodel 
DEPLOY=../data/model/resnet10/resnet10.prototxt 
CALIBRATION=../data/model/resnet10/CalibrationTable

CHANNELS=4
FILE_PATH=../data/video/
pushd ${FILE_PATH}
for((i=0;i<${CHANNELS};i++))
do
	file="sample_720p.h264,"
	FILE_LIST=${FILE_LIST}${FILE_PATH}${file}
done
popd
#echo ${FILE_LIST}

# 0: Titan x
DISPLAY_GPU=0
INFER_GPU=0
TILE_WIDTH=480
TILE_HEIGHT=270
TILES_IN_ROW=2

rm -rf log
mkdir log
../bin/sample_smartdetection	-devID_display=${DISPLAY_GPU}			\
						-devID_infer=${INFER_GPU}				\
						-nChannels=${CHANNELS}					\
						-fileList=${FILE_LIST} 					\
						-deployFile=${DEPLOY}					\
						-modelFile=${MODEL}						\
						-labelFile=${LABEL}						\
						-int8=1									\
						-calibrationTableFile=${CALIBRATION}	\
						-tileWidth=${TILE_WIDTH}				\
						-tileHeight=${TILE_HEIGHT}				\
						-tilesInRow=${TILES_IN_ROW}				\
						-fullscreen=0							\
						-endlessLoop=0							
						#2>&1 | tee ./log.txt
