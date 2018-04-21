#!/bin/bash

VAL=../data/video/val_224x224.txt
SYNSET=../data/model/googlenet/synset_words.txt
MODEL=../data/model/googlenet/bvlc_googlenet.caffemodel
DEPLOY=../data/model/googlenet/deploy.prototxt
MEAN=../data/model/googlenet/googlenet_mean.binaryproto

CHANNELS=1
FILE_LIST=
FILE_PATH=../data/video/
for((i=0;i<${CHANNELS};i++))
do
	file="sample_224x224.h264,"
	FILE_LIST=${FILE_LIST}${FILE_PATH}${file}
done

LOGDIR=./log
if [ ! -d ${LOGDIR} ]; then mkdir -p ${LOGDIR}; fi

../bin/sample_classification	-nChannels=${CHANNELS}		\
								-fileList=${FILE_LIST} 		\
								-deployFile=${DEPLOY}		\
								-modelFile=${MODEL}			\
								-meanFile=${MEAN}			\
								-synsetFile=${SYNSET}		\
								-validationFile=${VAL}		\
								-endlessLoop=0				\
								2>&1 | tee ${LOGDIR}/log.txt
