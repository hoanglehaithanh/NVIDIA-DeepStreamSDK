# NVIDIA DeepStream SDK 1.5

> THIS IS AN OLD VERSION OF DEEPSTREAM SDK, WHICH USES DIFFERENT ARCHITECTURE FROM LATER VERSIONS. THIS VERSION IS EASILY TO USE BUT ONLY USES TENSORRT 3 AND DOES NOT SUPPORT MANY LASTEST DEEPLEARNING MODELS.

## 1.INTRODUCTION

![](https://cdn-images-1.medium.com/max/2000/1*VNCy5iIQtBS2qO_RBQWpvw.png)

DeepStream provides an easy-to-use and high-performance SDK for video content analysis, which simplifies development of high-performance video analytics applications powered by deep learning. DeepStream enables customer to make optimum use of underlying GPU architectures, including hardware decoding support, thereby achieving high levels of efficiency, performance, and scale. Furthermore, DeepStream provides a flexible plug-in mechanism for the user to incorporate their custom functionality to video analytics applications to meet their unique needs.

DeepStream provides a high-level C++ API for GPU-accelerated video decoding, inference. DeepStream is built on the top of NVCODEC and NVIDIA® TensorRT™, which are responsible for video decoding and deep learning inference, respectively.

The following are the key features of DeepStream:

- Deploys widely-used neural network models such as GoogleNet, AlexNet, etc. for real-time image classification and object detection.
- Supports neural networks implemented using Caffe and TensorFlow frameworks.
- Supports common video formats: H.264, HEVC/H.265, MPEG-2, MPEG-4, VP9, and VC11.
- Takes inference with full precision float type (FP32) or optimized precision2 (FP16 and INT8).
- Provides flexible analytics workflow which allows users to implement a plug-in to define their inference workflow.

## 2.SYSTEM REQUIREMENTS

DeepStream has the following software dependencies:
- Ubuntu 16.04 LTS (with GCC 5.4)
- NVIDIA Display Driver R384
- NVIDIA VideoSDK 8.0
- NVIDIA CUDA® 9.0
- cuDNN 7 & TensorRT 3.0

NVIDIA recommends that DeepStream be run on a hardware platform with an NVIDIA Tesla® P4 or P40 graphics card.

## 3.DIRECTORY LAYOUT
The DeepStream SDK consists of two main parts: the library and the workflow demonstration samples. The installed DeepStream package includes the directories /lib, /include, /doc, and /samples.
- The dynamic library libdeepstream.so is in the /lib directory.
- There are two header files: deepStream.h and module.h.
  - deepStream.h includes the definition of decoded output, supported data type,
inference parameters, profiler class, and DeepStream worker, as well as the
declaration of functions.Installation
DeepStream SDK DU-08633-001_v04| 4
  - module.h is the header file for plug-in implementation. This file is not mandatory
for applications without plug-ins.
- The /samples folder includes examples of decoding, decoding and inference, and
plug-in implementations. More information can be found in the Samples chap

## 4.CUSTOM HARDWARE

## 5.CUSTOM IMPLEMENTATION
In Progress

## 6.LICENSE
[NVIDIA DeepStream SDK License Agreement](LicenseAgreement.pdf)
