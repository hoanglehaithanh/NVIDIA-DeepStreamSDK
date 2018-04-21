.SUFFIXES:

#################################################
# Note: DeepStream SDK is based on the following
# libraries, please download them before building.
# Software		|		version
#------------------------------------------------
# CUDA			|		9.0
#------------------------------------------------
# cuDNN			|		7.0.4
#------------------------------------------------
# TensorRT		|		3.0
#------------------------------------------------
# VideoSDK		|		8.0.14
#------------------------------------------------
# GPU driver	        |		384
#################################################

TARGET:=$(shell uname -s)
ifeq ($(TARGET),Linux)
  CUDA ?= cuda-9.0
  CUDAVERSION := $(CUDA)
  CUDA_INSTALL_LINUX=/usr/local/$(CUDA)
endif

# CUDA v9.0
CUDA_INC_PATH = $(CUDA_INSTALL_LINUX)/include
CUDA_LIB_PATH = $(CUDA_INSTALL_LINUX)/lib64
# TensorRT v3.0
TENSORRT_INC_PATH = /usr/include/x86_64-linux-gnu
TENSORRT_LIB_PATH = /usr/lib/x86_64-linux-gnu
# cuDNN v7.0.4
CUDNN_INC_PATH = /usr/include/x86_64-linux-gnu
CUDNN_LIB_PATH = /usr/lib/x86_64-linux-gnu
# Video SDK v8.0.14
VIDEOSDK_INSTALL_PATH = /usr/local/Video_Codec_SDK_8.0.14
# Nvidia display Driver
NVIDIA_DISPLAY_DRIVER_PATH = /usr/lib/nvidia-384
# cuvid library
CUVID_INC_PATH = $(VIDEOSDK_INSTALL_PATH)"/Samples/common/inc"
CUVID_LIB_PATH = $(NVIDIA_DISPLAY_DRIVER_PATH)

UNAME_S := $(shell uname -s)

DEBUG_MODE ?= 0

ifdef VERBOSE
AT=
else
AT=@
endif

AR = ar cr
ECHO = @echo
CC = g++

################
ifeq ($(TARGET),Linux)
  CC = g++
  ifeq ($(DVS),1)
    CUCC ?= /dvs/p4/build/sw/rel/gpgpu/toolkit/r9.0/bin/x86_64_Linux_release/nvcc
    LIB_RPATH=-Wl,-rpath,$(ROOT_PATH)/ \
              -Wl,-rpath,$(ROOT_PATH)/externals/nvidia/lib  \
              -Wl,-rpath,/usr/local/$(CUDA)/lib64/ \
              -Wl,-rpath,/usr/lib 
  else
    CUCC ?= $(CUDA_INSTALL_LINUX)"/bin/nvcc"
  endif
endif

define concat
$1$2$3$4$5$6$7$8
endef

#$(call make-depend,source-file,object-file,depend-file)
define make-depend
  $(AT)$(CC) -MM -MF $3 -MP -MT $2 $(COMMON_FLAGS) $1
endef
