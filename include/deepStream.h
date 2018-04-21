/*
* Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/
#ifndef DEEPSTREAM_H
#define DEEPSTREAM_H

#include <iostream>
#include <vector>
#include <cstdint>
// CUDA toolkit
#include <cuda_runtime.h>
#include <cuda.h>
// Video SDK
#include <dynlink_nvcuvid.h>

#include "module.h"

/** 
 * \file deepSteam.h
 *
 * This is the top-level API file for deepStream.
 */

/**
 * \struct DEC_OUTPUT
 *
 * \brief the output of decode.
 * The cuda stream is created with flag cudaStreamNonBlocking,
 * and should perform no implicit synchronization with stream 0.
 *
 */
typedef struct {
	int frameIndex_;					//!< Frame index
	int videoIndex_;					//!< Video index
	int nWidthPixels_;					//!< Frame width
	int nHeightPixels_;					//!< Frame height
	uint8_t *dpFrame_;					//!< Frame data (nv12 format)
	size_t frameSize_;					//!< Frame size in bytes
	cudaStream_t stream_;				//!< CUDA stream
} DEC_OUTPUT;

/**
 * \brief The callback function of decoding.
 *
 * The function need to be defined by user, and will be called after decode
 * 
 */
typedef void (*DECODER_CALLBACK)(void *pUserData, DEC_OUTPUT *decOutput);

/**
 * \enum DATA_TYPE
 *
 * \brief The type of weights and tensors.
 *
 */
typedef enum {
	FLOAT = 0,			//!< Float type (fp32)
	HALF = 1,			//!< Half  type (fp16)
	INT8 = 2			//!< INT8  type (int8)
} DATA_TYPE;

/**
 * \struct inferenceParams
 *
 * \brief The parameters of inference.
 *
 */
typedef struct {
	size_t workspaceSize_{ 16 << 20 }; 			//!< workspace size for tensorRT engine
	DATA_TYPE dataType{ FLOAT };				//!< data type
	bool bSwapChannel_{ false };				//!< Swap channel. BGR(default) or RGB
	bool bScale_{ false };						//!< scaling the input of inference or not
	float scale_{ 1.f };						//!< scale 
	float shift_{ 0.f };						//!< shift 
	char *calibrationTableFile_{ nullptr };		//!< calibration table for INT8 inference
} inferenceParams;


/** \class IDecodeProfiler
 *
 * \brief application-implemented interface for profiling
 * 
 * When a frame is decoded by decoder, Decode profiler will report the frame index and decoding time.
 *
 * the profiler will only be called after one frame is decoded.
 */
class IDecodeProfiler
{
public:
	/** \brief The callback of frame decode frame index and time reporting callback
	 * 
	 * \param frameIndex Frame index of decoding.
	 * 
	 * \param channel Channel of decoding.
	 * 
	 * \param deviceId Device ID.
	 * 
	 * \param ms Decoding time in millisecond.
	 */
	virtual void reportDecodeTime(const int frameIndex, const int channel, const int deviceId, double ms) = 0;

protected:
	virtual ~IDecodeProfiler() {}
};

/** \class IModuleProfiler 
 *
 * \brief Virtual base class of analysis pipeline profiler
 * 
 * \details user-implemented profiler. User needs to provide the function implementation. The DeviceWorker class will use the profiler to get the analysis performance of each batch of frames.
 *
 */
class IAnalysisProfiler 
{
public:
	/**
	* \brief Virtual function of performance reporting.
	*
	* \param ms The time in millisecond to execute the module.
	* \param batchSize The batch size on-the-fly.
	*/
	virtual void reportTime(double ms, int batchSize) = 0;

protected:
	virtual ~IAnalysisProfiler() {}
};

/** \class IDeviceWorker
 *
 * \brief DeepStream worker for GPU device.
 * 
 * \details All the functions on one device are grouped into a device worker.
 * The workflow of DeepStream SDK is:
 * 1. launch device worker;
 * 2. add decode task;
 * 3. add inference task;
 * 4. or (optional) other module tasks; 
 * 5. and launch the worker. 
 *
 */
class IDeviceWorker {
public:
	/** \brief Add decode task for multi-videos
	*
	* DeviceWorker will create nChannels docoders to decoding videos.
	*
	* \param codec The codec of video.
	*
	*/
	virtual void addDecodeTask(cudaVideoCodec codec, const int nMaxWidth = 0, const int nMaxHeight = 0) = 0;
	
	/** \brief Set Decode callback function
	*
	* User can define his/her own callback function to get the NV12 frame.
	*
	* \param pUserData The data defined by user.
	*
	* \param callback The callback function defined by user.
	*
	* \param channel The channel index of video.
	*
	*/
	virtual void setDecCallback(void *pUserData, DECODER_CALLBACK callback, const int channel) = 0;
	
	/** \brief Add encoding task for multi videos
	*
	* DeviceWorker will create nChannel encoders to encode videos.
	*
	* \param eCodec the codec of video encoding
	*
	* \param srcWidth the re-encoding width
	*
	* \param srcHeight the re-encoding height
	*
	* \param benchmark benchmark the performance, if true, transcoding result will not be saved.
	*
	* \param batchResize if true, deepStream will choose batch resize.
	*
	*/
    //virtual void addEncodeTask(cudaVideoCodec eCodec, const int srcWidth, const int srcHeight, const bool benchmark, const bool batchResize) = 0;
	
	/** \brief Add color space convertor (module), the module is pre-defined in DeepStream. \see IModule.
	*
	* This module is used to convert decoded frames to RGB planar images.
	* The module has two output tensors, one is RGB planar images, another one is NV12 frames.
	*
	* \param format The format of output (RGB planar or BGR planar).
	*
	* \return The pointer of this module.
	*
	*/
	virtual IModule* addColorSpaceConvertorTask(IMAGE_FORMAT format) = 0;
	
	/** \brief Add inference task (module), the module is pre-defined in DeepStream. \see IModule.
	*
	* DeviceWorker will create inference module for inference.
	*
	* \param moduleParam Information of previous module connected.
	*
	* \param deployFile The network defination (caffe prototxt file).
	*
	* \param modelFile The weights of network (caffe weights file).
	*
	* \param meanFile The mean of training images (caffe mean file, set null if no mean file).
	*
	* \param inputs The layer as module input.
	*
	* \param outputs The layers as module outputs.
	*
	* \param maxBatchSize The maximum batch size of inference.
	*
	* \param pInferParam The parameter of inference preprocess.
	*
	* \return The pointer of the module.
	*
	*/
	
	virtual IModule* addInferenceTask(PRE_MODULE moduleParam,
										const char *deployFile, // caffe prototxt file
										const char *modelFile, // trained caffe model
										const char *meanFile,  // mean file
										const std::string& inputs,
										const std::vector<std::string >& outputs,
										const int maxBatchSize,
										inferenceParams *pInferParam = nullptr) = 0;

        /** \brief Add inference task for uff model.	
	 */
virtual IModule* addInferenceTask_uff(PRE_MODULE moduleParam,
                                                                                const char *uffFile, // uff model file
										const char *meanFile,  // mean file
                                                                                int nC, int cH, int nW, /* input tensor dimensions*/
										const std::string& inputs,
										const std::vector<std::string >& outputs,
										const int maxBatchSize,
										inferenceParams *pInferParam = nullptr) = 0;

 
	/** \brief Add a user-defined task
	* Users can define their own module. The user-defined module should be added into DeviceWorker engine and will be executed when calling DeviceWorker->start().
   	*
    * \see IModule
	* \param pModule The pointer of the user-defined module.
	*
	* \return The pointer of the module.
	*
	*/
	virtual IModule* addCustomerTask(IModule *pModule)  = 0;
	
	/** \brief Add a object crop task.
	*
    * Object cropping module is pre-defined in DeepStream. 
	* For object detection post-process, we might need to extract the object sub-images from original images. This module will cutoff
	* all the objects and resize them to the same size. For example, we have two kinds of objects 
	* need to be detected (object A and B), object A will need to be resized to two size (size a1 and a2), object B need to be resized
	* to one size (size b1). With this information, this module will output three tensor, tensor 1 includes all object A in size a1, tensor 2 includes all object A in size a2, 
    * and tensor 3 includes all the cropped object B with size b1.
	* With the output, user can easily add other modules to process the cropped images.
	*
	* \param pModule The pointer of the user-defined module.
	*
	* \return The pointer of this module.
	*
	*/
	virtual IModule* addObjectCropTask(PRE_MODULE_LIST moduleList,
										const std::vector<CATEGORY_INFO>& vCategoryInfo,
										const int nMaxBatch) = 0;
	
	/** \brief Set the profiler.
	*
 	* Application-implemented interface for decode profiling.
	*
	* \param pDecProfiler The decode profiler.
	*
	* \param channel The channel of video for decoding. 
	*
	*/
	
	virtual void setDecodeProfiler(IDecodeProfiler *pDecProfiler, const int channel) = 0;
	
	/** \brief Set the profiler.
	*
 	* Application-implemented interface for analysis profiling.
	*
	* \param pAnalysisProfiler The analysis profiler.
	*
	*/
	
	virtual void setAnalysisProfiler(IAnalysisProfiler *pAnalysisProfiler) = 0;
	
	/** \brief Start the DeviceWorker.
	*
	* The DeviceWorker will start to work when feeding video stream packet into DeviceWorker.
	*
	*/
	virtual void setInferFrameSkip(const int nFrames) = 0;
	
	virtual void start() = 0;
	
	/** \brief Stop the DeviceWorker.
	*
	*/
	
	virtual void stop()  = 0;
	
	/** \brief Push video packet into the DeviceWorker.
	*
	* \param pBuf The pointer of packet buffer.
	*
	* \param nBuf The length of packet.
	*
	* \param channel The channel of video in the DeviceWorker. 
	*
	*/
	
	virtual void pushPacket(const uint8_t *pBuf, const int nBuf, const int channel) = 0;
	
	/** \brief Stop push video packet into the DeviceWorker.
	*
	* \param channel The channel of video in the DeviceWorker.
	*
	*/
	
	virtual void stopPushPacket(const int channel) = 0;
	
	/** \brief Destroy the DeviceWorker
	*
	*/
	
	virtual void destroy() = 0;

protected:
	virtual ~IDeviceWorker() {}

};

/**
* Internal C entry point for creating IDeviceWorker 
*
**/

extern "C" { 
	void *createDeviceWorkerInternal(const int nChannels, const int deviceId, const int maxBatchSize);
}

/**
* \brief create an instance of an IDeviceWorker class
*
* This class is the logging class for the DeviceWorker
*
* \param nChannels the video channels
*
* \param deviceId the GPU id
*
*/

inline IDeviceWorker *createDeviceWorker(const int nChannels, const int deviceId, const int maxBatchSize = 0) {
	return reinterpret_cast<IDeviceWorker *>(createDeviceWorkerInternal(nChannels, deviceId, maxBatchSize));
}

/**
* \brief Function to get the video format
*
* \param [in] szVideoFilePath path to the video file
*
*/
CUVIDEOFORMAT getVideoFormat(const char *szVideoFilePath);

/**
* \brief Batched NV12 frame resize (nearest neighbor and lanczos algorithm).
*
* \param [in] dpSrc			source NV12 frame pointer
* \param [in] nSrcPitch		pitch of NV12 frame (in bytes)
* \param [in] nSrcWidth		width of NV12 frame
* \param [in] nSrcHeight	height of NV12 frame
* \param [in] nDstPitch		pitch of dst NV12 frame (in bytes)
* \param [in] nDstWidth		width of NV12 frame
* \param [in] nDstHeight	height of NV12 frame
* \param [in] nBatchSize	batch size of NV12 frame
* \param [in] stream		CUDA stream
* \param [out] dpDst		destination NV12 frame pointer 
*
*/
void resize_nv12_batch(uint8_t *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight,
					   uint8_t *dpDst, int nDstPitch, int nDstWidth, int nDstHeight, int nBatchSize, cudaStream_t stream);
void resize_nv12_lanczos_batch(uint8_t *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight,
								uint8_t *dpDst, int nDstPitch, int nDstWidth, int nDstHeight, int xOrder, int yOrder, int nBatchSize, cudaStream_t stream);

/**
* \brief Batched RGB planar resize (nearest neighbor).
*
* \param [in] dpSrc			source NV12 frame pointer
* \param [in] nSrcPitch		pitch of NV12 frame (in bytes)
* \param [in] nSrcWidth		width of NV12 frame
* \param [in] nSrcHeight	height of NV12 frame
* \param [in] nDstPitch		pitch of dst NV12 frame (in bytes)
* \param [in] nDstWidth		width of NV12 frame
* \param [in] nDstHeight	height of NV12 frame
* \param [in] nBatchSize	batch size of NV12 frame
* \param [in] stream		CUDA stream
* \param [out] dpDst		destination NV12 frame pointer 
*
*/
void resize_rgb_planar_batch(float *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, float *dpDst, int nDstPitch, int nDstWidth, int nDstHeight, int nBatchSize, cudaStream_t stream);

/**
* \brief Batched gray resize (nearest neighbor).
*
* \param [in] dpSrc			source pointer
* \param [in] nSrcPitch		pitch of image (in bytes)
* \param [in] nSrcWidth		width of image
* \param [in] nSrcHeight	height of image 
* \param [in] nDstPitch		pitch of dst image (in bytes)
* \param [in] nDstWidth		width of image
* \param [in] nDstHeight	height of image
* \param [in] nBatchSize	batch size of iamges 
* \param [in] stream		CUDA stream
* \param [out] dpDst		destination image pointer 
*
*/
void resize_gray_batch(float *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, float *dpDst, int nDstPitch, int nDstWidth, int nDstHeight, int nBatchSize, cudaStream_t stream);
/**
* \brief Convert NV12 frame to RGB or BGR planar (batch version).
*
* \param [in] pNv12			source NV12 frame pointer
* \param [in] nNv12Pitch	pitch of NV12 frame (in bytes)
* \param [in] nRgbPitch		pitch of RGB frame (in bytes)
* \param [in] nWidth		width of NV12 frame
* \param [in] nHeight		height of NV12 frame
* \param [in] nBatchSize	batch size of NV12 frame
* \param [in] bSwap			BGR(default) or RGB
* \param [in] stream		CUDA stream
* \param [out] pRgb			destination RGB frame pointer 
*
*/

void nv12_to_bgr_planar_batch(uint8_t *pNv12, int nNv12Pitch, float *pRgb, int nRgbPitch,
								int nWidth, int nHeight, int nBatchSize, bool bSwap, cudaStream_t stream);

/**
* \brief Convert NV12 frame to gray float (batch version).
*
* \param [in] pNv12			source NV12 frame pointer
* \param [in] nNv12Pitch	pitch of NV12 frame (in bytes)
* \param [in] nGrayPitch	pitch of gray (in bytes)
* \param [in] nWidth		width of NV12 frame
* \param [in] nHeight		height of NV12 frame
* \param [in] nBatchSize	batch size of NV12 frame
* \param [in] stream		CUDA stream
* \param [out] pGray		destination Gray image pointer 
*
*/
void nv12_to_gray_batch(const uint8_t *pNv12, int nNv12Pitch, float *pGray, int nGrayPitch, int nWidth, int nHeight, int nBatchSize, cudaStream_t stream);

/**
* \brief Convert NV12 frame to BGRA.
*
* \param [in] dpNv12			source NV12 frame pointer
* \param [in] nNv12Pitch	pitch of NV12 frame (in bytes)
* \param [in] nBgraPitch	pitch of BGRA frame (in bytes)
* \param [in] nWidth		width of NV12 frame
* \param [in] nHeight		height of NV12 frame
* \param [in] stream		CUDA stream
* \param [out] dpRgb		destination BGRA frame pointer 
*/

void nv12_to_bgra(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpRgb, int nBgraPitch,
					int nWidth, int nHeight, cudaStream_t stream);

void deepStreamInit();

#endif // DEEPSTREAM_H
