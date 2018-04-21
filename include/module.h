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

#ifndef MODULE_H
#define MODULE_H

#include <cuda_runtime.h>

/** 
 * \file module.h
 *
 * This is the top-level API file for creating user-defined modules in DeepStream.
 */

// forward declare
class IModule;

/**
 * \brief Maximum number of bbox per frame. It is used to limit detected objects in one frame.
 */
#define MAX_BOXPERFRAME         50

/**
 * \brief Maximum number of style per category. It is used to limit resizing style (size) for one object catergory.
 */
#define MAX_STYLEPERCATEGORY    3

/**
 * \struct BBOX_INFO
 *
 * \brief The information of bounding box for object detection.
 * The box can be resized to several size based on its category. 
 *
 */
typedef struct {
    float h	= 0;			//!< height of bounding box
    float w	= 0;			//!< width of bounding box
    float x	= 0;			//!< x-coord of left-top of box in original image
    float y	= 0;			//!< y-coord of left-top of box in original image
    int category = 0;		//!< category of object in bounding box
	float score = 0.;		//!< prob of category
	bool bSkip = false;		//!< skip the anlaysis step for this sub-image or not
} BBOX_INFO;

/**
 * \struct BBOXS_PER_FRAME
 *
 * \brief The information bounding boxes in frames for object detection.
 * The class will record the original frame and video for the detected objects.
 *
 */
typedef struct {
    int frameIndex = 0;				//!< index of frame
	int videoIndex = 0;				//!< index of video
	int nBBox = 0;						//!< number of box per frame
    BBOX_INFO bbox[MAX_BOXPERFRAME];	//!< info of each box
} BBOXS_PER_FRAME;


/**
 * \struct TRACE_INFO
 *
 * \brief The information of tensor recording bounding box, frame and video index.
 *
 */
typedef struct {
	int frameIndex = -1;	//!< frame index
	int videoIndex = -1;	//!< video index
	int originalFrame = -1;	//!< index in current batch of frames
	BBOX_INFO boxInfo;		//!< info of bbox
} TRACE_INFO;

/**
 * \struct STYLE_INFO
 *
 * \brief The detected object will be resized to style provide size.
 * It is possible to resize one object (or one category) to multi-style.
 *
 */
typedef struct {
    int h;			//!< height
    int w;			//!< width
} STYLE_INFO;

/**
 * \struct CATEGORY_INFO
 *
 * \brief The resizing of detected objects is based on category, and each category may require multi-style.
 *  CATEGORY_INFO is used to record the styles of each category.
 *
 */
typedef struct {
    int nStyle;								//!< number of styles
    STYLE_INFO style[MAX_STYLEPERCATEGORY]; //!< info of styles
} CATEGORY_INFO;

/**
 * \enum MEMORY_TYPE 
 *
 * \brief CPU data or GPU data.
 *
 */
typedef enum {
	GPU_DATA = 0,	//!< gpu data
	CPU_DATA = 1    //!< cpu data
} MEMORY_TYPE;

/**
 * \enum TENSOR_TYPE 
 *
 * \brief The type of tensor.
 *
 */
typedef enum {
	FLOAT_TENSOR = 0,	//!< float tensor
	NV12_FRAME = 1,     //!< nv12 frames
	OBJ_COORD = 2,		//!< coordinates of object
	CUSTOMER_TYPE = 3   //!< user-defined type
} TENSOR_TYPE;

/**
 * \enum IMAGE_FORMAT
 *
 * \brief RGB planar or BGR planar
 *
 */
typedef enum {
	RGB_PLANAR = 0,
	BGR_PLANAR = 1,
	GRAY = 2
} IMAGE_FORMAT; 


/** \class IStreamTensor
 *
 * \brief Virtual base class of stream tensor
 * 
 * \details The input and output data of module is defined as stream tensor.
 *
 * \see createStreamTensor
 *
 */
class IStreamTensor {
public:
	
	/**
	* \brief Get the pointer of GPU data.
	*
	*/
	virtual void *getGpuData() = 0;
	
	/**
	* \brief Get the const pointer of GPU data.
	*
	*/
	virtual const void *getConstGpuData() = 0;
	
	/**
	* \brief Get the pointer of CPU data.
	*
	*/
	virtual void *getCpuData() = 0;
	
	/**
	* \brief Get the const pointer of CPU data.
	*
	*/
	virtual const void *getConstCpuData() = 0;
	
	/**
	* \brief Get the size of element
	*
	* \return size of element
	*
	*/
	virtual size_t getElemSize() const = 0;

	/**
	* \brief Get the type of memory (CPU or GPU data)
	*
	* \return type of memory 
	*
	*/
	virtual MEMORY_TYPE getMemoryType() const = 0;
	
	/**
	* \brief Get the type of tensor
	*
	* \return type of tensor
	*
	*/
	virtual TENSOR_TYPE getTensorType() const = 0;

	/**
	* \brief Set the shape of tensor data with vector. 
	* NCHW * sizeof(nElemSize), where N is the batch size.
	*
	*/
	virtual void setShape(const std::vector<int>& shape) = 0;
	
	/**
	* \brief Set the shape of tensor data with tensor parameters. 
	* NCHW * sizeof(nElemSize), where N is the batch size
	*
	*/
	virtual void setShape(const int n, const int c, const int h, const int w) = 0;

	/**
	* \brief Get the shape of the tensor (NCHW). 
	* NCHW * sizeof(nElemSize), where N is the batch size
    *
    * \return vector<int> &, the size of tensor, the four elements are N, C, H and W.
	*
	*/
	virtual std::vector<int>& getShape() = 0;
	
	/**
	* \brief Get the trace information of the tensor. 
	* The size of returned vector is equal to the batch size. Each element in the 
	* tensor needs to record orignal frame index and video index.
	*
	* \return Trace information of tensor in batch. The size of returned vector is batch size.
	*/
	virtual std::vector<TRACE_INFO > getTraceInfos() = 0;
	
	/**
	* \brief Get the max batch size of the tensor.
	*
	* \return Maximum batch size
	*/
	virtual int getMaxBatch() const = 0;
	
	/**
	* \brief Set the trace information of the tensor.
	* 
    * The tensor information is pass to the stream through TRACE_INFO calss. The size of passed vector is batch size.
	*
	*/
	virtual void setTraceInfo(std::vector<TRACE_INFO >& vTraceInfos) = 0;
	
	/**
	* \brief Destroy the tensor, and free the memory
	*
	*/
	virtual void destroy() = 0;

protected:
	virtual ~IStreamTensor() {}
};

/**
* Internal C entry point for creating IStreamTensor
*
**/
extern "C" { 
	void *createStreamTensorInternal(const int nMaxLen, const size_t nElemSize, const TENSOR_TYPE Ttype, const MEMORY_TYPE Mtype, const int devID);
}

/**
* \brief Create an instance of an IStreamTensor class
*
* \param nMaxLen The max length of element.
*
* \param nElemSize The size of each element.
*
* \param Ttype The type of tensor.
*
* \param Mtype The type of memory.
*
* \param deviceId GPU device ID.
*
*/
inline IStreamTensor *createStreamTensor(const int nMaxLen, const size_t nElemSize, const TENSOR_TYPE Ttype, const MEMORY_TYPE Mtype, const int deviceId) {
	return reinterpret_cast<IStreamTensor *>(createStreamTensorInternal(nMaxLen, nElemSize, Ttype, Mtype, deviceId));
}


/**
 * \brief The pointer of pre-module and the index of its output tensor.
 */
typedef std::pair<IModule*, int> PRE_MODULE;

/**
 * \brief A list of input tensors of pre-modules.
 */
typedef std::vector<std::pair<IModule*, int> > PRE_MODULE_LIST;

/** \class ModuleContext
 *
 * \brief The context of module
 * 
 */
class ModuleContext {
public:
	cudaStream_t stream{ nullptr };
};

/** \class IModuleProfiler 
 *
 * \brief Virtual base class of module profiler
 * 
 * \details user-implemented profiler. User needs to provide the function implementation. The DeviceWorker class will call the profiler to profile the performance of this module.
 *
 */
class IModuleProfiler
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
	virtual ~IModuleProfiler() {}
};

/**
 * \brief MODULE_CALLBACK
 *
 * \brief The callback function of module execution.
 * The function needs to be defined by user, and will be called after module execution.
 *
 * \param pUserData The pointer of user data.
 *
 * \param out The output tensor of module.
 *
 * \see setCallback
 * 
 */
typedef void (*MODULE_CALLBACK)(void *pUserData, std::vector<IStreamTensor *>& out);

/** \class IModule 
 *
 * \brief Virtual base class of module.
 * 
 * \details After decoding, DeviceWorker can add a sequence of modules to analysis the frames.
 * DeepStream includes three implemented modules: color space convert, inference, and object sub-image cropping and resizing.
 * User can define own module which have to include following routine:
 * 1. initialize;
 * 2. execute;
 * 3. destroy;
 * 4. (optional) and the profiler in IModuleProfiler class.
 *
 * \see IDeviceWorker::addCustomerTask()
 *
 */
class IModule {
public:
	/** \brief The routine of module initialization.
	*
	* The DeviceWorker will call this function to initialize this module.
	*
	*/
	virtual void initialize() = 0;

	/** \brief The routine of module execution.
	*
 	* The DeviceWorker will call this function to execute this module.
	*
	* \param context Module context
	*
	* \param in The Input tensors
	*
	* \param out The output tensors
	*
	*/
	virtual void execute(const ModuleContext& context, const std::vector<IStreamTensor *>& in,  const std::vector<IStreamTensor *>& out) = 0;
	
	/** \brief The routine of module destroy.
	*
	* The DeviceWorker will call this function to destroy this module.
	*
	*/
	virtual void destroy() = 0;

	/** \brief Get number of input tensors.
	*
	*/
	virtual int getNbInputs() const = 0;

	/** \brief Get the info of premodule.
	*
	* \param tensorIndex The index of input tensor.
	*
	* \return \see PRE_MODULE
	*
	*/
	virtual PRE_MODULE getPreModule(const int tensorIndex) const = 0;

	/** \brief Get the number of output tensors.
	*
	*/
	virtual int getNbOutputs() const = 0;
	
	/** \brief Get the output tensor.
	*
	* \param tensorIndex the index of input tensor.
	*
	* \return The pointer of output tensors (\see IStreamTensor).
	*
	*/
	virtual IStreamTensor* getOutputTensor(const int tensorIndex) const = 0;
	
	/** \brief Set the module profiler (user-defined).
	* The profiler (user-defined) will be call by DeviceWorker.    
	* 
    * \param pProfiler The pointer of profiler (user-defined).
	*
	*/
	virtual void setProfiler(IModuleProfiler *pProfiler) = 0;
	
	/** \brief Get the module profiler (user-defined).
    *
	* \return The pointer of profiler (user-defined).
	*
	*/
	virtual IModuleProfiler* getProfiler() const = 0;

	/** \brief Set the module callback. 
	* User need to define the callback function to get the output of this module.
	*
	* \param pUserData user datat
	*
	* \param callback user-defined callback function
	*
	*/
	virtual void setCallback(void *pUserData, MODULE_CALLBACK callback) = 0;
	
	/** \brief get the module callback 
	* DeviceWorker will get the callback function by this routine, if the pointer is not nullptr
	* the callback function will be run by DeviceWorker after this module.
	*
	* \return the pointer of user data and callback function
	*
	*/
	virtual std::pair<void*, MODULE_CALLBACK> getCallback() const = 0;

protected:
	virtual ~IModule() {}
};


#endif

