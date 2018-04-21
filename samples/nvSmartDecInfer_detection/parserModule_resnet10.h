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
#ifndef PARSER_RESNET_H
#define PARSER_RESNET_H

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include "deepStream.h"

typedef struct {
	int c;
	int h;
	int w;
} Dims3;

typedef struct {
  	float threshold = 0.5f;
  //  	float threshold = 0.00f;
} CLASS_ATTR;


#define MAX_CLASSES 4

class ParserModule : public IModule {
public:
	explicit
	ParserModule(PRE_MODULE_LIST &preModules,
					const int nChannels,
					const int devID,
					simplelogger::Logger *logger) 
	: preModules_(preModules), nChannels_(nChannels), devID_(devID), logger_(logger) {}

	~ParserModule() {}

	// override
	void initialize() override;

	void execute(const ModuleContext& context, const std::vector<IStreamTensor *>& vpInputTensors,  const std::vector<IStreamTensor *>& vpOutputTensors) override;
	
	void destroy() override {
		for (int i = 0; i < vpOutputTensors_.size(); ++i) {
			vpOutputTensors_[i]->destroy();
		}
	}

	int getNbInputs() const override {
		return preModules_.size();
	}
	
	PRE_MODULE getPreModule(const int tensorIndex) const override {
		return preModules_[tensorIndex];
	}
	
	int getNbOutputs() const override {
		return vpOutputTensors_.size();
	}
	
	IStreamTensor* getOutputTensor(const int tensorIndex) const override {
		return vpOutputTensors_[tensorIndex];
	}
	
	void setProfiler(IModuleProfiler *pProfiler) override {
		pProfiler_ = pProfiler;
	}
	
	IModuleProfiler* getProfiler() const override {
		return pProfiler_;
	}
	
	void setCallback(void *pUserData, MODULE_CALLBACK callback) override {
		pUserData_ = pUserData;
		callback_ = callback;
	}

	std::pair<void *, MODULE_CALLBACK> getCallback() const override {
		return std::pair<void*, MODULE_CALLBACK>(pUserData_, callback_);
	}

private:
	void parseNvhelnet(Dims3 outputDims, const float *outputCov, Dims3 outputDimsBBOX, const float *outputBBOX, std::vector<cv::Rect> *rectList, const int class_num);
	
	int net_height =  368;
	int net_width = 640;
        const int class_num = 4;
	CLASS_ATTR class_attrs[4];

	int nChannels_{ 0 };
	int devID_{ 0 };
	simplelogger::Logger *logger_{ nullptr };
	void *pUserData_{ nullptr };
	MODULE_CALLBACK callback_{ nullptr };
	IModuleProfiler* pProfiler_{ nullptr };	

	std::vector<int > vFrameCount_;
	PRE_MODULE_LIST preModules_;
	std::vector<IStreamTensor*> vpOutputTensors_;
};

void ParserModule::initialize() {
	vFrameCount_.resize(nChannels_);
	vpOutputTensors_.resize(1, nullptr);
	vpOutputTensors_[0] = createStreamTensor(nChannels_, sizeof(BBOXS_PER_FRAME),
											OBJ_COORD, CPU_DATA, devID_);
	assert(nullptr != vpOutputTensors_[0]);
}

void ParserModule::execute(const ModuleContext& context, const std::vector<IStreamTensor *>& vpInputTensors,  const std::vector<IStreamTensor *>& vpOutputTensors) {
	assert(2 == vpInputTensors.size());
	Dims3 outputDims;
	Dims3 outputDimsBBOX;
	
	// tensor 0: cov 
	std::vector<int> shape_0 = vpInputTensors[0]->getShape();
	int nFrames = shape_0[0];
	if (0 == nFrames) {
		return;
	}
	
	outputDims.c = shape_0[1];
	outputDims.h = shape_0[2];
	outputDims.w = shape_0[3];
	const float *pCov = reinterpret_cast<const float*>(vpInputTensors[0]->getConstCpuData());
	std::vector<TRACE_INFO > trace_0 = vpInputTensors[0]->getTraceInfos();
	assert(shape_0[0] == trace_0.size());
	
	// tensor 1: bbox
	std::vector<int> shape_1 = vpInputTensors[1]->getShape();
	assert(shape_1[0] == shape_0[0]);
	
	outputDimsBBOX.c = shape_1[1];
	outputDimsBBOX.h = shape_1[2];
	outputDimsBBOX.w = shape_1[3];
	const float *pBBOX = reinterpret_cast<const float*>(vpInputTensors[1]->getConstCpuData());
	
	
	std::vector<BBOXS_PER_FRAME > bboxs_batch;
	for (int iB = 0; iB < nFrames; ++iB) {
    	std::vector<cv::Rect> *rectListCLass;
    	rectListCLass = new std::vector<cv::Rect>[class_num];
		const float *outputCov  = pCov  + iB * shape_0[1] * shape_0[2] * shape_0[3];
		const float *outputBBOX = pBBOX + iB * shape_1[1] * shape_1[2] * shape_1[3];
		parseNvhelnet(outputDims, outputCov, outputDimsBBOX, outputBBOX, rectListCLass, class_num);
		
		BBOXS_PER_FRAME bboxs;
		bboxs.frameIndex = trace_0[iB].frameIndex;
		bboxs.videoIndex = trace_0[iB].videoIndex;
		bboxs.nBBox = 0;
		for (int iClass = 0; iClass < class_num; ++iClass) {
        	std::vector<cv::Rect> &rectList = rectListCLass[iClass];
			for (int iRect = 0; iRect < (int)rectList.size(); ++iRect) {
            	cv::Rect &r = rectList[iRect];
				
				int nBBox = bboxs.nBBox;
	            if (nBBox < MAX_BOXPERFRAME) {
					BBOX_INFO bbox; // all norm to (0, 1)
					bbox.x = (float)r.x / (float)net_width;
					bbox.y = (float)r.y / (float)net_height;
					bbox.w = (float)r.width / (float)net_width;
					bbox.h = (float)r.height / (float)net_height;
					bbox.category = iClass;
					bboxs.bbox[nBBox] = bbox;
					bboxs.nBBox++;
            	}
			}
		}
		bboxs_batch.push_back(bboxs);
		delete [] rectListCLass;
	}

	assert(1 == vpOutputTensors.size());
	IStreamTensor *pOutputTensor = vpOutputTensors[0];
	// just for check
	{
		assert(nullptr != pOutputTensor);
		size_t nElemSize = pOutputTensor->getElemSize();
		assert(sizeof(BBOXS_PER_FRAME) == nElemSize);
		TENSOR_TYPE Ttype = pOutputTensor->getTensorType();
		assert(OBJ_COORD == Ttype);
		assert(nFrames <= pOutputTensor->getMaxBatch());
	}

	BBOXS_PER_FRAME *pBBox_batch = reinterpret_cast<BBOXS_PER_FRAME*>(pOutputTensor->getCpuData());	
	for (int iF = 0; iF < nFrames; ++iF) {
		pBBox_batch[iF] = bboxs_batch[iF];
	}
	// batch size changes at runtime, so we need to set the shape
	pOutputTensor->setShape(nFrames, 1, 1, 1);
}

void ParserModule::parseNvhelnet(Dims3 outputDims, const float *outputCov, Dims3 outputDimsBBOX, const float *outputBBOX, std::vector<cv::Rect> *rectList, const int class_num) {
  int grid_x_ = outputDims.w;
  int grid_y_ = outputDims.h;
  int gridsize_ = grid_x_ * grid_y_;

  assert(class_num == outputDims.c);

  if (outputDims.c > MAX_CLASSES)
  {
    printf ("*** ERROR : Network Classes (%d) more than available classes (%d)\n", outputDims.c, MAX_CLASSES);
    exit (-1);
  }

  int target_shape[2] = {grid_x_, grid_y_};
  //  float bbox_norm[2] = {640.0, 368.0};
  //  float bbox_norm[2] = {960.0, 544.0};
  float bbox_norm[2] = {35,35};
  float gc_centers_0[target_shape[0]];
  float gc_centers_1[target_shape[1]];
  for (int i = 0; i < target_shape[0]; i++)
  {
      gc_centers_0[i] = (float)(i * 16 + 0.5);
      gc_centers_0[i] /= (float)bbox_norm[0];

  }
  for (int i = 0; i < target_shape[1]; i++)
  {
      gc_centers_1[i] = (float)(i * 16 + 0.5);
      gc_centers_1[i] /= (float)bbox_norm[1];

  }
  
  for (int c = 0; c < outputDims.c; c++)
  {
    const float *output_x1 = outputBBOX + c * 4 * outputDimsBBOX.h * outputDimsBBOX.w;
    const float *output_y1 = output_x1 + outputDims.h * outputDims.w;
    const float *output_x2 = output_y1 + outputDims.h * outputDims.w;
    const float *output_y2 = output_x2 + outputDims.h * outputDims.w;
    {
        for (int h = 0; h < grid_y_; h++)
        {
            for (int w = 0; w < grid_x_; w++)
            {
                int i = w + h * grid_x_;
                if (outputCov[c*gridsize_+i] >= class_attrs[c].threshold)
                {
                    float rectx1_f, recty1_f, rectx2_f, recty2_f;
                    rectx1_f = recty1_f = rectx2_f = recty2_f = 0.0;
					
                    rectx1_f = output_x1[w + h * grid_x_] - gc_centers_0[w];
                    recty1_f = output_y1[w + h * grid_x_] - gc_centers_1[h];
                    rectx2_f = output_x2[w + h * grid_x_] + gc_centers_0[w];
                    recty2_f = output_y2[w + h * grid_x_] + gc_centers_1[h];

                    rectx1_f *= (float)(-bbox_norm[0]);
                    recty1_f *= (float)(-bbox_norm[1]);
                    rectx2_f *= (float)(bbox_norm[0]);
                    recty2_f *= (float)(bbox_norm[1]);
					
  		    int rectx1, recty1, rectx2, recty2;

                    rectx1 = (int)rectx1_f;
                    recty1 = (int)recty1_f;
                    rectx2 = (int)rectx2_f;
                    recty2 = (int)recty2_f;
                	
					if (rectx1 < 0)
                	    rectx1 = 0;
                	if (rectx2 < 0)
                	    rectx2 = 0;
                	if (recty1 < 0)
                	    recty1 = 0;
                	if (recty2 < 0)
                	    recty2 = 0;

                    if (rectx1 >= net_width)
                        rectx1 = net_width - 1;
                    if (rectx2 >= net_width)
                        rectx2 = net_width - 1;
                    if (recty1 >= net_height)
                        recty1 = net_height - 1;
                    if (recty2 >= net_height)
                        recty2 = net_height - 1;

                    rectList[c].push_back(cv::Rect(rectx1, recty1, rectx2-rectx1, recty2-recty1));
                }
            }
        }
    }
	cv::groupRectangles(rectList[c], 1, 0.6);
  }
}
#endif // PARSER_RESNET_H
