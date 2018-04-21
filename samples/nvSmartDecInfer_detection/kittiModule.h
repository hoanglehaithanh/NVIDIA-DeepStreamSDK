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

#ifndef KITTI_MODULE_H
#define KITTI_MODULE_H

#include "common.h"

class KittiLoggerModule : public IModule {
public:
	explicit
	KittiLoggerModule(PRE_MODULE_LIST &preModules,
					const int nChannels,
					const int devID_display,
					const int devID_infer,
					char *labelFile,
					simplelogger::Logger *logger) 
	  : preModules_(preModules), nChannels_(nChannels), devID_display_(devID_display), devID_infer_(devID_infer), labelFile_(labelFile), logger_(logger) {}

	~KittiLoggerModule() {}

	// override
	void initialize() override;

	void execute(const ModuleContext& context, const std::vector<IStreamTensor *>& vpInputTensors,  const std::vector<IStreamTensor *>& vpOutputTensors) override;
	
	void destroy() override {
	  
		for(int i = 0; i < MAX_SUPPORTED_CHANNELS; i++)
		  if (logFile[i] != NULL) {
		    logFile[i]->close();
		    delete logFile[i];

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
	int nChannels_{ 0 };
	int devID_display_{ -1 };
	int devID_infer_{ -1 };
	std::vector<std::string > vSynsets_;
	
	void *pUserData_{ nullptr };
	MODULE_CALLBACK callback_{ nullptr };
	IModuleProfiler* pProfiler_{ nullptr };	
	
	char *labelFile_{ nullptr };
	uint8_t *pFrameResized_{ nullptr };
	
	uint8_t * pBGRA_local_{ nullptr };
	size_t pitchInbytes_{ 0 };

	simplelogger::Logger *logger_{ nullptr };
	
	PRE_MODULE_LIST preModules_;
	std::vector<IStreamTensor*> vpOutputTensors_;

	static const int MAX_SUPPORTED_CHANNELS = 128;

        std::ofstream *logFile[MAX_SUPPORTED_CHANNELS];
};
	
void KittiLoggerModule::initialize() {
	//LOG_DEBUG(logger, "PlaybackModule: parse synset file...");
	std::ifstream iLabel(labelFile_);
	if (iLabel.is_open()) {
		std::string line;
		while (std::getline(iLabel, line)) {
			vSynsets_.push_back(line);
			line.clear();
		}
		iLabel.close();
	} else {
		LOG_ERROR(logger, "Failed to open synset file " << labelFile_ << "!");
		exit(-1);
	}

	for(int i = 0; i < MAX_SUPPORTED_CHANNELS; i++)
	  logFile[i] = NULL;
	
}

void KittiLoggerModule::execute(const ModuleContext& context, const std::vector<IStreamTensor *>& vpInputTensors,  const std::vector<IStreamTensor *>& vpOutputTensors) {
	assert(2 == vpInputTensors.size());
	cudaStream_t stream = context.stream;
	//=================================================================
	// NV12 frames
	//=================================================================
	std::vector<int> shape_nv12 = vpInputTensors[0]->getShape();
	TENSOR_TYPE Ttype_0 = vpInputTensors[0]->getTensorType();
	assert(NV12_FRAME == Ttype_0);
	
	int nFrames = shape_nv12[0];
	int nHeight = shape_nv12[2] * 2; // NV12, YUV420
	int nWidth = shape_nv12[3];
	
	if (0 == nFrames) {
		return;
	}

	uint8_t *dpFrames = reinterpret_cast<uint8_t*>(vpInputTensors[0]->getGpuData());
	std::vector<TRACE_INFO > tensorInfo_nv12 = vpInputTensors[0]->getTraceInfos();
	assert(shape_nv12[0] == tensorInfo_nv12.size());

	//=================================================================
	// nvhelnet Object Coords
	//=================================================================
	std::vector<int> shape_coord = vpInputTensors[1]->getShape();
	TENSOR_TYPE Ttype_1 = vpInputTensors[1]->getTensorType();
	assert(OBJ_COORD == Ttype_1);
	assert(shape_coord[0] == nFrames);
	
	BBOXS_PER_FRAME *pBBox_batch = reinterpret_cast<BBOXS_PER_FRAME*>(vpInputTensors[1]->getCpuData());	
	assert(nullptr != pBBox_batch);
	

	for (int iF = 0; iF < nFrames; ++iF) {
   	        int frameIndex = tensorInfo_nv12[iF].frameIndex;
		int videoIndex = tensorInfo_nv12[iF].videoIndex;

   	        if (videoIndex > MAX_SUPPORTED_CHANNELS) {
   	           LOG_ERROR(logger, "Can supoprt maximum of 128 channels. Exiting");
	           exit(-1);
   	        }

		std::string sFileName = "./log/log_ch";
		sFileName += std::to_string(videoIndex) + ".txt";
		
                if (logFile[videoIndex] == NULL) {
		  logFile[videoIndex] = new std::ofstream;
		  logFile[videoIndex]->open(sFileName, std::ios::trunc);
  	          if (!logFile[videoIndex]->is_open()) {
     	             LOG_ERROR(logger, "Failed to Open file " << sFileName);
  	             exit(0);
   	           }

		}
		/*
		else {
    	          logFile[videoIndex]->open(sFileName, std::ios::app | std::ios::binary);
	        }
		*/
		
		// log that  bounding box
		BBOXS_PER_FRAME &bboxs = pBBox_batch[iF];
		for (int i = 0; i < bboxs.nBBox; ++i) {
		  //			if (!bboxs.bbox[i].bSkip)
			  {
				*logFile[videoIndex] <<"Frame ["<<frameIndex<<"]"<<vSynsets_[bboxs.bbox[i].category]<<" 0.0 0 0.0 "<<bboxs.bbox[i].x*nWidth<<" "<<bboxs.bbox[i].y*nHeight<<" "<<(bboxs.bbox[i].x + bboxs.bbox[i].w)*nWidth<<" "<<(bboxs.bbox[i].y + bboxs.bbox[i].h)*nHeight<<" 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n";
			}
		}
		//	logFile[videoIndex]->close();
		logFile[videoIndex]->flush();
	
	}
}

#endif
