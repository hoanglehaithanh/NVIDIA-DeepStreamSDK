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

#ifndef PLAYBACK_MODULE_H
#define PLAYBACK_MODULE_H

#include "common.h"

class PlaybackModule : public IModule {
public:
	explicit
	PlaybackModule(PRE_MODULE_LIST &preModules,
					const int nChannels,
					const int devID_display,
					const int devID_infer,
					char *labelFile,
					int tileWidth,
					int tileHeight,
					int tilesInRow,
					bool bFullScreen,
					simplelogger::Logger *logger) 
	: preModules_(preModules), nChannels_(nChannels), devID_display_(devID_display), devID_infer_(devID_infer), labelFile_(labelFile), tileWidth_(tileWidth), tileHeight_(tileHeight), tilesInRow_(tilesInRow), bFullScreen_(bFullScreen), logger_(logger) {}

	~PlaybackModule() {}

	// override
	void initialize() override;

	void execute(const ModuleContext& context, const std::vector<IStreamTensor *>& vpInputTensors,  const std::vector<IStreamTensor *>& vpOutputTensors) override;
	
	void destroy() override {
		if (nullptr != pFrameResized_) {
			ck(cudaFree(pFrameResized_));
		}
		if (nullptr != pPresenterGL_) {
			delete pPresenterGL_;
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
	int tileWidth_ 	{ 0 };
	int tileHeight_ { 0 };
	int tilesInRow_	{ 0 };
	bool bFullScreen_ 		{ false };
	demo_button demoButton_;
	std::vector<std::string > vSynsets_;
	
	void *pUserData_{ nullptr };
	MODULE_CALLBACK callback_{ nullptr };
	IModuleProfiler* pProfiler_{ nullptr };	
	
	char *labelFile_{ nullptr };
	uint8_t *pFrameResized_{ nullptr };
	
	uint8_t * pBGRA_local_{ nullptr };
	size_t pitchInbytes_{ 0 };

	simplelogger::Logger *logger_{ nullptr };
	PresenterGL	*pPresenterGL_{ nullptr };
	
	PRE_MODULE_LIST preModules_;
	std::vector<IStreamTensor*> vpOutputTensors_;
};
	
void PlaybackModule::initialize() {
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
	
	// Init PresenterGL (OpenGL Display)
	pPresenterGL_ = new PresenterGL(devID_display_,
										tileWidth_,
										tileHeight_,
										nChannels_,
										tilesInRow_,
										vSynsets_,
										bFullScreen_,
										&demoButton_);
}

void PlaybackModule::execute(const ModuleContext& context, const std::vector<IStreamTensor *>& vpInputTensors,  const std::vector<IStreamTensor *>& vpOutputTensors) {
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
	
	// playback and draw bounding box
	assert(nullptr != pPresenterGL_);
	int dstWidth 				= 	pPresenterGL_->GetSubWindowWidth();
	int dstHeight 				= 	pPresenterGL_->GetSubWindowHeight();
	size_t nFrameSizeResized 	= 	dstWidth * dstHeight * 3 / 2;
	
	if (NULL == pFrameResized_) {
		ck( cudaMalloc((void **)&pFrameResized_, nFrameSizeResized) );
	}
	
	int nx = nChannels_ > tilesInRow_ ? tilesInRow_ : nChannels_;
	int ny = (nChannels_ + nx - 1) / nx;
	
	int nWindowWidth  = nx * tileWidth_;
	int nWindowHeight = ny * tileHeight_;
	
	if (nullptr == pBGRA_local_) {
		ck(cudaMalloc((void **)&pBGRA_local_, nWindowWidth * nWindowHeight * 4 * sizeof(uint8_t)));
		ck(cudaMemset(pBGRA_local_, 0, nWindowWidth * nWindowHeight * 4 * sizeof(uint8_t)));
	}
	
	StopWatch myTimer_draw;
	myTimer_draw.Start();
	for (int iF = 0; iF < nFrames; ++iF) {
		int frameIndex = tensorInfo_nv12[iF].frameIndex;
		int videoIndex = tensorInfo_nv12[iF].videoIndex;
		uint8_t *dpFrame = dpFrames + iF * nWidth * nHeight * 3 / 2;

		resize_nv12_batch(dpFrame, nWidth * 1, nWidth, nHeight,
							pFrameResized_, dstWidth * 1, dstWidth, dstHeight,
							1, stream);
		
		int id_x = videoIndex % nx;
		int id_y = videoIndex / nx;
		int rect_x = tileWidth_ * id_x;
		int rect_y = tileHeight_ * id_y;
		int stride = rect_x * 4 + rect_y * nWindowWidth * 4;
		uint8_t *dpDst = pBGRA_local_ + stride;
		nv12_to_bgra(pFrameResized_, dstWidth * 1,
						dpDst, 4*nWindowWidth,
						dstWidth, dstHeight, stream);

		// draw bounding box
		BBOXS_PER_FRAME &bboxs = pBBox_batch[iF];
		pPresenterGL_->SetText(bboxs, videoIndex);
		for (int i = 0; i < bboxs.nBBox; ++i) {
			if (!bboxs.bbox[i].bSkip) {
				int x_min = bboxs.bbox[i].x * dstWidth;
				int y_min = bboxs.bbox[i].y * dstHeight;
				int x_max = (bboxs.bbox[i].x + bboxs.bbox[i].w) * dstWidth;
				int y_max = (bboxs.bbox[i].y + bboxs.bbox[i].h) * dstHeight;
				
				drawBoundingBox_cuda(dpDst, dstWidth, dstHeight, nWindowWidth*4*1, x_min, y_min, x_max, y_max, stream);
			}
		}
		// sync
		ck(cudaStreamSynchronize(stream));
	}
	
	int nBgraPitch = 0;
	uint8_t * pBGRA = nullptr;
	pPresenterGL_->DeviceFrameBuffer(&pBGRA, &nBgraPitch, 0);
	ck(cudaMemcpy2DAsync(pBGRA, nBgraPitch, pBGRA_local_, nWindowWidth * 4, nWindowWidth*4, nWindowHeight, cudaMemcpyDeviceToDevice, stream));
	ck(cudaStreamSynchronize(stream));
	
	double t_draw = myTimer_draw.Stop();
}

#endif
