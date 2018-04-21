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

#include <cstring>
#include <cstdio>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <thread>
#include <iomanip>
#include <sys/time.h>
#include <helper_cuda.h>

#include "common.h"
#define RESNET10

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

int g_devID_infer 		= -1;
int g_devID_display 	= -1;
int  g_nChannels 		= 0;
int g_tileWidth			= 0;
int g_tileHeight		= 0;
int g_tilesInRow		= 0;
bool g_endlessLoop		= false;
bool g_fullScreen		= false;
bool g_gui                      = false;

char *g_fileList 		= nullptr;
char *g_deployFile 		= nullptr;
char *g_modelFile 		= nullptr;
char *g_meanFile 		= nullptr;
char *g_labelFile 		= nullptr;

char *g_calibrationTableFile = nullptr;
DATA_TYPE g_dataType = FLOAT;

__inline__ bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

class DecodeProfiler : public IDecodeProfiler {
public:	
	void reportDecodeTime(const int frameIdx, const int videoIdx, const int devID, double ms) {
		nCount++;
		timeElasped += ms;
		if ((frameIdx+1) % interval == 0) {
			LOG_DEBUG(logger, "Video[" << videoIdx << "] " 
								<< std::fixed << std::setprecision(2) 
								<< "Decoding" << " Performance: " << float(interval * 1000.0 / timeElasped) << " frames/second"
								<< " || Total Frames: " << nCount);
			timeElasped = 0.;
		}
	}
	
	int nCount{ 0 };
	int interval{ 100 };
	double timeElasped{ 0 }; // ms
};

class AnalysisProfiler : public IAnalysisProfiler {
public:
	explicit
	AnalysisProfiler(simplelogger::Logger *logger) {
		logger_ = logger;
	}
	void reportTime(double ms, int batchSize) override {
		nCount += batchSize;
		nTotalFrames += batchSize;
		timeElasped += ms;
		if (nCount >= interval) {
			LOG_DEBUG(logger_, "Analysis Pipeline" << std::fixed << std::setprecision(2) 
								<< " Performance: " << nCount * 1000.f / timeElasped << " frames/second"
								<< " || Total Frames: " << nTotalFrames);
			nCount = 0;
			timeElasped = 0;
		}
	}

private:
	simplelogger::Logger *logger_{ nullptr };
	
	int nCount{ 0 };
	int interval{ 200 };
	int nTotalFrames{ 0 };
	double timeElasped{ 0 }; // ms
};

bool parseArg(int argc, char **argv);
void getFileNames(const int nFiles, char *fileList, std::vector<std::string> &files);
void userPushPacket(DataProvider *pDataProvider, IDeviceWorker *pDeviceWorker, const int laneID);

std::vector<FileDataProvider *> vpDataProviders;
std::vector<DecodeProfiler *> g_vpDecProfilers;
AnalysisProfiler *g_analysisProfiler;

int main(int argc, char **argv) {

        deepStreamInit();

	bool ret = parseArg(argc, argv);
	if (!ret) {
		LOG_ERROR(logger, "Error in parseArg!");
		return 0;
	}

	// Init a worker on a GPU device
	IDeviceWorker *pDeviceWorker = createDeviceWorker(g_nChannels, g_devID_infer);
	
	// Add decode task
	pDeviceWorker->addDecodeTask(cudaVideoCodec_H264);
	
	// Add color space convertor
	IModule *pConvertor = pDeviceWorker->addColorSpaceConvertorTask(BGR_PLANAR);

	// Add inference task
	std::string inputLayerName("data");
	#ifdef RESNET10
	std::vector<std::string > outputLayerNames{"Layer7_cov", "Layer7_bbox"};
	#else
	std::vector<std::string > outputLayerNames{"Layer11_cov", "Layer11_bbox"};
	#endif
	
	inferenceParams param;
	param.dataType = g_dataType;
	if (INT8 == g_dataType) {
		param.calibrationTableFile_ = g_calibrationTableFile;
	}
	param.bScale_ = true; param.scale_ = 0.00392156f; param.shift_ = 0.f;
	PRE_MODULE preModule_infer{std::make_pair(pConvertor, 0)};
	IModule *pInfer = pDeviceWorker->addInferenceTask(preModule_infer,
														g_deployFile,
														g_modelFile,
														g_meanFile,
														inputLayerName,
														outputLayerNames,
														g_nChannels,
														&param);

	// Detection
	PRE_MODULE_LIST preModules_parser;
	preModules_parser.push_back(std::make_pair(pInfer, 0)); // cov
	preModules_parser.push_back(std::make_pair(pInfer, 1)); // bbox
	ParserModule *pParser = new ParserModule(preModules_parser,
												g_nChannels,
												g_devID_infer,
												logger);
	assert(nullptr != pParser);
	pDeviceWorker->addCustomerTask(pParser);
	
	KittiLoggerModule *pKitti = NULL;

	// Kitti logging of results
        PRE_MODULE_LIST preModules_kitti;
	preModules_kitti.push_back(std::make_pair(pConvertor, 1)); // NV12
	preModules_kitti.push_back(std::make_pair(pParser, 0)); // COORDS
	pKitti = new KittiLoggerModule(preModules_kitti,
			g_nChannels,
                        g_devID_display,
			g_devID_infer,
			g_labelFile, logger);
	assert(nullptr != pKitti);
	pDeviceWorker->addCustomerTask(pKitti);

		
	for (int i = 0; i < g_nChannels; ++i) {
		g_vpDecProfilers.push_back(new DecodeProfiler);
		pDeviceWorker->setDecodeProfiler(g_vpDecProfilers[i], i);
	}
	
	g_analysisProfiler = new AnalysisProfiler(logger);
	pDeviceWorker->setAnalysisProfiler(g_analysisProfiler);
	
	// start the device worker.
	pDeviceWorker->start();
		
	// what the users need to do is 
	// push video packets into a packet cache
	std::vector<std::thread > vUserThreads;
	for (int i = 0; i < g_nChannels; ++i) {
		vUserThreads.push_back( std::thread(userPushPacket,
											vpDataProviders[i],
											pDeviceWorker,
											i
											) );	
	}

	for (auto& th : vUserThreads) {
		th.join();
	}
	
	pDeviceWorker->stop();
	
	// free
	pDeviceWorker->destroy();
	
	while (!vpDataProviders.empty()) {
		FileDataProvider *temp = vpDataProviders.back();
		vpDataProviders.pop_back();
		delete temp;
	}
	for (int i = 0; i < g_nChannels; ++i) {
		delete g_vpDecProfilers[i];
	}
	if (nullptr != pParser) {
		delete pParser;
	}

	if (nullptr != logger) {
		delete logger;
	}
	if (nullptr != g_analysisProfiler) {
		delete g_analysisProfiler;
	}
        if (nullptr != pKitti) {
	        delete pKitti;
	}
	return 0;
}
	
void getFileNames(const int nFiles, char *fileList, std::vector<std::string> &files) {
	int count = 0;
	char *str;
	str = strtok(fileList, ",");
	while (NULL != str) {
		files.push_back(std::string(str));
		str = strtok(NULL, ",");
		count++;
		if (count >= nFiles) {
			break;
		}
	}
}

void userPushPacket(DataProvider *pDataProvider, IDeviceWorker *pDeviceWorker, const int laneID) {
	assert(NULL != pDeviceWorker);
	assert(NULL != pDataProvider);
	int nBuf = 0;
	uint8_t *pBuf = nullptr;
	int nPkts = 0;
	struct timeval timerOfLastPkt;
	struct timeval timerOfCurrPkt;
	
	gettimeofday(&timerOfLastPkt, NULL);
	while (true) {
		// get a frame packet from a video file
		int bStatus = pDataProvider->getData(&pBuf, &nBuf);
		if (bStatus == 0) {
			if (g_endlessLoop) {
				//LOG_DEBUG(logger, "User: Reloading...");
				pDataProvider->reload();
			} else {
				LOG_DEBUG(logger, "User: Ending...");
				// push the last NAL unit packet into deviceWorker
				pDeviceWorker->pushPacket(pBuf, nBuf, laneID);
				pDeviceWorker->stopPushPacket(laneID);
				break;
			}
		} else {
			gettimeofday(&timerOfCurrPkt, NULL);
			double t = (timerOfCurrPkt.tv_sec - timerOfLastPkt.tv_sec) * 1000.0
						+ (timerOfCurrPkt.tv_usec - timerOfLastPkt.tv_usec) / 1000.0;
			if (t < 40.0) {
				std::this_thread::sleep_for(std::chrono::milliseconds((int)(40.0-t))); // ms
			}
			gettimeofday(&timerOfLastPkt, NULL);
			// Push packet into deviceWorker.
			pDeviceWorker->pushPacket(pBuf, nBuf, laneID);
		}
	}
}
	
bool parseArg(int argc, char **argv) {
	bool ret = false;
        bool b_gui = false;	
	int nDevs = 0;
	cudaError_t err = cudaGetDeviceCount(&nDevs);
	if (0 == nDevs) {
		LOG_ERROR(logger, "Warning: No CUDA capable device!");
		exit(1);
	}
	assert(err == cudaSuccess);
	cudaDeviceProp deviceProp;
	
	g_devID_display = getCmdLineArgumentInt(argc, (const char **)argv, "devID_display");
	if (g_devID_display < 0 || g_devID_display >= nDevs) { 
		LOG_ERROR(logger, "Warning: No such GPU device!");
		return false; 
	}
    cudaGetDeviceProperties(&deviceProp, g_devID_display);

	LOG_DEBUG(logger, "Device ID for display [" << g_devID_display << "]: " << deviceProp.name);
	
	g_devID_infer = getCmdLineArgumentInt(argc, (const char **)argv, "devID_infer");
	if (g_devID_infer < 0 || g_devID_infer >= nDevs) { 
		LOG_ERROR(logger, "Warning: No such GPU device!");
		return false; 
	}
    cudaGetDeviceProperties(&deviceProp, g_devID_infer);
	LOG_DEBUG(logger, "Device ID for inference [" << g_devID_infer << "]: " << deviceProp.name);

	g_nChannels = getCmdLineArgumentInt(argc, (const char **)argv, "nChannels");
	if (g_nChannels <= 0) { return false; }
	LOG_DEBUG(logger, "Video channels: " << g_nChannels);
	
	ret = getCmdLineArgumentString(argc, (const char **)argv, "fileList", &g_fileList);
	if (!ret) {
		LOG_ERROR(logger, "Warning: No h264 files.");
		return false;
	}

	ret = getCmdLineArgumentString(argc, (const char **)argv, "deployFile", &g_deployFile);
	if (!ret) {
		LOG_ERROR(logger, "Warning: No deploy files.");
		return false;
	}
	
	ret = getCmdLineArgumentString(argc, (const char **)argv, "modelFile", &g_modelFile);
	if (!ret) {
		LOG_ERROR(logger, "Warning: No model files.");
		return false;
	}
	
	ret = getCmdLineArgumentString(argc, (const char **)argv, "meanFile", &g_meanFile);
	if (!ret) {
		LOG_ERROR(logger, "Warning: No mean files.");
	}
	
	ret = getCmdLineArgumentString(argc, (const char **)argv, "labelFile", &g_labelFile);
	if (!ret) {
		LOG_ERROR(logger, "Warning: No synset files.");
		return false;
	}
	
	ret = getCmdLineArgumentString(argc, (const char **)argv, "calibrationTableFile", &g_calibrationTableFile);
	if (!ret) {
		LOG_ERROR(logger, "Warning: No calibration table file.");
	}
	
	g_tileWidth = getCmdLineArgumentInt(argc, (const char **)argv, "tileWidth");
	if (g_tileWidth <= 0) { 
		LOG_ERROR(logger, "Warning: Illegal width!");
		return false; 
	}
	g_tileHeight = getCmdLineArgumentInt(argc, (const char **)argv, "tileHeight");
	if (g_tileHeight <= 0) { 
		LOG_ERROR(logger, "Warning: Illegal height!");
		return false; 
	}
	g_tilesInRow = getCmdLineArgumentInt(argc, (const char **)argv, "tilesInRow");
	if (g_tilesInRow <= 0) { 
		LOG_ERROR(logger, "Warning: Illegal number of tiles!");
		return false; 
	}
	
	g_fullScreen = (bool)getCmdLineArgumentInt(argc, (const char **)argv, "fullScreen");
	if (true == g_fullScreen) {
		LOG_DEBUG(logger, "Full screen playback.");
	}

	b_gui = (bool)getCmdLineArgumentInt(argc, (const char **)argv, "gui");
	if (true == b_gui) {
		LOG_DEBUG(logger, "GUI not supported for this sample. GUI disabled. KITTI log files will be generated.");
	} else {
	        LOG_DEBUG(logger, "GUI disabled. KITTI log files will be generated.");
	}

	g_endlessLoop = getCmdLineArgumentInt(argc, (const char **)argv, "endlessLoop");
	assert(0 == g_endlessLoop || 1 == g_endlessLoop);
	LOG_DEBUG(logger, "Endless Loop: " << g_endlessLoop);
	
	int type = getCmdLineArgumentInt(argc, (const char **)argv, "int8");
	if (1 == type) {
		g_dataType = INT8;
	} else {
		g_dataType = FLOAT;
	}
	
	// create data provider
	std::vector<std::string > vFiles;
	getFileNames(g_nChannels, g_fileList, vFiles);
	for (int i = 0; i < g_nChannels; ++i) {
	  vpDataProviders.push_back(new FileDataProvider(vFiles[i].c_str(), i));
	}
	
	return true;
}



