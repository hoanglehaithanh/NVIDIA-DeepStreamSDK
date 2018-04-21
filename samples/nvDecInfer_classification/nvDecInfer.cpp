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

#include "ds_nvUtils.h"
#include "dataProvider.h"
#include "deepStream.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

int  g_nChannels 		= 0;
char *g_fileList 		= nullptr;
char *g_deployFile 		= nullptr;
char *g_modelFile 		= nullptr;
char *g_meanFile 		= nullptr;
char *g_synsetFile 		= nullptr;
char *g_validationFile 	= nullptr; 
bool g_endlessLoop		= false;
const char *g_appName	= "nvDecInfer";
cudaVideoCodec g_codec;

class DecodeProfiler;
class AnalysisProfiler;
class UserDefinedModule;

template <typename T>
inline std::string convert(const T a);
bool parseArg(int argc, char **argv);
inline bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs);
void getFileNames(const int nFiles, char *fileList, std::vector<std::string> &files);
void userPushPacket(DataProvider *pDataProvider, IDeviceWorker *pDeviceWorker, const int channel);

std::vector<FileDataProvider *> vpDataProviders;
std::vector<DecodeProfiler *> g_vpDecProfilers;
AnalysisProfiler *g_analysisProfiler;

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

class UserDefinedModule : public IModule {
public:
	explicit
	UserDefinedModule(PRE_MODULE_LIST &preModules,
					char* valFile,
					char* synsetFile,
					const int nChannels,
					simplelogger::Logger *logger) 
	: preModules_(preModules), valFile_(valFile), synsetFile_(synsetFile),
	nChannels_(nChannels), logger_(logger) {}

	~UserDefinedModule() {}

	// override
	void initialize() override {
		vFrameCount_.resize(nChannels_);
		vHitCount_.resize(nChannels_);

		// validation initialize
		std::ifstream iVal(valFile_);
		if (iVal.is_open()) {
			std::string line;
			int count = 0;
			while (std::getline(iVal, line)) {
				int num = std::stoi(line, nullptr, 10);
				vVals_.push_back(num);
				count++;
			}
			iVal.close();
		} else {
			LOG_ERROR(logger_, "Failed to open validation file " << valFile_);
			exit(0);
		}
		// synset initialize
		std::ifstream iLabel(synsetFile_);
		if (iLabel.is_open()) {
			std::string line;
			while (std::getline(iLabel, line)) {
				vSynsets_.push_back(line);
				line.clear();
			}
			iLabel.close();
		} else {
			LOG_ERROR(logger_, "Failed to open synset file " << synsetFile_);
			exit(0);
		}
	}

	void execute(const ModuleContext& context, const std::vector<IStreamTensor *>& vpInputTensors,  const std::vector<IStreamTensor *>& vpOutputTensors) {
		assert(preModules_.size() == vpInputTensors.size());

		// batch size changes in runtime	
		std::vector<int> shape = vpInputTensors[0]->getShape();
		int batch_rt = shape[0];
		assert(batch_rt <= nChannels_);
		
		const float *data = reinterpret_cast<const float*>(vpInputTensors[0]->getConstCpuData());
		std::vector<TRACE_INFO> vTraceInfos = vpInputTensors[0]->getTraceInfos();
		assert(batch_rt == vTraceInfos.size());
		
		for (int iBatch = 0; iBatch < batch_rt; ++iBatch) {
			TRACE_INFO &traceInfo = vTraceInfos[iBatch];

			int frameIndex = traceInfo.frameIndex;
			int videoIndex = traceInfo.videoIndex;
			int nDataLen = shape[1] * shape[2] * shape[3];
			const float *pData = data + iBatch * nDataLen;
			
			// user defined post processing of inference
			int TOP_N = 5;
			std::string sFileName = "./log/log_";
			sFileName += convert<int>(videoIndex) + ".txt";
			std::ofstream logFile;
				
			if (frameIndex == 0) {
				// clear the content of file
				logFile.open(sFileName, std::ios::trunc);
			} else {
				logFile.open(sFileName, std::ios::app | std::ios::binary);
			}

			if (!logFile.is_open()) {
				LOG_ERROR(logger, "Failed to Open file " << sFileName);
				exit(0);
			}
			// check the inference result
			// Find the top N classes
			std::vector<std::pair<float, int> > inferResultWithIndex;
			for (int j = 0; j < nDataLen; ++j) {
				inferResultWithIndex.push_back(std::make_pair(pData[j], j));
			}
			
			std::partial_sort(inferResultWithIndex.begin(),
								inferResultWithIndex.begin() + TOP_N,
								inferResultWithIndex.end(), PairCompare);// Top N
			
			bool check = false;
			
			logFile << ">>[Inference][video " << videoIndex << "][frame "
					  <<  frameIndex << "]" << " belongs to: \n";
				
			for (int j = 0; j < TOP_N; ++j) {
				std::stringstream ss;
				ss << "  " << std::fixed << std::setprecision(4) 
							<< inferResultWithIndex[j].first << " - \"" 
							<< vSynsets_[inferResultWithIndex[j].second] << "\"";
				
				if (vVals_[frameIndex % (vVals_.size())] == inferResultWithIndex[j].second) {
					check = true;
					ss << ", [Hit]";
					vHitCount_[videoIndex]++;
				}
				ss << '\n';
				logFile << ss.str();
			}
				
			vFrameCount_[videoIndex]++;
			logFile << "  Video[" << videoIndex << "] total frames = " << vFrameCount_[videoIndex] << "\n";
			logFile << "  Video[" << videoIndex << "] hit frames = " << vHitCount_[videoIndex] << "\n";
			logFile << "  Video[" << videoIndex << "] accuracy = "
					<< (float)vHitCount_[videoIndex]/vFrameCount_[videoIndex] 
					<< "\n\n";
	
			logFile.close();
		}
	
		// No module is connected to this module, so the output tensor is empty
	}
	
	void destroy() override {}

	int getNbInputs() const override {
		return preModules_.size();
	}
	
	PRE_MODULE getPreModule(const int tensorIndex) const override {
		return preModules_[tensorIndex];
	}
	
	int getNbOutputs() const override {
		return 0;
	}
	
	IStreamTensor* getOutputTensor(const int tensorIndex) const override {
		return nullptr;
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
	char *valFile_{ nullptr };
	char *synsetFile_{ nullptr };
	simplelogger::Logger *logger_{ nullptr };

	std::vector<std::string > 			vSynsets_;
	std::vector<int > 					vVals_;
	std::vector<int > 					vFrameCount_;
	std::vector<int > 					vHitCount_;
	
	void *pUserData_{ nullptr };
	MODULE_CALLBACK callback_{ nullptr };
	IModuleProfiler* pProfiler_{ nullptr };	
	PRE_MODULE_LIST preModules_;
	std::vector<IStreamTensor*> vpOutputTensors_;
};

int main(int argc, char **argv) {
        deepStreamInit();
	
	bool ret = parseArg(argc, argv);
	if (!ret) {
		LOG_ERROR(logger, "Error in parseArg!");
		return 0;
	}
	
	// set a GPU device ID
	int devID = 0; 

	// Create a worker on a GPU device
	IDeviceWorker *pDeviceWorker = createDeviceWorker(g_nChannels, devID);
	
	// Add decode task
	pDeviceWorker->addDecodeTask(g_codec);
		
	// Add frame paser
	IModule *pConvertor = pDeviceWorker->addColorSpaceConvertorTask(BGR_PLANAR);

	// Add inference task
	std::string inputLayerName("data");
	std::vector<std::string > outputLayerNames{"prob"};
	IModule *pInferModule = pDeviceWorker->addInferenceTask( std::make_pair(pConvertor, 0),
																g_deployFile,
																g_modelFile,
																g_meanFile,
																inputLayerName,
																outputLayerNames,
																g_nChannels);
	
	PRE_MODULE_LIST preModules{std::make_pair(pInferModule,0)};
	UserDefinedModule *pAccurancyModule = new UserDefinedModule(preModules, g_validationFile, g_synsetFile, g_nChannels, logger);
	assert(nullptr != pAccurancyModule);
	pDeviceWorker->addCustomerTask(pAccurancyModule);
	
	for (int i = 0; i < g_nChannels; ++i) {
		g_vpDecProfilers.push_back(new DecodeProfiler);
		pDeviceWorker->setDecodeProfiler(g_vpDecProfilers[i], i);
	}

	g_analysisProfiler = new AnalysisProfiler(logger);
	pDeviceWorker->setAnalysisProfiler(g_analysisProfiler);
	
	// start the device worker.
	pDeviceWorker->start();
	
	// User push video packets into a packet cache
	std::vector<std::thread > vUserThreads;
	for (int i = 0; i < g_nChannels; ++i) {
		vUserThreads.push_back( std::thread(userPushPacket,
											vpDataProviders[i],
											pDeviceWorker,
											i
											) );
	}
	
	// wait for user push threads
	for (auto& th : vUserThreads) {
		th.join();
	}
	
	// stop device worker
	pDeviceWorker->stop();
	
	LOG_DEBUG(logger, "Note:");
	LOG_DEBUG(logger, "[1] The speed of video input is set to 30 FPS.");
	LOG_DEBUG(logger, "[2] The inference results are recorded in the files under the './log' directory.");

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
	if (nullptr != g_analysisProfiler) {
		delete g_analysisProfiler;
	}
	if (nullptr != pAccurancyModule) {
		delete pAccurancyModule;
	}
	if (nullptr != logger) {
		delete logger;
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

void userPushPacket(DataProvider *pDataProvider, IDeviceWorker *pDeviceWorker, const int channel) {
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
			  	pDeviceWorker->pushPacket(pBuf, nBuf, channel);
				pDataProvider->reload();
			} else {
				//LOG_DEBUG(logger, "User: Ending...");
				// push the last NAL unit packet into deviceWorker
				pDeviceWorker->pushPacket(pBuf, nBuf, channel);
				pDeviceWorker->stopPushPacket(channel);
				break;
			}
		} else {
			gettimeofday(&timerOfCurrPkt, NULL);
			double t = (timerOfCurrPkt.tv_sec - timerOfLastPkt.tv_sec) * 1000.0
						+ (timerOfCurrPkt.tv_usec - timerOfLastPkt.tv_usec) / 1000.0;
			// 30 FPS
			if (t < 33.0) {
				std::this_thread::sleep_for(std::chrono::milliseconds((int)(33.0-t))); // ms
			}
			gettimeofday(&timerOfLastPkt, NULL);
			// Push packet into deviceWorker.
			pDeviceWorker->pushPacket(pBuf, nBuf, channel);
			
		}
	}
}
	
bool parseArg(int argc, char **argv) {
	bool ret = false;
	
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
		return false;
	}
	
	ret = getCmdLineArgumentString(argc, (const char **)argv, "synsetFile", &g_synsetFile);
	if (!ret) {
		LOG_ERROR(logger, "Warning: No synset files.");
		return false;
	}
	
	ret = getCmdLineArgumentString(argc, (const char **)argv, "validationFile", &g_validationFile);
	if (!ret) {
		LOG_ERROR(logger, "Warning: No validation files.");
	}
	
	g_endlessLoop = getCmdLineArgumentInt(argc, (const char **)argv, "endlessLoop");
	assert(0 == g_endlessLoop || 1 == g_endlessLoop);
	LOG_DEBUG(logger, "Endless Loop: " << g_endlessLoop);
	
	std::vector<std::string > vFiles;
	getFileNames(g_nChannels, g_fileList, vFiles);
	g_codec = getVideoFormat(vFiles[0].c_str()).codec;

	for (int i = 0; i < g_nChannels; ++i) {
		vpDataProviders.push_back(new FileDataProvider(vFiles[i].c_str()));
	}

	return true;
}

template <typename T>
inline std::string convert(const T a) {
	std::stringstream ss;
	ss << a;
	return ss.str();
}

inline bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

