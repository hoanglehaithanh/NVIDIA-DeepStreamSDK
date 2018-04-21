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

#include <iostream>
#include <thread>
#include <iomanip>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda.h>

#include "deepStream.h"
#include "dataProvider.h"
#include "ds_nvUtils.h"
#include "helper_cuda.h"

int g_devID 			= 0;
int g_nChannels 		= 0;
int g_endlessLoop		= false;
char *g_fileList		= nullptr;
cudaVideoCodec g_codec;

class DecodeProfiler : public IDecodeProfiler {
public:	
	void reportDecodeTime(const int frameIndex, const int channel, const int deviceId, double ms) {
		nCount++;
		timeElasped += ms;
		if ((frameIndex+1) % interval == 0) {
			LOG_DEBUG(logger, "Video [" << channel << "]:  " 
								<< std::fixed << std::setprecision(2) 
								<< "Decode Performance: " << float(interval * 1000.0 / timeElasped) << " frames/second"
								<< " || Decoded Frames: " << nCount);
			timeElasped = 0.;
		}
	}
	
	int nCount = 0;
	const int interval = 500;
	double timeElasped = 0.; // ms
};

std::vector<DecodeProfiler *> g_vpDecProfilers;
std::vector<FileDataProvider *> vpDataProviders;
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

bool parseArg(int argc, char **argv);
void getFileNames(const int nFiles, char *fileList, std::vector<std::string> &files);

// user interface of input
void userPushPacket(DataProvider *pDataProvider, IDeviceWorker *pDeviceWorker, const int channel);

int main(int argc, char **argv) {

        deepStreamInit();

	bool ret = parseArg(argc, argv);
	if (!ret) {
		LOG_ERROR(logger, "Error in parseArg!");
		return 0;
	}
	
	// Init a worker on a GPU device
	IDeviceWorker *pDeviceWorker = createDeviceWorker(g_nChannels, g_devID);
	
	// Add decode task
	pDeviceWorker->addDecodeTask(g_codec);
	
	for (int i = 0; i < g_nChannels; ++i) {
		g_vpDecProfilers.push_back(new DecodeProfiler);
		pDeviceWorker->setDecodeProfiler(g_vpDecProfilers[i], i);
	}
	
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
	
	// free
	while (!vpDataProviders.empty()) {
		FileDataProvider *temp = vpDataProviders.back();
		vpDataProviders.pop_back();
		delete temp;
	}
	
	pDeviceWorker->destroy();

	for (int i = 0; i < g_nChannels; ++i) {
		delete g_vpDecProfilers[i];
	}
	
	return 0;
}


bool parseArg(int argc, char **argv) {
	bool ret;

	int nDevs = 0;
	cudaError_t err = cudaGetDeviceCount(&nDevs);
	if (0 == nDevs) {
		LOG_ERROR(logger, "Warning: No CUDA capable device!");
		exit(1);
	}
	assert(err == cudaSuccess);

	g_devID = getCmdLineArgumentInt(argc, (const char **)argv, "devID");
	if (g_devID < 0 || g_devID >= nDevs) { 
		LOG_ERROR(logger, "Warning: No such GPU device!");
		return false; 
	}
	LOG_DEBUG(logger, "Device ID: " << g_devID);
	
	g_nChannels = getCmdLineArgumentInt(argc, (const char **)argv, "channels");
	if (g_nChannels <= 0) { return false; }
	LOG_DEBUG(logger, "Video channels: " << g_nChannels);
	
	g_endlessLoop = getCmdLineArgumentInt(argc, (const char **)argv, "endlessLoop");
	assert(0 == g_endlessLoop || 1 == g_endlessLoop);
	LOG_DEBUG(logger, "Endless Loop: " << g_endlessLoop);
	
	ret = getCmdLineArgumentString(argc, (const char **)argv, "fileList", &g_fileList);
	if (!ret) {
		LOG_ERROR(logger, "Warning: No h264 files.");
		return false;
	}
	
	std::vector<std::string > vFiles;
	getFileNames(g_nChannels, g_fileList, vFiles);
	g_codec = getVideoFormat(vFiles[0].c_str()).codec;

	for (int i = 0; i < g_nChannels; ++i) {
		vpDataProviders.push_back(new FileDataProvider(vFiles[i].c_str()));
	}
	
	return true;
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
				pDeviceWorker->pushPacket(pBuf, nBuf, channel);
				pDeviceWorker->stopPushPacket(channel);
				break;
			}
		} else {
			// Push packet into deviceWorker.
			pDeviceWorker->pushPacket(pBuf, nBuf, channel);
		}
	}
}

