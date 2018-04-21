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
#ifndef DATA_PROVIDER_H
#define DATA_PROVIDER_H

#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>

class DataProvider {
public:
    virtual ~DataProvider() {}
    virtual bool getData(uint8_t **ppBuf, int *pnBuf) = 0;
	virtual void reload() = 0;
};

// Define the file data provider
class FileDataProvider : public DataProvider {
public:
    FileDataProvider(const char *szFilePath, simplelogger::Logger *logger)
	: logger_(logger)
	{
        fp_ = fopen(szFilePath, "rb");
    	if (nullptr == fp_) {
			LOG_ERROR(logger, "Failed to open file " << szFilePath);
			exit(1);
		}
		pLoadBuf_ = new uint8_t[nLoadBuf_];
    	pPktBuf_ = new uint8_t[nPktBuf_];
		assert(nullptr != pLoadBuf_);
	}
    ~FileDataProvider() {
        if (fp_) {
            fclose(fp_);
        }
		if (pLoadBuf_) {
			delete [] pLoadBuf_;
		}
		if (pPktBuf_) {
			delete [] pPktBuf_;
		}
    }

    
	bool getData(uint8_t **ppBuf, int *pnBuf) {
        if (!fp_) {
            return 0;
        }
		// Warning: only support H264, HEVC
		int nBytesToDecode;
		do {
			nBytesToDecode = findEndOfFrame(vCache_);
			if (0 == nBytesToDecode) {
				// need to load more video data to find a frame
				int nBytesLoaded = loadDataFromFile(nLoadBuf_);
				if (0 == nBytesLoaded) {
					//std::cout << "User: File End.\n";
					//std::cout << "Warning: vCache.size() = " << vCache_.size() << std::endl;
					memcpy(pPktBuf_, vCache_.data(), vCache_.size());
					*ppBuf = pPktBuf_;
					*pnBuf = vCache_.size();
					
					vCache_.clear();
					return false;
				}
			} else {
				//std::cout << "Note: find a frame.\n";
				//std::cout << "Warning: vCache.size() = " << vCache_.size() << std::endl;
			}
		} while (nBytesToDecode == 0);
		
		//std::cout << "Warning: nBytesToDecode = " << nBytesToDecode << std::endl;
		assert(0 != nBytesToDecode);
		memcpy(pPktBuf_, vCache_.data(), nBytesToDecode);
		vCache_.erase(vCache_.begin(), vCache_.begin() + nBytesToDecode);
		*ppBuf = pPktBuf_;
		*pnBuf = nBytesToDecode;
		
		return true;
	}
	
	void reload() {
		fseek(fp_, 0, SEEK_SET);
	}

private:
	int loadDataFromFile(const int count) {
		if (NULL == fp_) {
			return 0;
		}
		int nRead = fread(pLoadBuf_, 1, count, fp_);
		if (0 == nRead) {
			return 0;
		} else {
			vCache_.insert(vCache_.end(), &pLoadBuf_[0], &pLoadBuf_[nRead]);
		}
		return vCache_.size();
	}
	
	int findEndOfFrame(std::vector<uint8_t > &vCache) {
        const int nBytesOfPrefix = 3;
		if (vCache.size() < nBytesOfPrefix) {
			return 0;
		}
		unsigned r = 0, sum = 0;
		for (r = 0; r < (vCache.size() - nBytesOfPrefix + 1); ++r) {
			if (0 == r) {
				for (int i = 0; i < nBytesOfPrefix; ++i) {
					sum += vCache[i];
				}
			} else {
				sum = sum - vCache[r-1] + vCache[r+nBytesOfPrefix-1];
			}
			if (1 == sum && 1 == vCache[r+nBytesOfPrefix-1]) {
				// h264 aud nail units
				if (r+nBytesOfPrefix != 6) {
					return r + nBytesOfPrefix;
				}
			}
		}
		return 0;
	}
    
	FILE *fp_{ nullptr };
	uint8_t *pLoadBuf_{ nullptr };
	int nLoadBuf_{ 1 << 20 };
	
	uint8_t *pPktBuf_{ nullptr };
	int nPktBuf_{ 1 << 20 };
	
	std::vector<uint8_t > vCache_;
	simplelogger::Logger *logger_{ nullptr };
};

/*
// Note: this implementation don't have H.264 parser
class FileDataProvider : public DataProvider {
public:
    FileDataProvider(const char *szFilePath, simplelogger::Logger *logger)
	: logger_(logger)
	{
        fp_ = fopen(szFilePath, "rb");
    	if (nullptr == fp_) {
			LOG_DEBUG(logger, "Fail to open file " << szFilePath);
			exit(1);
		}
		pLoadBuf_ = new uint8_t[nLoadBuf_];
		assert(NULL != pLoadBuf_);
	}
    ~FileDataProvider() {
        if (fp_) {
            fclose(fp_);
        }
		if (pLoadBuf_) {
			delete [] pLoadBuf_;
		}
    }
	bool getData(uint8_t **ppBuf, int *pnBuf) {
        if (!fp_) {
            return 0;
        }
		int nRead = fread(pLoadBuf_, 1, nLoadBuf_, fp_);
    	if (0 == nRead) {
			return false;
		}
		*ppBuf = pLoadBuf_;
		*pnBuf = nRead;
		return true;
	}
	
	void reload() {
		fseek(fp_, 0, SEEK_SET);
	}

private:
    FILE *fp_ = NULL;
	uint8_t *pLoadBuf_{ nullptr };
	int nLoadBuf_{ 1 << 20 };
	simplelogger::Logger *logger_{ nullptr };
};
*/

#endif // DATA_PROVIDER_H
