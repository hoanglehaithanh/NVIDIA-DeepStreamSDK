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

#include "ds_nvUtils.h"

class DataProvider {
public:
    virtual ~DataProvider() {}
    virtual bool getData(uint8_t **ppBuf, int *pnBuf) = 0;
	virtual void reload() = 0;
};

// Define the file data provider
class FileDataProvider : public DataProvider {
public:
    FileDataProvider(const char *szFilePath) {
        fp_ = fopen(szFilePath, "rb");
        if (!fp_) {
			LOG_ERROR(logger, "File descriptor is empty!");
        	exit(1);
		}
    	pBuf_ = new uint8_t[nBuf_];
	}
    ~FileDataProvider() {
        if (fp_) {
            fclose(fp_);
        }
		if (pBuf_) {
			delete[] pBuf_;
		}
    }
    bool getData(uint8_t **ppBuf, int *pnBuf) override {
        if (!fp_) {
			LOG_ERROR(logger, "File descriptor is empty!");
        	exit(1);
		}
        int nRead = fread(pBuf_, 1, nBuf_, fp_);
		if (0 == nRead) {
			*ppBuf = nullptr;
			*pnBuf = 0;
            return false;
		}

		*ppBuf = pBuf_;
		*pnBuf = nRead;
		return true;
    }
	
	void reload() {
		fseek(fp_, 0, SEEK_SET);
	}

private:
    FILE *fp_ = nullptr;
	uint8_t *pBuf_ = nullptr;
	const int nBuf_ = 1 << 20; // 1MB
};

#endif // DATA_PROVIDER_H
