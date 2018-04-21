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
#include <vector>
#include <cstring>
#include <cassert>

#define StatIFrameInfo

class DataProvider {
public:
    virtual ~DataProvider() {}
    virtual bool getData(uint8_t **ppBuf, int *pnBuf) = 0;
	virtual void reload() = 0;
};

// Define the file data provider, parse each frame in raw video data
class FileDataProvider : public DataProvider {
public:
  FileDataProvider(const char *szFilePath, int index) {
        fp_ = fopen(szFilePath, "rb");
        if (NULL == fp_)
        {
            LOG_ERROR(logger, "Failed to open file "<<szFilePath);
            exit(1);
        }

    	pLoadBuf_ = new uint8_t[nLoadBuf_ + nLoadBufSafe_];
    	pPktBuf_  = new uint8_t[nPktBuf_];
	assert(NULL != pLoadBuf_);
	assert(NULL != pPktBuf_);
        IDInVideo  = 0;
        IDInIFrame = 0;
        dpIndex = index;
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
			nBytesToDecode = findIFrame();
			if (0 == nBytesToDecode) {
				// need to load more video data to find a frame
				int nBytesLoaded = loadDataFromFile(nLoadBuf_);

				if (0 == nBytesLoaded) {
                    // copy the residual data in pLoadBuf_ to packet buffer pPktBuf_
					memcpy(pPktBuf_, pLoadBuf_ + loadBuf_Start, (loadBuf_End - loadBuf_Start));
					*ppBuf = pPktBuf_;
					*pnBuf = loadBuf_End - loadBuf_Start;

                    // reset the load buffer pLoadBuf_
                    loadBuf_Start  = 0;
                    loadBuf_End    = 0;
                    loadBuf_Length = 0;
					return false;
				}
			} 
            else {
			}
		} while (nBytesToDecode == 0);

		memcpy(pPktBuf_, &pLoadBuf_[loadBuf_Start], nBytesToDecode);
		*ppBuf = pPktBuf_;
		*pnBuf = nBytesToDecode;
        
        // reset the searching point for next loop
        loadBuf_Start += nBytesToDecode; 
        loadBuf_Length = loadBuf_End - loadBuf_Start;

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

        loadBuf_Length = loadBuf_End - loadBuf_Start;
        if (loadBuf_Length)
        {
            memcpy(&pLoadBuf_[0], &pLoadBuf_[loadBuf_Start], loadBuf_Length);
            // update the state of load buffer
            loadBuf_Start = 0;
            loadBuf_End   = loadBuf_Length;
        }

        // note count is nLoadBuf_
		int nRead = fread(&pLoadBuf_[loadBuf_End], 1, count, fp_);
		if (0 == nRead) {
			return 0;
		} else {
            // add nRead bytes into load buffer
            // update the state of load buffer
            loadBuf_End    = loadBuf_End + nRead;
            loadBuf_Length = loadBuf_End - loadBuf_Start;

            return nRead;
		}
	}
	
    // this function works with start code 0x00000001 and/or 0x000001
    int findIFrame()
    {
        //start code of NAL nuit: 0x00000001 or 0x000001
        const int nBytesStartCode = 4; 

        loadBuf_Length = loadBuf_End - loadBuf_Start;
		if (loadBuf_Length < nBytesStartCode) {
			return 0;
		}

        int r;
        // 0 indicate initial state
        // 1 indicate 0x00
        // 2 indicate 0x0000
        // if 0x000001, find pattern and return
        int iState = 0;
        for (r=loadBuf_Start; r<loadBuf_End; ++r)
        {
            if (iState == 0){
                if (!pLoadBuf_[r])
                {
                    iState = 1; 
                }
                continue;
            }
            else if (iState == 1) {
                iState = pLoadBuf_[r] ? 0 : 2;
                continue;
            }
            else if (iState == 2) {
                if (!pLoadBuf_[r]) {
                    continue;
                } 
                // If true, find a matched pattern 0x000001 
                if (pLoadBuf_[r] == 1) {
                    if (firstStartCodeSeted_)
                    {
                        if ((pLoadBuf_[loadBuf_Start] & 0x1F) == 7)
                        {
                            if (!bSPS_)
                            {
                                bSPS_ = true;
                                LOG_DEBUG(logger, "Set SPS for smart decoding");

                                iState = 0;
                                return r - loadBuf_Start + 1;
                            }
                            else
                            {
                                iState = 0;
                                loadBuf_Start = r+1;

                                continue;
                            }
                        }
                        else if ((pLoadBuf_[loadBuf_Start] & 0x1F) == 8)
                        {
                            if (!bPPS_)
                            {
                                bPPS_ = true;
                                LOG_DEBUG(logger, "Set PPS for smart decoding");

                                iState = 0;
                                return r - loadBuf_Start + 1;
                            }
                            else
                            {
                                iState = 0;
                                loadBuf_Start = r+1;

                                continue;
                            }
                        }
                        else if ((pLoadBuf_[loadBuf_Start] & 0x1F) == 5) // indicate I frame
                        {
#ifdef StatIFrameInfo
                            IDInVideo++;
                            IDInIFrame++;
                            LOG_DEBUG(logger, "For video "<<dpIndex<<" the index of "<<IDInIFrame<<" I Frame in the video is "<<IDInVideo-1);

#endif
                            iState = 0;
                            return r-loadBuf_Start + 1;
                        }
                        else
                        {
#ifdef StatIFrameInfo
                            if (((pLoadBuf_[loadBuf_Start] & 0x1F) != 0) && ((pLoadBuf_[loadBuf_Start] & 0x1F) < 5))
                                IDInVideo++;
#endif
                            iState = 0;
                            loadBuf_Start  = r+1;
                            loadBuf_Length = loadBuf_End - loadBuf_Start;

                            continue;
                        }
                    }
                    else // send the first start code anyway
                    {
                        iState = 0;
                        firstStartCodeSeted_ = true;
                        return r-loadBuf_Start + 1;
                    }
                }
                iState = 0;
                continue;
            }
        }
        return 0;
    }

    
	FILE *fp_ = nullptr;
	uint8_t *pLoadBuf_ = nullptr;
    // load 64M data from file each time
	int nLoadBuf_     = 1 << 26;
    // 1MB for residual data
    int nLoadBufSafe_ = 1 << 20;

    int loadBuf_Start = 0;
    int loadBuf_Length= 0;
    int loadBuf_End   = 0;
	
	uint8_t *pPktBuf_ = nullptr;
	int nPktBuf_ = 1 << 20;

    bool firstStartCodeSeted_ = false;
    bool bSPS_ = false;
    bool bPPS_ = false;
    int IDInVideo;
    int IDInIFrame;
    int dpIndex;

};
