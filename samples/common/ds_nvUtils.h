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

#ifndef NV_UTILS_H
#define NV_UTILS_H

#include <chrono>
#include <logger.h>
#include <sys/stat.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

extern simplelogger::Logger *logger;

#ifdef __cuda_cuda_h__
inline bool CHECK_(CUresult e, int iLine, const char *szFile) {
    if (e != CUDA_SUCCESS) {
        LOG_ERROR(logger, "CUDA error " << e << " at line " << iLine << " in file " << szFile);
        return false;
    }
    return true;
}
#endif

#ifdef __CUDA_RUNTIME_H__
inline bool CHECK_(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        LOG_ERROR(logger, "CUDA runtime error " << e << " at line " << iLine << " in file " << szFile);
        return false;
    }
    return true;
}
#endif

#if defined(__gl_h_) || defined(__GL_H__)
inline bool CHECK_(GLenum e, int iLine, const char *szFile) {
    if (e != 0) {
        LOG_ERROR(logger, "GLenum error " << e << " at line " << iLine << " in file " << szFile);
        return false;
    }
    return true;
}
#endif

#define ck(call) CHECK_(call, __LINE__, __FILE__)

class StopWatch {
public:
    void Start() {
        t0 = std::chrono::high_resolution_clock::now();
    }
    double Stop() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch() - t0.time_since_epoch()).count() / 1.0e9;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> t0;
};


#endif // NV_UTILS_H
