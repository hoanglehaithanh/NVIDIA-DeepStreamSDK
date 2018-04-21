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

#ifndef PRESENTER_GL_H
#define PRESENTER_GL_H

#pragma once

#include <string.h>
#include <stdlib.h>
#include <mutex>
#include <thread>
#include <chrono>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include <cuda.h>
#include <cudaGL.h>
#include "ds_nvUtils.h"
#include "logger.h"
#include <cuda_runtime.h>
#include "module.h"
typedef struct {
	int x;
	int y;
} coordinate;

typedef struct {
	bool button_begin = false;
	bool button_text = true;
} demo_button;

class PresenterGL
{
public:
	PresenterGL(int devForDisplay, int nSubWindowWidth, int nSubWindowHeight, int nChannels, int nSubWindowsPerRow, std::vector<std::string > &vSynsets, bool bFullScreen, demo_button *pDemoButton);
    ~PresenterGL();
    void DeviceFrameBuffer(uint8_t **ppFrame, int *pnPitch, int channel);
    void Lock();
	void Unlock();
	void SetText(BBOXS_PER_FRAME& bboxs, int subWindowID);
	BBOXS_PER_FRAME& GetBboxs(const int subWindowID);
	void SetDisplayFPS(float fps);
	void SetInferFPS(float fps);
	void SetDecFPS(float fps);
	int GetWindowWidth();
	int GetWindowHeight();
	int GetSubWindowWidth();
	int GetSubWindowHeight();
	uint8_t *GetWindowBufferPointer();

private:
    static void ThreadProc(PresenterGL *This);
    static void DisplayProc();
	static void keyboardProc(unsigned char key, int x, int y);
	
	void Run();
    void Display(void);
    void PrintText(int iFont, std::string &strText, int x, int y, bool bFillBackground);
	
private:
    // Display size
	int nSubWindowWidth = 0, nSubWindowHeight = 0;
	int nWindowHeight = 0, nWindowWidth = 0;
	int nSubWindowsPerRow = 0;
	int nChannels = 0;
	int interval = 0;
	bool bFullScreen = 0;
	
	std::vector<std::pair<BBOXS_PER_FRAME , coordinate> > vertexs;
	std::vector<std::string > synsets;

	CUstream cuStream;
	size_t nFrameSize = 0;
    CUcontext cuContext = NULL;
    CUgraphicsResource cuResource;
    CUdeviceptr dpFrame = 0;
    std::thread *pthMessageLoop = NULL;
    std::mutex mutex;
	GLuint fbo;
    GLuint tex;
    GLuint shader;
	static PresenterGL *pInstance;
	volatile demo_button *pDemoButton = nullptr;
};

#endif
