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

#include "presenterGL.h"
#include <nvToolsExt.h>

template <class T>
std::string convert(T src) {
	std::stringstream ss;
	ss << src;
	return ss.str();
}

PresenterGL *PresenterGL::pInstance;

void PresenterGL::ThreadProc(PresenterGL *This) {
    This->Run();
}

void PresenterGL::DisplayProc() {
    pInstance->Display();
}

PresenterGL::PresenterGL(int devForDisplay, int nSubWindowWidth, int nSubWindowHeight, int nChannels, int nSubWindowsPerRow, std::vector<std::string > &vSynsets, bool bFullScreen, demo_button *pDemoButton) {
	CUcontext pctx;
	ck(cuCtxCreate(&pctx, 0, devForDisplay)); 

	this->cuContext 		= pctx;	
	this->nChannels 		= nChannels;
	this->nSubWindowsPerRow = nSubWindowsPerRow;
	this->nSubWindowWidth 	= nSubWindowWidth;
	this->nSubWindowHeight 	= nSubWindowHeight;
	this->bFullScreen 		= bFullScreen;
	
	// nx is the subwindow number in the x axis
	int nx = nChannels > nSubWindowsPerRow ? nSubWindowsPerRow : nChannels;
	int ny = (nChannels + nx - 1) / nx;
	
	this->nWindowWidth  = nx * nSubWindowWidth;
	this->nWindowHeight = ny * nSubWindowHeight;
		
	for (int i = 0; i < nChannels; ++i) {
		int channel = i;
		int id_x = channel % nx;
		int id_y = channel / nx;
		
		coordinate coord; // coordinate of the sub window vertex
		coord.x = nSubWindowWidth  * id_x;
		coord.y = nSubWindowHeight * id_y;
		vertexs.push_back(std::make_pair(BBOXS_PER_FRAME(), coord));
	}
	
	
	for (int i = 0; i < vSynsets.size(); ++i) {
		synsets.push_back(vSynsets[i]);
	}
	ck(cuStreamCreate(&cuStream, CU_STREAM_NON_BLOCKING));
	
	this->pDemoButton = pDemoButton;

	pthMessageLoop = new std::thread(ThreadProc, this);
    while (!pInstance) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

PresenterGL::~PresenterGL() {
    glutLeaveMainLoop();
    pthMessageLoop->join();
	delete pthMessageLoop;
	ck(cuStreamDestroy(cuStream));
}

void PresenterGL::Lock() {
    pInstance->mutex.lock();

}

void PresenterGL::Unlock() {
	pInstance->mutex.unlock();
}

void PresenterGL::DeviceFrameBuffer(uint8_t **ppFrame, int *pnPitch, int channel) {
	coordinate coord = vertexs[channel].second;
	int stride = coord.x * 4 + coord.y * nWindowWidth * 4; // BGRA -> 4 byte
	
	*ppFrame = (uint8_t *)dpFrame + stride;
    *pnPitch = (int)(nFrameSize / nWindowHeight);
}

int PresenterGL::GetWindowWidth() {
	return nWindowWidth;
}

int PresenterGL::GetWindowHeight() {
	return nWindowHeight;
}

int PresenterGL::GetSubWindowWidth() {
	return nSubWindowWidth;
}

int PresenterGL::GetSubWindowHeight() {
	return nSubWindowHeight;
}

uint8_t *PresenterGL::GetWindowBufferPointer() {
	return (uint8_t *)dpFrame;
}

void PresenterGL::SetText(BBOXS_PER_FRAME& bboxs, int subWindowID) {
	if (subWindowID >= vertexs.size()) {
		exit(-1);
	}
	
	BBOXS_PER_FRAME & bboxs_1 = vertexs[subWindowID].first;
	bboxs_1 = bboxs;
}

BBOXS_PER_FRAME& PresenterGL::GetBboxs(const int subWindowID) {
	return vertexs[subWindowID].first;
}

void PresenterGL::keyboardProc(unsigned char key, int x, int y) {
	switch(key) {
		// key 'b' to begin demo
		case 'b':
			pInstance->pDemoButton->button_begin = true;
			break;
		// key 't' to turn on/off text display
		case 't':
			pInstance->pDemoButton->button_text = !(pInstance->pDemoButton->button_text);
			break;
		// key 'ESC' to end demo
		case 27:
			glutDestroyWindow(glutGetWindow());	
			return;
			break;
	}
}

void PresenterGL::Run() {
	int argc1 = 1;
    const char *argv1[] = {"dummy"};
    glutInit(&argc1, (char **)argv1);
    
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(nWindowWidth, nWindowHeight);
    glutCreateWindow("Object Detection Sample");
	glutKeyboardFunc(pInstance->keyboardProc);
	if (bFullScreen) {
		glutFullScreen();
	}

    glViewport(0, 0, nWindowWidth, nWindowHeight);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glewInit();
    glGenBuffersARB(1, &fbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, fbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, nWindowWidth * nWindowHeight * 4, NULL, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex);
    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA8, nWindowWidth, nWindowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

    static const char *code =
        "!!ARBfp1.0\n"
        "TEX result.color, fragment.texcoord, texture[0], RECT; \n"
        "END";
    glGenProgramsARB(1, &shader);
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
    glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(code), (GLubyte *)code);

    ck(cuCtxSetCurrent(cuContext));
    ck(cuGraphicsGLRegisterBuffer(&cuResource, fbo, 0));
	
	nFrameSize = nWindowWidth * nWindowHeight * 4 * sizeof(uint8_t);
	ck(cuMemAlloc(&dpFrame, nFrameSize));	
	ck(cuMemsetD8(dpFrame, 0, nFrameSize)); // black background
	
    pInstance = this;
    glutDisplayFunc(DisplayProc);
    glutMainLoop();
    pInstance = NULL;
	ck(cuMemFree(dpFrame));
	ck(cuGraphicsUnregisterResource(cuResource));

    glDeleteBuffersARB(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteProgramsARB(1, &shader);
}

void PresenterGL::Display(void) {
	pInstance->mutex.lock();
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
    CUdeviceptr dpImageData = 0;
    size_t nSize = 0;
    ck(cuGraphicsMapResources(1, &cuResource, 0));
    ck(cuGraphicsResourceGetMappedPointer(&dpImageData, &nSize, cuResource));
   	
	CUDA_MEMCPY2D m = { 0 };
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = dpFrame;
    m.srcPitch = nWindowWidth * 4;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstDevice = dpImageData;
    m.dstPitch = nSize / nWindowHeight;
    m.WidthInBytes = nWindowWidth * 4;
    m.Height = nWindowHeight;
	cuMemcpy2DAsync(&m, cuStream);
	ck(cuStreamSynchronize(cuStream));
	ck(cuGraphicsUnmapResources(1, &cuResource, 0));
	pInstance->mutex.unlock();
	
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, fbo);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, nWindowWidth, nWindowHeight, GL_BGRA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    
	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    glTexCoord2f(0, (GLfloat)nWindowHeight);
    glVertex2f(0, 0);
    glTexCoord2f((GLfloat)nWindowWidth, (GLfloat)nWindowHeight);
    glVertex2f(1, 0);
    glTexCoord2f((GLfloat)nWindowWidth, 0);
    glVertex2f(1, 1);
    glTexCoord2f(0, 0);
    glVertex2f(0, 1);
    glEnd();
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
    glDisable(GL_FRAGMENT_PROGRAM_ARB);
	
	if (pInstance->pDemoButton->button_text) {
		mutex.lock();
		int h_realtime = glutGet(GLUT_WINDOW_HEIGHT);
		int w_realtime = glutGet(GLUT_WINDOW_WIDTH);
		
		for (size_t i = 0; i < vertexs.size(); ++i) {
			coordinate coord = vertexs[i].second;
			BBOXS_PER_FRAME bboxs = vertexs[i].first;
			int videoIndex = bboxs.videoIndex;
			if (videoIndex < 0) {
				continue;
			}
			int nBBox = bboxs.nBBox;
			for (size_t iBox = 0; iBox < nBBox; ++iBox) {
				if (!bboxs.bbox[iBox].bSkip) {
					int x = (int)((float)(coord.x+bboxs.bbox[iBox].x*nSubWindowWidth) * (float)w_realtime / nWindowWidth);
					int y = (int)((float)(coord.y+bboxs.bbox[iBox].y*nSubWindowHeight - 10) * (float)h_realtime / nWindowHeight);
					
					std::string strLabel;
					strLabel = synsets[bboxs.bbox[iBox].category];
					PrintText(2, strLabel, x, y, true);
				}
			}
		}
		mutex.unlock();
	}
    glutSwapBuffers();
    glutPostRedisplay();
	
}


void PresenterGL::PrintText(int iFont, std::string &strText, int x, int y, bool bFillBackground) {
    struct {void *font; int d1; int d2;} fontData[] = {
        /*0*/ GLUT_BITMAP_9_BY_15,        13, 4,
        /*1*/ GLUT_BITMAP_8_BY_13,        11, 4,
        /*2*/ GLUT_BITMAP_TIMES_ROMAN_10, 9,  3,
        /*3*/ GLUT_BITMAP_TIMES_ROMAN_24, 20, 7,
        /*4*/ GLUT_BITMAP_HELVETICA_10,   10, 3,
        /*5*/ GLUT_BITMAP_HELVETICA_12,   11, 4,
        /*6*/ GLUT_BITMAP_HELVETICA_18,   16, 5,
    };
    const int nFont = sizeof(fontData) / sizeof(fontData[0]);

    if (iFont >= nFont) {
        iFont = 0;
    }
    void *font = fontData[iFont].font;
    int d1 = fontData[iFont].d1, d2 = fontData[iFont].d2, d = d1 + d2, 
        w = glutGet(GLUT_WINDOW_WIDTH), h = glutGet(GLUT_WINDOW_HEIGHT);
	
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, w, 0.0, h, 0.0, 1.0);

	std::stringstream ss(strText);
    std::string str;
    int iLine = 0;
    while (std::getline(ss, str)) {
    	glColor3f(0.43, 0.29, 0.37); // gray background
        if (bFillBackground) {
			glRasterPos2i(x, h - y - iLine * d - d1);
    	    for (char c : str) {
    	        glutBitmapCharacter(font, c);
    	    }
    	    GLint pos[4];
    	    glGetIntegerv(GL_CURRENT_RASTER_POSITION, pos);
    	    glRecti(x, h - y - iLine * d, pos[0], h - y - (iLine + 1) * d);
    	    glColor3f(1, 1, 1); //white words
		}
        glRasterPos2i(x, h - y - iLine * d - d1);
        for (char c : str) {
            glutBitmapCharacter(font, c);
        }
        iLine++;
    }
	glPopMatrix();
}
