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

#include "drawBbox.h"


__global__ void drawBoundingBox_kernel(uchar4 *pBGRA, const int nWidth, const int nHeight, const int stride, const int x_min, const int y_min, const int x_max, const int y_max) {
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int nPixels = 1;
	int x_min_l = x_min > 0 ? x_min : 0;
	int x_max_l = x_max > nWidth ? nWidth : x_max;
	int y_min_l = y_min > 0 ? y_min : 0;
	int y_max_l = y_max > nHeight ? nHeight : y_max;

	if (nPixels > x_max_l - x_min_l) {
		nPixels = 0;
	}
	if (nPixels > y_max_l - y_min_l) {
		nPixels = 0;
	}
	
	if (idx_x > x_max_l || idx_y > y_max_l) {
		return;
	}

	// left and right
	bool bDraw = ((idx_x >= x_min_l && idx_x <= (x_min_l+nPixels))
				|| (idx_x >= x_max_l-nPixels && idx_x <= x_max_l))
				&& (idx_y >= y_min_l && idx_y <= y_max_l);
	if (bDraw) {
		pBGRA[idx_y*stride+idx_x].x = 0;
		pBGRA[idx_y*stride+idx_x].y = 0;
		pBGRA[idx_y*stride+idx_x].z = 255;
	}

	// up and down
	bDraw = ((idx_y >= y_min_l && idx_y <= y_min_l + nPixels)
			|| (idx_y >= y_max_l - nPixels && idx_y <= y_max_l))
			&& (idx_x >= x_min_l + nPixels && idx_x <= x_max_l - nPixels);
	if (bDraw) {
		pBGRA[idx_y*stride+idx_x].x = 0;
		pBGRA[idx_y*stride+idx_x].y = 0;
		pBGRA[idx_y*stride+idx_x].z = 255;
	}
}

void drawBoundingBox_cuda(uint8_t *pBGRA, const int nWidth, const int nHeight, const int nBgraPitch, const int x_min, const int y_min, const int x_max, const int y_max, cudaStream_t stream) {
	drawBoundingBox_kernel<<<dim3((nWidth+15)/16, (nHeight+15)/16), dim3(16, 16), 0, stream>>>((uchar4 *)pBGRA, nWidth, nHeight, nBgraPitch/sizeof(uchar4), x_min, y_min, x_max, y_max);
}



