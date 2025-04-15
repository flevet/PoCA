/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      CoreMisc.h
*
* Copyright: Florian Levet (2020-2022)
*			 Modified from https://github.com/zsef123/Connected_components_PyTorch/tree/main
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*
* PoCA is a free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 3 of the License, or (at your option) any later version.
*
* The algorithms that underlie PoCA have required considerable
* development. They are described in the original SR-Tesseler paper,
* doi:10.1038/nmeth.3579. If you use PoCA as part of work (visualization,
* manipulation, quantification) towards a scientific publication, please include
* a citation to the original paper.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program; if not, write to the Free Software Foundation,
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#ifndef H_CONNECTED_COMPONENTS_H
#define H_CONNECTED_COMPONENTS_H

#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <iostream>
#ifndef NO_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust\device_vector.h>
#include <thrust/transform.h>
#endif

#include <General/Image.hpp>

#ifndef NO_CUDA
void connectedComponnets2DLabelingBinary(uint8_t* const _pixels, const uint32_t W, const uint32_t H, uint32_t* _labels);
void connectedComponnets3DLabelingBinary(uint8_t* const _pixels, const size_t _nbValues, const uint32_t W, const uint32_t H, const uint32_t D, uint32_t* _labels);
template <class T>
poca::core::ImageInterface* connectedComponnetsLabelingGPU(const T* _pixels, const T _thresholdMin, const T _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
#endif
#endif