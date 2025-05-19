/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      BasicOperationsImage.h
*
* Copyright: Florian Levet (2020-2025)
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

#ifndef H_BASIC_OPERATIONS_IMAGE_H
#define H_BASIC_OPERATIONS_IMAGE_H

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
template <class T, class M>
poca::core::ImageInterface* convertAndCreateLabelImage(thrust::device_vector<T>& d_labels, const uint32_t _w, const uint32_t _h, const uint32_t _d);
template <class T>
void relabel_kernel_gpu(thrust::device_vector<T>& d_labels);
template <class T>
__global__ void kernel_threshold(const T* image, const T _thresholdMin, const T _thresholdMax, uint8_t* thresholdedImage, uint32_t size);

template <class T>
float getAutoThreshold(const T* image, const uint32_t _w, const uint32_t _h, const uint32_t _d);

void relabelI32(std::vector <uint32_t>& _labels, std::vector <uint32_t>& _relabels);
void computeFeaturesLabelImage(poca::core::ImageInterface*);
poca::core::ImageInterface* thresholdLabelsFeature(poca::core::ImageInterface*);
#endif

#endif