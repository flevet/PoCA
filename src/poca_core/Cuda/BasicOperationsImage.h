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
void count_occurences_label_kernel_gpu(thrust::device_vector<T>& d_pixels, thrust::device_vector<T>& d_labels, thrust::device_vector<T>& d_counts);
template <class T>
void remove_small_labels_kernel_gpu(thrust::device_vector<T>& d_pixels, T threshold);
template <class T>
__global__ void kernel_threshold(const T* image, const T _thresholdMin, const T _thresholdMax, uint8_t* thresholdedImage, uint32_t size);

template <class T>
__global__ void binarize(const T* image, uint8_t* bimage, uint32_t size);

template <class T>
float getAutoThreshold(const T* image, const uint32_t _w, const uint32_t _h, const uint32_t _d);

void relabelI32(std::vector <uint32_t>& _labels, std::vector <uint32_t>& _relabels);
void computeFeaturesLabelImage(poca::core::ImageInterface*);
poca::core::ImageInterface* thresholdLabelsFeature(poca::core::ImageInterface*);
uint32_t relabel_kernel_uint32t_gpu(thrust::device_vector<uint32_t>& d_labels);

template <class T>
void identify_holes(const thrust::device_vector <T>& _pixels, thrust::device_vector <T>& _holes, const uint32_t _width, const uint32_t _height, const uint32_t _depth);
template <class T>
void map_holes_to_labels(const thrust::device_vector <T>& _pixels, thrust::device_vector <T>& _holes, const uint32_t _width, const uint32_t _height, const uint32_t _depth);
template <class T>
void fill_holes_gpu(thrust::device_vector<T>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const T _threshold);
template <class T>
void run_fill_holes_2(std::vector<T>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const T _threshold);

template <class T, class M>
void pad(const thrust::device_vector<T>& _source, thrust::device_vector<M>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template <class T, class M>
void unpad(const thrust::device_vector<T>& _source, thrust::device_vector<M>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
#endif

#endif