/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      CoreMisc.h
*
* Copyright: Florian Levet (2020-2022)
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

#ifndef H_MISC_CORE_H
#define H_MISC_CORE_H

#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <iostream>
#ifndef NO_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#ifndef NO_CUDA
	void computeHistogram_GPUU8(const uint8_t* values, const size_t nbValues, std::vector<float>& bins, const float _min, const float _max);
	void computeHistogram_GPUU16(const uint16_t* values, const size_t nbValues, std::vector<float>& bins, const float _min, const float _max);
	void computeHistogram_GPUU32(const uint32_t* values, const size_t nbValues, std::vector<float>& bins, const float _min, const float _max);
	void computeHistogram_GPUI32(const int32_t* values, const size_t nbValues, std::vector<float>& bins, const float _min, const float _max);
	void computeHistogram_GPUF(const float* values, const size_t nbValues, std::vector<float>& bins, const float _min, const float _max);
	
	void computeStats_GPUU8(const uint8_t* values, const size_t nbValues, std::vector<float>& stats);
	void computeStats_GPUU16(const uint16_t* values, const size_t nbValues, std::vector<float>& stats);
	void computeStats_GPUU32(const uint32_t* values, const size_t nbValues, std::vector<float>& stats);
	void computeStats_GPUI32(const int32_t* values, const size_t nbValues, std::vector<float>& stats);
	void computeStats_GPUF(const float* values, const size_t nbValues, std::vector<float>& stats);
#endif

	template <class T>
	void computeHistogram_CPU(const std::vector<T>& values, std::vector<float>& histo, const float _min, const float _max)
	{
		std::size_t nbValues = values.size(), nbBins = histo.size();
		float stepX = (_max - _min) / (float)(nbBins - 1);
		for (unsigned int i = 0; i < nbValues; i++) {
			if (values[i] == -1) continue;
			unsigned short index = (unsigned short)floor((values[i] - _min) / stepX);
			if (index < nbBins)
				histo[index]++;
		}
	}

	template <class T>
	void computeHistogram(const std::vector<T>& values, std::vector<float>& histo, const float _min, const float _max) {
#ifndef NO_CUDA
		int devCount; // Number of CUDA devices
		cudaError_t err = cudaGetDeviceCount(&devCount);
		if (err != cudaSuccess) {
			std::cout << "CPU" << std::endl;
			computeHistogram_CPU(values, histo, _min, _max);
		}
		else {
			std::cout << "GPU" << std::endl;
			std::string type = typeid(T).name();
			if (type == "uint8_t" || type == "unsigned char") {
				const uint8_t* data = (const uint8_t*)values.data();
				computeHistogram_GPUU8(data, values.size(), histo, _min, _max);
			}
			else if (type == "uint16_t" || type == "unsigned short") {
				const uint16_t* data = (const uint16_t*)values.data();
				computeHistogram_GPUU16(data, values.size(), histo, _min, _max);
			}
			else if (type == "uint32_t" || type == "unsigned int") {
				const uint32_t* data = (const uint32_t*)values.data();
				computeHistogram_GPUU32(data, values.size(), histo, _min, _max);
			}
			else if (type == "int32_t" || type == "int") {
				const int32_t* data = (const int32_t*)values.data();
				computeHistogram_GPUI32(data, values.size(), histo, _min, _max);
			}
			else if (type == "float") {
				const float* data = (const float*)values.data();
				computeHistogram_GPUF(data, values.size(), histo, _min, _max);
			}
			else {
				std::cout << "This is a problem" << std::endl;
				computeHistogram_CPU(values, histo, _min, _max);
			}
		}
#else
		computeHistogram_CPU(values, histo, _min, _max);
#endif
	}

	template <class T>
	void computeStats_CPU(const std::vector<T>& values, std::vector<float>& stats)
	{
		/*_values.resize(_keys.size());
		std::iota(std::begin(_values), std::end(_values), 0);

		//Sort wrt the distance to the camera position
		std::sort(_values.begin(), _values.end(),
			[&](int A, int B) -> bool {
				return _keys[A] < _keys[B];
			});*/
	}
	
	template <class T>
	void computeStats(const std::vector<T>& values, std::vector<float>& stats) {
#ifndef NO_CUDA
		int devCount; // Number of CUDA devices
		cudaError_t err = cudaGetDeviceCount(&devCount);
		if (err != cudaSuccess) {
			std::cout << "CPU" << std::endl;
			computeStats_CPU(values, stats);
		}
		else {
			std::cout << "GPU" << std::endl;
			std::string type = typeid(T).name();
			if (type == "uint8_t" || type == "unsigned char") {
				const uint8_t* data = (const uint8_t*)values.data();
				computeStats_GPUU8(data, values.size(), stats);
			}
			else if (type == "uint16_t" || type == "unsigned short") {
				const uint16_t* data = (const uint16_t*)values.data();
				computeStats_GPUU16(data, values.size(), stats);
			}
			else if (type == "uint32_t" || type == "unsigned int") {
				const uint32_t* data = (const uint32_t*)values.data();
				computeStats_GPUU32(data, values.size(), stats);
			}
			else if (type == "int32_t" || type == "int") {
				const int32_t* data = (const int32_t*)values.data();
				computeStats_GPUI32(data, values.size(), stats);
			}
			else if (type == "float") {
				const float* data = (const float*)values.data();
				computeStats_GPUF(data, values.size(), stats);
			}
			else {
				std::cout << "This is a problem" << std::endl;
				computeStats_CPU(values, stats);
			}
		}
#else
		computeStats_CPU(values, stats);
#endif
	}



#endif