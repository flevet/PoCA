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
// this functor converts values of T to M
// in the actual scenario this method does perform much more useful operations
template <class T, class M>
struct Functor : public thrust::unary_function<T, M> {
	Functor() {}

	__host__ __device__ M operator() (const T& val) const {
		return M(val);
	}
};

template <class T, class M>
poca::core::ImageInterface* convertAndCreateLabelImage(thrust::device_vector<T>& d_labels, const uint32_t _w, const uint32_t _h, const uint32_t _d)
{
	poca::core::Image<M>* image = new poca::core::Image<M>(poca::core::LABEL);
	if (typeid(T).name() == "unsigned int") {
		//No need to convert label image
		std::vector <M>& labels = image->pixels();
		labels.resize(d_labels.size());
		cudaMemcpy(labels.data(), thrust::raw_pointer_cast(d_labels.data()), labels.size() * sizeof(M), cudaMemcpyDeviceToHost);
	}
	else {
		thrust::device_vector<M> d_converted(d_labels.size());
		thrust::transform(d_labels.begin(), d_labels.end(), d_converted.begin(), Functor<T, M>());
		std::vector <M>& labels = image->pixels();
		labels.resize(d_labels.size());
		cudaMemcpy(labels.data(), thrust::raw_pointer_cast(d_converted.data()), labels.size() * sizeof(M), cudaMemcpyDeviceToHost);
		d_converted.clear();
		d_converted.shrink_to_fit();
	}
	d_labels.clear();
	d_labels.shrink_to_fit();
	image->finalizeImage(_w, _h, _d);
	return image;
}

	poca::core::ImageInterface* connectedComponnetsLabeling3dUI8(const uint8_t* _pixels, const size_t _nbValues, const uint8_t _thresholdMin, const uint8_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
	poca::core::ImageInterface* connectedComponnetsLabeling3dUI16(const uint16_t* _pixels, const size_t _nbValues, const uint16_t _thresholdMin, const uint16_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
	poca::core::ImageInterface* connectedComponnetsLabeling3dUI32(const uint32_t* _pixels, const size_t _nbValues, const uint32_t _thresholdMin, const uint32_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
	poca::core::ImageInterface* connectedComponnetsLabeling3dI32(const int32_t* _pixels, const size_t _nbValues, const int32_t _thresholdMin, const int32_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
	poca::core::ImageInterface* connectedComponnetsLabeling3dF(const float* _pixels, const size_t _nbValues, const float _thresholdMin, const float _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);

	void relabelI32(std::vector <uint32_t>& _labels, std::vector <uint32_t>& _relabels);
	void computeFeaturesLabelImage(poca::core::ImageInterface*);
	poca::core::ImageInterface* thresholdLabelsFeature(poca::core::ImageInterface*);
#endif

	template <class T>
	poca::core::ImageInterface* connectedComponnetsLabeling3d(const std::vector <T>& _pixels, const T _thresholdMin, const T _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d)
	{
#ifndef NO_CUDA
		int devCount; // Number of CUDA devices
		cudaError_t err = cudaGetDeviceCount(&devCount);
		if (err != cudaSuccess) {
			std::cout << "GPU is not available. No connected component implemented on CPU" << std::endl;
			return NULL;
		}
		else {
			std::cout << "GPU" << std::endl;
			std::string type = typeid(T).name();
			if (type == "uint8_t" || type == "unsigned char") {
				const uint8_t* data = (const uint8_t*)_pixels.data();
				return connectedComponnetsLabeling3dUI8(data, _pixels.size(), _thresholdMin, _thresholdMax, _w, _h, _d);
			}
			else if (type == "uint16_t" || type == "unsigned short") {
				const uint16_t* data = (const uint16_t*)_pixels.data();
				return connectedComponnetsLabeling3dUI16(data, _pixels.size(), _thresholdMin, _thresholdMax, _w, _h, _d);
			}
			else if (type == "uint32_t" || type == "unsigned int") {
				const uint32_t* data = (const uint32_t*)_pixels.data();
				return connectedComponnetsLabeling3dUI32(data, _pixels.size(), _thresholdMin, _thresholdMax, _w, _h, _d);
			}
			else if (type == "int32_t" || type == "int") {
				const int32_t* data = (const int32_t*)_pixels.data();
				return connectedComponnetsLabeling3dI32(data, _pixels.size(), _thresholdMin, _thresholdMax, _w, _h, _d);
			}
			else if (type == "float") {
				const float* data = (const float*)_pixels.data();
				return connectedComponnetsLabeling3dF(data, _pixels.size(), _thresholdMin, _thresholdMax, _w, _h, _d);
			}
			else {
				std::cout << "GPU is not available. No connected component implemented on CPU" << std::endl;
				return NULL;
			}
		}
#else
		computeStats_CPU(values, stats);
#endif
	}
#endif