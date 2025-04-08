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

	/*poca::core::ImageInterface* connectedComponnetsLabeling3dUI8(const uint8_t* _pixels, const size_t _nbValues, const uint8_t _thresholdMin, const uint8_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
	poca::core::ImageInterface* connectedComponnetsLabeling3dUI16(const uint16_t* _pixels, const size_t _nbValues, const uint16_t _thresholdMin, const uint16_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
	poca::core::ImageInterface* connectedComponnetsLabeling3dUI32(const uint32_t* _pixels, const size_t _nbValues, const uint32_t _thresholdMin, const uint32_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
	poca::core::ImageInterface* connectedComponnetsLabeling3dI32(const int32_t* _pixels, const size_t _nbValues, const int32_t _thresholdMin, const int32_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
	poca::core::ImageInterface* connectedComponnetsLabeling3dF(const float* _pixels, const size_t _nbValues, const float _thresholdMin, const float _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);*/

	template <class T>
	poca::core::ImageInterface* connectedComponnetsLabeling3DGPU(const T* _pixels, const T _thresholdMin, const T _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
#endif

	/*poca::core::ImageInterface* connectedComponnetsLabeling3D(poca::core::ImageInterface* input, const float _thresholdMin, const float _thresholdMax)
	{
#ifndef NO_CUDA
		switch (input->type())
		{
		case poca::core::UINT8:
		{
			std::vector <uint8_t>& pixels = static_cast<poca::core::Image<uint8_t>*>(input)->pixels();
			return connectedComponnetsLabeling3DGPU<uint8_t>(pixels.data(), static_cast<uint8_t>(_thresholdMin), static_cast<uint8_t>(_thresholdMax), input->width(), input->height(), input->depth());
		}
		break;
		case poca::core::UINT16:
		{
			std::vector <uint16_t>& pixels = static_cast<poca::core::Image<uint16_t>*>(input)->pixels();
			return connectedComponnetsLabeling3DGPU<uint16_t>(pixels.data(), static_cast<uint16_t>(_thresholdMin), static_cast<uint16_t>(_thresholdMax), input->width(), input->height(), input->depth());
		}
		break;
		case poca::core::INT32:
		{
			std::vector <int32_t>& pixels = static_cast<poca::core::Image<int32_t>*>(input)->pixels();
			return connectedComponnetsLabeling3DGPU<int32_t>(pixels.data(), static_cast<int32_t>(_thresholdMin), static_cast<int32_t>(_thresholdMax), input->width(), input->height(), input->depth());
		}
		break;
		case poca::core::FLOAT:
		{
			std::vector <float>& pixels = static_cast<poca::core::Image<float>*>(input)->pixels();
			return connectedComponnetsLabeling3DGPU<float>(pixels.data(), static_cast<float>(_thresholdMin), static_cast<float>(_thresholdMax), input->width(), input->height(), input->depth());
		}
		break;
		default:
			return NULL;
			break;
		}
		return NULL;
#else
		std::cout << "GPU is not available. No connected component implemented on CPU" << std::endl;
		return NULL;
#endif
	}*/

	/*template <class T>
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
				return connectedComponnetsLabeling3DGPU<uint8_t>(data, _pixels.size(), _thresholdMin, _thresholdMax, _w, _h, _d);
			}
			else if (type == "uint16_t" || type == "unsigned short") {
				const uint16_t* data = (const uint16_t*)_pixels.data();
				return connectedComponnetsLabeling3DGPU<uint16_t>(data, _pixels.size(), _thresholdMin, _thresholdMax, _w, _h, _d);
			}
			else if (type == "uint32_t" || type == "unsigned int") {
				const uint32_t* data = (const uint32_t*)_pixels.data();
				return connectedComponnetsLabeling3DGPU<uint32_t>(data, _pixels.size(), _thresholdMin, _thresholdMax, _w, _h, _d);
			}
			else if (type == "int32_t" || type == "int") {
				const int32_t* data = (const int32_t*)_pixels.data();
				return connectedComponnetsLabeling3DGPU<int32_t>(data, _pixels.size(), _thresholdMin, _thresholdMax, _w, _h, _d);
			}
			else if (type == "float") {
				const float* data = (const float*)_pixels.data();
				return connectedComponnetsLabeling3DGPU<float>(data, _pixels.size(), _thresholdMin, _thresholdMax, _w, _h, _d);
			}
			else {
				std::cout << "GPU is not available. No connected component implemented on CPU" << std::endl;
				return NULL;
			}
		}
#else
		std::cout << "GPU is not available. No connected component implemented on CPU" << std::endl;
		return NULL;
#endif
	}*/
#endif