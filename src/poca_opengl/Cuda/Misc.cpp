/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Misc.cpp
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

#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#ifndef NO_CUDA
#include <cuda_runtime.h>
#endif

#include "Misc.h"

void sortArrayWRTKeys_CPU(std::vector <float>& _keys, std::vector <uint32_t>& _values)
{
	_values.resize(_keys.size());
	std::iota(std::begin(_values), std::end(_values), 0);

	//Sort wrt the distance to the camera position
	std::sort(_values.begin(), _values.end(),
		[&](int A, int B) -> bool {
			return _keys[A] < _keys[B];
		});
}

void sortArrayWRTKeys(std::vector <float>& _keys, std::vector <uint32_t>& _values) {
#ifndef NO_CUDA
	int devCount; // Number of CUDA devices
	cudaError_t err = cudaGetDeviceCount(&devCount);
	if (err != cudaSuccess) 
		sortArrayWRTKeys_CPU(_keys, _values);
	else 
		sortArrayWRTKeys_GPU(_keys, _values);
#else
	sortArrayWRTKeys_CPU(_keys, _values);
#endif
}

namespace poca::utils {
	void sortWrtCameraPosition(const glm::vec3& _cameraPosition, const glm::vec3& _cameraForwardVec, const std::vector <float>& _xs, const std::vector <float>& _ys, const std::vector <float>& _zs, std::vector <uint32_t>& _indices)
	{
		try {
			_indices.resize(_xs.size());
			std::iota(std::begin(_indices), std::end(_indices), 0);

			//Compute a vector of distances of the points to the camera position
			std::vector <float> distances(_xs.size());

#pragma omp parallel for
			for (int n = 0; n < _xs.size(); n++)
				distances[n] = glm::dot(glm::vec3(_xs[n], _ys[n], _zs[n]) - _cameraPosition, _cameraForwardVec);

			sortArrayWRTKeys(distances, _indices);
		}
		catch (std::runtime_error const& e) {
			std::string mess("Error: sorting localizations with respect to the camera position failed with error message: " + std::string(e.what()));
			std::cout << mess << std::endl;
		}
	}
}