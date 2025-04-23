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
