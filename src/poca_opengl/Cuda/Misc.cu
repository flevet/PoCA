/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Misc.cu
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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>
#include <numeric>

#include "Misc.h"

#include <stdbool.h>

static
void
cuda_assert(const cudaError_t code, const char* const file, const int line, const bool abort)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "cuda_assert: %s %s %d\n", cudaGetErrorString(code), file, line);

        if (abort)
        {
            cudaDeviceReset();
            exit(code);
        }
    }
}

#define cuda(...) { cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__, true); }

#ifndef NO_CUDA
void sortArrayWRTKeys_GPU(std::vector <float>& _keys, std::vector <uint32_t>& _values)
{
    thrust::device_vector<float> d_keys(_keys);
    thrust::device_vector<uint32_t> d_values(_values);

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

    thrust::copy(d_values.begin(), d_values.end(), _values.begin());

}
#endif

