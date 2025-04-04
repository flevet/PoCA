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
#include <thrust/iterator/discard_iterator.h>
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

struct dkeygen : public thrust::unary_function<int, int>
{
    int dim;
    int numd;

    dkeygen(const int _dim, const int _numd) : dim(_dim), numd(_numd) {};

    __host__ __device__ int operator()(const int val) const {
        return (val / dim);
    }
};

typedef thrust::tuple<float, float> mytuple;
struct my_dist : public thrust::unary_function<mytuple, float>
{
    __host__ __device__ float operator()(const mytuple& my_tuple) const {
        float temp = thrust::get<0>(my_tuple) - thrust::get<1>(my_tuple);
        return temp * temp;
    }
};


struct d_idx : public thrust::unary_function<int, int>
{
    int dim;
    int numd;

    d_idx(int _dim, int _numd) : dim(_dim), numd(_numd) {};

    __host__ __device__ int operator()(const int val) const {
        return (val % (dim * numd));
    }
};

struct c_idx : public thrust::unary_function<int, int>
{
    int dim;
    int numd;

    c_idx(int _dim, int _numd) : dim(_dim), numd(_numd) {};

    __host__ __device__ int operator()(const int val) const {
        return (val % dim) + (dim * (val / (dim * numd)));
    }
};

struct my_sqrt : public thrust::unary_function<float, float>
{
    __host__ __device__ float operator()(const float val) const {
        return sqrtf(val);
    }
};

void sortArrayWRTPoint_GPU(const float* _xs, const float* _ys, const float* _zs, const size_t num_data, const poca::core::Vec3mf& _point, std::vector <uint32_t>& _values)
{
    thrust::device_vector<float> arr1(_xs, _xs + num_data), arr2(_ys, _ys + num_data), arr3(_zs, _zs + num_data);
    //First, interleaved the xs, ys, zs together in one vector
    //https://stackoverflow.com/questions/76865810/replace-merge-operations-in-vectors-using-cuda-thrust
    const auto xs_ptr = arr1.data();
    const auto ys_ptr = arr2.data();
    const auto zs_ptr = arr3.data();
    const auto batch_interleave_it = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [xs_ptr, ys_ptr, zs_ptr]
        __host__ __device__(int idx) -> float {
        int in_idx = idx / 3;
        int res = idx % 3;
        if (res == 0)
            return xs_ptr[in_idx];
        else if(res == 1)
            return ys_ptr[in_idx];
        else
            return zs_ptr[in_idx];
    });

    //Second, compute the distance to the point
    //https://stackoverflow.com/questions/27823951/thrust-vector-distance-calculation
    int dim = 3;
    thrust::device_vector<float> d_point(_point.getValues(), _point.getValues() + 3);
    thrust::device_vector<float> d_distances(num_data);
    thrust::reduce_by_key(thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), dkeygen(dim, num_data)),
        thrust::make_transform_iterator(thrust::make_counting_iterator<int>(dim* num_data), dkeygen(dim, num_data)),
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    thrust::make_permutation_iterator(d_point.begin(), thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), c_idx(dim, num_data))),
                    thrust::make_permutation_iterator(batch_interleave_it, thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), d_idx(dim, num_data)))
                )
            ),
            my_dist()),
        thrust::make_discard_iterator(), d_distances.begin());

    //Third, sort the vector to the point wrt to the distances
    thrust::device_vector<uint32_t> d_values(_values);
    thrust::sort_by_key(d_distances.begin(), d_distances.end(), d_values.begin());

    //Copy values back to host
    thrust::copy(d_values.begin(), d_values.end(), _values.begin());
}
#endif

