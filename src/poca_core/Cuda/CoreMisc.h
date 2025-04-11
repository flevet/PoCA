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
#include <thrust\pair.h>
#include <thrust\device_vector.h>
#include <thrust\extrema.h>
#include <thrust\sort.h>
#endif

#ifndef NO_CUDA
// This example computes several statistical properties of a data
// series in a single reduction.  The algorithm is described in detail here:
// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
//
// Thanks to Joseph Rhoads for contributing this example


// structure used to accumulate the moments and other 
// statistical properties encountered so far.
template <typename T>
struct summary_stats_data
{
	T n;
	T min;
	T max;
	T mean;
	T M2;
	T M3;
	T M4;

	// initialize to the identity element
	void initialize()
	{
		n = mean = M2 = M3 = M4 = 0;
		min = std::numeric_limits<T>::max();
		max = std::numeric_limits<T>::min();
	}

	T variance() { return M2 / (n - 1); }
	T variance_n() { return M2 / n; }
	T skewness() { return std::sqrt(n) * M3 / std::pow(M2, (T)1.5); }
	T kurtosis() { return n * M4 / (M2 * M2); }
};

// stats_unary_op is a functor that takes in a value x and
// returns a variace_data whose mean value is initialized to x.
template <typename T>
struct summary_stats_unary_op
{
	__host__ __device__
		summary_stats_data<T> operator()(const T& x) const
	{
		summary_stats_data<T> result;
		result.n = 1;
		result.min = x;
		result.max = x;
		result.mean = x;
		result.M2 = 0;
		result.M3 = 0;
		result.M4 = 0;

		return result;
	}
};

// summary_stats_binary_op is a functor that accepts two summary_stats_data 
// structs and returns a new summary_stats_data which are an
// approximation to the summary_stats for 
// all values that have been agregated so far
template <typename T>
struct summary_stats_binary_op
	: public thrust::binary_function<const summary_stats_data<T>&,
	const summary_stats_data<T>&,
	summary_stats_data<T> >
{
	__host__ __device__
		summary_stats_data<T> operator()(const summary_stats_data<T>& x, const summary_stats_data <T>& y) const
	{
		summary_stats_data<T> result;

		// precompute some common subexpressions
		T n = x.n + y.n;
		T n2 = n * n;
		T n3 = n2 * n;

		T delta = y.mean - x.mean;
		T delta2 = delta * delta;
		T delta3 = delta2 * delta;
		T delta4 = delta3 * delta;

		//Basic number of samples (n), min, and max
		result.n = n;
		result.min = thrust::min(x.min, y.min);
		result.max = thrust::max(x.max, y.max);

		result.mean = x.mean + delta * y.n / n;

		result.M2 = x.M2 + y.M2;
		result.M2 += delta2 * x.n * y.n / n;

		result.M3 = x.M3 + y.M3;
		result.M3 += delta3 * x.n * y.n * (x.n - y.n) / n2;
		result.M3 += (T)3.0 * delta * (x.n * y.M2 - y.n * x.M2) / n;

		result.M4 = x.M4 + y.M4;
		result.M4 += delta4 * x.n * y.n * (x.n * x.n - x.n * y.n + y.n * y.n) / n3;
		result.M4 += (T)6.0 * delta2 * (x.n * x.n * y.M2 + y.n * y.n * x.M2) / n2;
		result.M4 += (T)4.0 * delta * (x.n * y.M3 - y.n * x.M3) / n;

		return result;
	}
};

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
			//std::cout << "GPU" << std::endl;
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
			//std::cout << "GPU" << std::endl;
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