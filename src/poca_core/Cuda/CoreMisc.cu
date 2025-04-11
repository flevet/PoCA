/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      CoreMisc.cu
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

#include "CoreMisc.h"

#ifndef NO_CUDA
#define cuda_check(x) if (x!=cudaSuccess) exit(1);
#define IF_VERBOSE(x) //x

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

template <class T> struct GPUBuffer {
    void init(T* data) {
        IF_VERBOSE(std::cerr << "GPU: " << size * sizeof(T) / 1048576 << " Mb used" << std::endl);
        cpu_data = data;
        cuda_check(cudaMalloc((void**)&gpu_data, size * sizeof(T)));
        cpu2gpu();
    }
    void init(std::vector<T>& data) {
        size = data.size();
        IF_VERBOSE(std::cerr << "GPU: " << size * sizeof(T) / 1048576 << " Mb used" << std::endl);
        cpu_data = data.data();
        cuda_check(cudaMalloc((void**)&gpu_data, size * sizeof(T)));
        cpu2gpu();
    }
    GPUBuffer() {}
    GPUBuffer(std::vector<T>& v) { size = v.size(); init(v.data()); }
    GPUBuffer(T* _v, int _size) { size = _size; init(_v); }
    ~GPUBuffer() { cuda_check(cudaFree(gpu_data)); }

    void cpu2gpu() { cuda_check(cudaMemcpy(gpu_data, cpu_data, size * sizeof(T), cudaMemcpyHostToDevice)); }
    void gpu2cpu() { cuda_check(cudaMemcpy(cpu_data, gpu_data, size * sizeof(T), cudaMemcpyDeviceToHost)); }

    T* cpu_data;
    T* gpu_data;
    int size;
};

template <class T> struct JustGPUBuffer {
    void init(const T* data) {
        IF_VERBOSE(std::cerr << "GPU: " << size * sizeof(T) / 1048576 << " Mb used" << std::endl);
        cuda_check(cudaMalloc((void**)&gpu_data, size * sizeof(T)));
        cuda_check(cudaMemcpy(gpu_data, data, size * sizeof(T), cudaMemcpyHostToDevice));
    }
    JustGPUBuffer(const T* data, const size_t _size) { size = _size; init(data); }
    JustGPUBuffer(const std::vector<T>& v) { size = v.size(); init(v.data()); }
    ~JustGPUBuffer() { cuda_check(cudaFree(gpu_data)); }

    void cpu2gpu() { cuda_check(cudaMemcpy(gpu_data, cpu_data, size * sizeof(T), cudaMemcpyHostToDevice)); }

    T* gpu_data;
    int size;
};

__global__ void kernel_getHist(float* array, std::size_t size, float* histo, std::size_t buckets, float minV, float step)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= size)   return;

    float value = array[tid];

    if (-0.01 <= value && value <= 0.01) return;

    int bin = (int)floor((value - minV) / step);

    if (bin >= buckets)   return;

    atomicAdd(&histo[bin], 1);
}

template <class T>
__global__ void GPUTypeToFloat(const T* src, float* dest, std::size_t size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= size)   return;

    dest[tid] = (float)src[tid];
}

void computeHistogram_GPUU8(const uint8_t* values, const size_t nbValues, std::vector<float>& histo, const float _min, const float _max)
{
    std::vector <float> convertBuffer;
    GPUBuffer<float> out_array;
    convertBuffer.resize(nbValues);
    out_array.init(convertBuffer);
    dim3 block(32);
    dim3 grid((nbValues + block.x - 1) / block.x);
    JustGPUBuffer<uint8_t> inputToConvert(values, nbValues);
    GPUTypeToFloat << <grid, block >> > (inputToConvert.gpu_data, out_array.gpu_data, nbValues);

    GPUBuffer<float> out_histo(histo);

    std::size_t nbBins = histo.size();

    float stepX = (_max - _min) / (float)(nbBins - 1);

    kernel_getHist << <grid, block >> > (out_array.gpu_data, nbValues, out_histo.gpu_data, nbBins, _min, stepX);

    out_histo.gpu2cpu();
}

void computeHistogram_GPUU16(const uint16_t* values, const size_t nbValues, std::vector<float>& histo, const float _min, const float _max)
{
    std::vector <float> convertBuffer;
    GPUBuffer<float> out_array;
    convertBuffer.resize(nbValues);
    out_array.init(convertBuffer);
    dim3 block(32);
    dim3 grid((nbValues + block.x - 1) / block.x);
    JustGPUBuffer<uint16_t> inputToConvert(values, nbValues);
    GPUTypeToFloat << <grid, block >> > (inputToConvert.gpu_data, out_array.gpu_data, nbValues);

    GPUBuffer<float> out_histo(histo);

    std::size_t nbBins = histo.size();

    float stepX = (_max - _min) / (float)(nbBins - 1);

    kernel_getHist << <grid, block >> > (out_array.gpu_data, nbValues, out_histo.gpu_data, nbBins, _min, stepX);

    out_histo.gpu2cpu();
}

void computeHistogram_GPUU32(const uint32_t* values, const size_t nbValues, std::vector<float>& histo, const float _min, const float _max)
{
    std::vector <float> convertBuffer;
    GPUBuffer<float> out_array;
    convertBuffer.resize(nbValues);
    out_array.init(convertBuffer);
    dim3 block(32);
    dim3 grid((nbValues + block.x - 1) / block.x);
    JustGPUBuffer<uint32_t> inputToConvert(values, nbValues);
    GPUTypeToFloat << <grid, block >> > (inputToConvert.gpu_data, out_array.gpu_data, nbValues);

    GPUBuffer<float> out_histo(histo);

    std::size_t nbBins = histo.size();

    float stepX = (_max - _min) / (float)(nbBins - 1);

    kernel_getHist << <grid, block >> > (out_array.gpu_data, nbValues, out_histo.gpu_data, nbBins, _min, stepX);

    out_histo.gpu2cpu();
}

void computeHistogram_GPUI32(const int32_t* values, const size_t nbValues, std::vector<float>& histo, const float _min, const float _max)
{
    std::vector <float> convertBuffer;
    GPUBuffer<float> out_array;
    convertBuffer.resize(nbValues);
    out_array.init(convertBuffer);
    dim3 block(32);
    dim3 grid((nbValues + block.x - 1) / block.x);
    JustGPUBuffer<int32_t> inputToConvert(values, nbValues);
    GPUTypeToFloat << <grid, block >> > (inputToConvert.gpu_data, out_array.gpu_data, nbValues);

    GPUBuffer<float> out_histo(histo);

    std::size_t nbBins = histo.size();

    float stepX = (_max - _min) / (float)(nbBins - 1);

    kernel_getHist << <grid, block >> > (out_array.gpu_data, nbValues, out_histo.gpu_data, nbBins, _min, stepX);

    out_histo.gpu2cpu();
}

void computeHistogram_GPUF(const float* values, const size_t nbValues, std::vector<float>& histo, const float _min, const float _max)
{
    JustGPUBuffer<float> out_array(values, nbValues);
    GPUBuffer<float> out_histo(histo);

    std::size_t nbBins = histo.size();

    float stepX = (_max - _min) / (float)(nbBins - 1);

    dim3 block(32);
    dim3 grid((nbValues + block.x - 1) / block.x);

    kernel_getHist <<<grid, block>>> (out_array.gpu_data, nbValues, out_histo.gpu_data, nbBins, _min, stepX);

    out_histo.gpu2cpu();
}

template <typename Iterator>
void print_range(const std::string& name, Iterator first, Iterator last)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    std::cout << name << ": ";
    thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));
    std::cout << "\n";
}

void computeStats_GPUU8(const uint8_t* values, const size_t nbValues, std::vector<float>& stats)
{
    std::vector <float> convertBuffer(nbValues);
    GPUBuffer<float> out_array(convertBuffer);
    dim3 block(32);
    dim3 grid((nbValues + block.x - 1) / block.x);
    JustGPUBuffer<uint8_t> inputToConvert(values, nbValues);
    GPUTypeToFloat << <grid, block >> > (inputToConvert.gpu_data, out_array.gpu_data, nbValues);

    GPUBuffer<float> out_stats(stats);

    // setup arguments
    summary_stats_unary_op<float> unary_op;
    summary_stats_binary_op<float> binary_op;
    summary_stats_data<float> init;

    init.initialize();

    // compute summary statistics
    summary_stats_data<float> result = thrust::transform_reduce(thrust::device_pointer_cast(out_array.gpu_data), thrust::device_pointer_cast(out_array.gpu_data) + nbValues, unary_op, init, binary_op);
    thrust::sort(thrust::device_pointer_cast(out_array.gpu_data), thrust::device_pointer_cast(out_array.gpu_data) + nbValues);

    out_array.gpu2cpu();

    stats[0] = result.mean;
    stats[1] = out_array.cpu_data[int(out_array.size / 2)];
    stats[2] = std::sqrt(result.variance_n());
    stats[3] = result.min;
    stats[4] = result.max;
}

void computeStats_GPUU16(const uint16_t* values, const size_t nbValues, std::vector<float>& stats)
{
    std::vector <float> convertBuffer(nbValues);
    GPUBuffer<float> out_array(convertBuffer);
    dim3 block(32);
    dim3 grid((nbValues + block.x - 1) / block.x);
    JustGPUBuffer<uint16_t> inputToConvert(values, nbValues);
    GPUTypeToFloat << <grid, block >> > (inputToConvert.gpu_data, out_array.gpu_data, nbValues);

    GPUBuffer<float> out_stats(stats);

    // setup arguments
    summary_stats_unary_op<float> unary_op;
    summary_stats_binary_op<float> binary_op;
    summary_stats_data<float> init;

    init.initialize();

    // compute summary statistics
    summary_stats_data<float> result = thrust::transform_reduce(thrust::device_pointer_cast(out_array.gpu_data), thrust::device_pointer_cast(out_array.gpu_data) + nbValues, unary_op, init, binary_op);
    thrust::sort(thrust::device_pointer_cast(out_array.gpu_data), thrust::device_pointer_cast(out_array.gpu_data) + nbValues);

    out_array.gpu2cpu();

    stats[0] = result.mean;
    stats[1] = out_array.cpu_data[int(out_array.size / 2)];
    stats[2] = std::sqrt(result.variance_n());
    stats[3] = result.min;
    stats[4] = result.max;
}

void computeStats_GPUI32(const int32_t* values, const size_t nbValues, std::vector<float>& stats)
{
    std::vector <float> convertBuffer(nbValues);
    GPUBuffer<float> out_array(convertBuffer);
    dim3 block(32);
    dim3 grid((nbValues + block.x - 1) / block.x);
    JustGPUBuffer<int32_t> inputToConvert(values, nbValues);
    GPUTypeToFloat << <grid, block >> > (inputToConvert.gpu_data, out_array.gpu_data, nbValues);

    GPUBuffer<float> out_stats(stats);

    // setup arguments
    summary_stats_unary_op<float> unary_op;
    summary_stats_binary_op<float> binary_op;
    summary_stats_data<float> init;

    init.initialize();

    // compute summary statistics
    summary_stats_data<float> result = thrust::transform_reduce(thrust::device_pointer_cast(out_array.gpu_data), thrust::device_pointer_cast(out_array.gpu_data) + nbValues, unary_op, init, binary_op);
    thrust::sort(thrust::device_pointer_cast(out_array.gpu_data), thrust::device_pointer_cast(out_array.gpu_data) + nbValues);

    out_array.gpu2cpu();

    stats[0] = result.mean;
    stats[1] = out_array.cpu_data[int(out_array.size / 2)];
    stats[2] = std::sqrt(result.variance_n());
    stats[3] = result.min;
    stats[4] = result.max;
}

void computeStats_GPUU32(const uint32_t* values, const size_t nbValues, std::vector<float>& stats)
{
    std::vector <float> convertBuffer(nbValues);
    GPUBuffer<float> out_array(convertBuffer);
    dim3 block(32);
    dim3 grid((nbValues + block.x - 1) / block.x);
    JustGPUBuffer<uint32_t> inputToConvert(values, nbValues);
    GPUTypeToFloat << <grid, block >> > (inputToConvert.gpu_data, out_array.gpu_data, nbValues);

    GPUBuffer<float> out_stats(stats);

    // setup arguments
    summary_stats_unary_op<float> unary_op;
    summary_stats_binary_op<float> binary_op;
    summary_stats_data<float> init;

    init.initialize();

    // compute summary statistics
    summary_stats_data<float> result = thrust::transform_reduce(thrust::device_pointer_cast(out_array.gpu_data), thrust::device_pointer_cast(out_array.gpu_data) + nbValues, unary_op, init, binary_op);
    thrust::sort(thrust::device_pointer_cast(out_array.gpu_data), thrust::device_pointer_cast(out_array.gpu_data) + nbValues);

    out_array.gpu2cpu();

    stats[0] = result.mean;
    stats[1] = out_array.cpu_data[int(out_array.size / 2)];
    stats[2] = std::sqrt(result.variance_n());
    stats[3] = result.min;
    stats[4] = result.max;
}

void computeStats_GPUF(const float* values, const size_t nbValues, std::vector<float>& stats)
{
    std::vector <float> copied(values, values + nbValues);
    GPUBuffer<float> out_array(copied);
    GPUBuffer<float> out_stats(stats);

    // setup arguments
    summary_stats_unary_op<float> unary_op;
    summary_stats_binary_op<float> binary_op;
    summary_stats_data<float> init;

    init.initialize();

    // compute summary statistics
    summary_stats_data<float> result = thrust::transform_reduce(thrust::device_pointer_cast(out_array.gpu_data), thrust::device_pointer_cast(out_array.gpu_data) + nbValues, unary_op, init, binary_op);
    thrust::sort(thrust::device_pointer_cast(out_array.gpu_data), thrust::device_pointer_cast(out_array.gpu_data) + nbValues);

    out_array.gpu2cpu();

    stats[0] = result.mean;
    stats[1] = out_array.cpu_data[int(out_array.size / 2)];
    stats[2] = std::sqrt(result.variance_n());
    stats[3] = result.min;
    stats[4] = result.max;
}


#endif

