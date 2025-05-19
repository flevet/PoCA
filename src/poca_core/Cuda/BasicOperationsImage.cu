/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      BasicOperationsImage.cu
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

#include <thrust\pair.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\extrema.h>
#include <thrust\sort.h>
#include <thrust\unique.h>
#include <thrust\sequence.h>
#include <thrust\distance.h>
#include <thrust/binary_search.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "BasicOperationsImage.h"

#ifndef NO_CUDA
#define cuda_check(x) if (x!=cudaSuccess) exit(1);
#define IF_VERBOSE(x) //x
#define IDX3(i, j, k, width, height) (((k) * (height) + (i)) * (width) + (j))
#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8
#define BLOCKDIM_Z 8

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

// this functor converts values of T to M
// in the actual scenario this method does perform much more useful operations
template <class T, class M>
struct Functor : public thrust::unary_function<T, M> {
    Functor() {}

    __host__ __device__ M operator() (const T& val) const {
        return M(val);
    }
};

template <class T>
void relabel_kernel_gpu(thrust::device_vector<T>& d_labels)
{
    thrust::device_vector<T> d_data(d_labels);
    // Sort the data copy
    thrust::sort(thrust::device, d_labels.begin(), d_labels.end());
    // Allocate an array to store unique values
    thrust::device_vector<T> d_unique = d_labels;
    // Compress all duplicates
    const auto end = thrust::unique(d_unique.begin(), d_unique.end());
    // Search for all original labels, in this compressed range, and write their
    // indices back as the result 
    thrust::lower_bound(d_unique.begin(), end, d_data.begin(), d_data.end(), d_labels.begin());
}

template <class T>
void count_occurences_label_kernel_gpu(thrust::device_vector<T>& d_pixels, thrust::device_vector<T>& d_labels, thrust::device_vector<T>& d_counts)
{
    thrust::sort(thrust::device, d_pixels.begin(), d_pixels.end());
    thrust::device_vector<T> d_unique = d_pixels;
    const auto end = thrust::unique(d_unique.begin(), d_unique.end());
    int nbUniqueLabels = thrust::distance(d_unique.begin(), end);
    thrust::device_vector<T> ones(d_pixels.size(), 1);
    d_labels.resize(nbUniqueLabels);
    d_counts.resize(nbUniqueLabels);
    //thrust::copy(d_unique.begin(), end, std::ostream_iterator<T>(std::cout, ","));
    thrust::equal_to<T> binary_pred;
    thrust::plus<T> binary_op;
    auto new_end = thrust::reduce_by_key(thrust::device, d_pixels.begin(), d_pixels.end(), ones.begin(), d_labels.begin(), d_counts.begin(), binary_pred, binary_op);
    //std::cout << std::endl;
    //thrust::copy(d_labels.begin(), new_end.first, std::ostream_iterator<T>(std::cout, ","));
    //std::cout << std::endl;
    //thrust::copy(d_counts.begin(), new_end.second, std::ostream_iterator<T>(std::cout, ","));

}

template <class T>
__global__ void kernel_threshold(const T* image, const T _thresholdMin, const T _thresholdMax, uint8_t* thresholdedImage, uint32_t size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    T value = image[tid];
    thresholdedImage[tid] = value >= _thresholdMin && value <= _thresholdMax ? 255 : 0;
}

template <class T>
__global__ void kernel_threshold32(const T* image, const T _thresholdMin, const T _thresholdMax, int32_t* thresholdedImage, uint32_t size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    T value = image[tid];
    thresholdedImage[tid] = value > _thresholdMin && value < _thresholdMax ? 255 : 0;
}

#define BLOCK_X 8
#define BLOCK_Y 4
#define BLOCK_Z 4

void relabel_kernel(std::vector <int32_t>& _labels, std::vector <int32_t>& _relabels)
{
    thrust::device_vector<int32_t> d_result(_labels);
    relabel_kernel_gpu(d_result);
    cudaMemcpy(_relabels.data(), thrust::raw_pointer_cast(d_result.data()), _labels.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
}

void count_occurences_label_kernel(std::vector <int32_t>& _labels, std::vector <int32_t>& _relabels)
{
    thrust::device_vector<int32_t> d_data(_labels);
    thrust::sort(thrust::device, d_data.begin(), d_data.end());
    thrust::device_vector<int32_t> d_unique = d_data;
    const auto end = thrust::unique(d_unique.begin(), d_unique.end());
    int nbUniqueLabels = thrust::distance(d_unique.begin(), end);
    thrust::device_vector<int32_t> ones(_labels.size(), 1), C(nbUniqueLabels), D(nbUniqueLabels);
    thrust::copy(d_unique.begin(), end, std::ostream_iterator<int32_t>(std::cout, ","));
    thrust::equal_to<int32_t> binary_pred;
    thrust::plus<int32_t> binary_op;
    auto new_end = thrust::reduce_by_key(thrust::device, d_data.begin(), d_data.end(), ones.begin(), C.begin(), D.begin(), binary_pred, binary_op);
    std::cout << std::endl;
    thrust::copy(C.begin(), new_end.first, std::ostream_iterator<int32_t>(std::cout, ","));
    std::cout << std::endl;
    thrust::copy(D.begin(), new_end.second, std::ostream_iterator<int32_t>(std::cout, ","));

}

void relabelI32(std::vector <uint32_t>& _labels, std::vector <uint32_t>& _relabels)
{
    _relabels.resize(_labels.size());
    thrust::device_vector<int32_t> d_result(_labels);
    relabel_kernel_gpu(d_result);
    cudaMemcpy(_relabels.data(), thrust::raw_pointer_cast(d_result.data()), _labels.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "Here" << std::endl;
}

void computeFeaturesLabelImage(poca::core::ImageInterface* _image)
{
    thrust::device_vector<float> d_labels, d_counts;
    switch (_image->type())
    {
    case poca::core::UINT8:
    {
        poca::core::Image<uint8_t>* casted = static_cast <poca::core::Image<uint8_t>*>(_image);
        thrust::device_vector<float> d_pixels(casted->pixels());
        count_occurences_label_kernel_gpu< float>(d_pixels, d_labels, d_counts);
    }
    break;
    case poca::core::UINT16:
    {
        poca::core::Image<uint16_t>* casted = static_cast <poca::core::Image<uint16_t>*>(_image);
        thrust::device_vector<float> d_pixels(casted->pixels());
        count_occurences_label_kernel_gpu<float>(d_pixels, d_labels, d_counts);
    }
    break;
    case poca::core::UINT32:
    {
        poca::core::Image<uint32_t>* casted = static_cast <poca::core::Image<uint32_t>*>(_image);
        thrust::device_vector<float> d_pixels(casted->pixels());
        count_occurences_label_kernel_gpu<float>(d_pixels, d_labels, d_counts);
    }
    break;
    default:
        break;
    }
    std::vector <float>& volume = _image->volumes();
    volume.resize(d_counts.size() - 1);
    std::vector <float> label(d_counts.size() - 1);
    cudaMemcpy(label.data(), thrust::raw_pointer_cast(d_labels.data() + 1), label.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(volume.data(), thrust::raw_pointer_cast(d_counts.data() + 1), volume.size() * sizeof(float), cudaMemcpyDeviceToHost);
    _image->addFeature("label", poca::core::generateDataWithLog(label));
    _image->addFeature("volume", poca::core::generateDataWithLog(volume));
    _image->setCurrentHistogramType("label");
}

template <class T>
__global__ void kernel_threshol_feature_label_gpu(const T* labels, const float* features, float _thresholdMin, const float _thresholdMax, uint32_t minPixel, uint32_t* thresholdedImage, uint32_t size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    T labelIdInFeature = labels[tid] - minPixel;
    float value = features[labelIdInFeature];
    thresholdedImage[tid] = value >= _thresholdMin && value <= _thresholdMax ? labels[tid] : 0;
}

poca::core::ImageInterface* thresholdLabelsFeature(poca::core::ImageInterface* _image) {
    std::vector <float>& labels = static_cast<poca::core::Histogram<float>*>(_image->getHistogram("label"))->getValues();
    std::vector <float>& features = static_cast<poca::core::Histogram<float>*>(_image->getCurrentHistogram())->getValues();
    uint32_t minLabel = _image->min(), maxLabel = _image->max();
    std::vector <float> values(maxLabel - minLabel + 1, 0.f);
    for (auto n = 0; n < labels.size(); n++) {
        uint32_t id = (uint32_t)labels[n];
        values[id - minLabel] = features[n];
    }
    const uint32_t nbValues = _image->width() * _image->height() * _image->depth();
    const auto minV = _image->getCurrentHistogram()->getCurrentMin(), maxV = _image->getCurrentHistogram()->getCurrentMax();

    thrust::device_vector<uint32_t> d_result(nbValues);
    thrust::device_vector<float> d_features(values);
    dim3 block(32);
    dim3 grid((nbValues + block.x - 1) / block.x);

    switch (_image->type())
    {
    case poca::core::UINT8:
    {
        poca::core::Image<uint8_t>* casted = static_cast <poca::core::Image<uint8_t>*>(_image);
        thrust::device_vector<uint8_t> d_pixels(casted->pixels());
        kernel_threshol_feature_label_gpu << <grid, block >> > (thrust::raw_pointer_cast(d_pixels.data()), thrust::raw_pointer_cast(d_features.data()), minV, maxV, (uint32_t)_image->min(), thrust::raw_pointer_cast(d_result.data()), nbValues);
    }
    break;
    case poca::core::UINT16:
    {
        poca::core::Image<uint16_t>* casted = static_cast <poca::core::Image<uint16_t>*>(_image);
        thrust::device_vector<uint16_t> d_pixels(casted->pixels());
        kernel_threshol_feature_label_gpu << <grid, block >> > (thrust::raw_pointer_cast(d_pixels.data()), thrust::raw_pointer_cast(d_features.data()), minV, maxV, (uint32_t)_image->min(), thrust::raw_pointer_cast(d_result.data()), nbValues);
    }
    break;
    case poca::core::UINT32:
    {
        poca::core::Image<uint32_t>* casted = static_cast <poca::core::Image<uint32_t>*>(_image);
        thrust::device_vector<uint32_t> d_pixels(casted->pixels());
        kernel_threshol_feature_label_gpu << <grid, block >> > (thrust::raw_pointer_cast(d_pixels.data()), thrust::raw_pointer_cast(d_features.data()), minV, maxV, (uint32_t)_image->min(), thrust::raw_pointer_cast(d_result.data()), nbValues);
    }
    break;
    default:
        break;
    }

    relabel_kernel_gpu<uint32_t>(d_result);
    uint32_t newMaxLabel = *thrust::max_element(d_result.begin(), d_result.end());
    poca::core::ImageInterface* image = NULL;
    if (newMaxLabel < std::numeric_limits<uint8_t>::max()) {
        image = convertAndCreateLabelImage<uint32_t, uint8_t>(d_result, _image->width(), _image->height(), _image->depth());
        image->setType(poca::core::UINT8);
    }
    else if (newMaxLabel < std::numeric_limits<uint16_t>::max()) {
        image = convertAndCreateLabelImage<uint32_t, uint16_t>(d_result, _image->width(), _image->height(), _image->depth());
        image->setType(poca::core::UINT16);
    }
    else {
        image = convertAndCreateLabelImage<uint32_t, uint32_t>(d_result, _image->width(), _image->height(), _image->depth());
        image->setType(poca::core::UINT32);
    }
    return image;
}

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
/*
template <class T>
__global__ void init_auto_threshold(const T* image, float* count, float* threshold, uint32_t width, uint32_t height, uint32_t depth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    int idx = IDX3(y, x, z, width, height);
    T value = image[idx];
    if (value > T(0)) {
        atomicAdd(count, 1.f);
        atomicAdd(threshold, float(value));
    }
}

template <class T>
__global__ void auto_threshold_step1(const T* image, float threshold, float* count0, float* count1, float* m0, float* m1, uint32_t width, uint32_t height, uint32_t depth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    int idx = IDX3(y, x, z, width, height);
    T value = image[idx];
    if (value > T(0)) {
        float valf = float(value);
        if (valf < threshold) {
            atomicAdd(count0, 1.f);
            atomicAdd(m0, valf);
        }
        else {
            atomicAdd(count1, 1.f);
            atomicAdd(m1, valf);
        }
    }
}

template <class T>
__global__ void auto_threshold_step2(const T* image, float threshold, float m0, float m1, float* s0, float* s1, uint32_t width, uint32_t height, uint32_t depth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    int idx = IDX3(y, x, z, width, height);
    T value = image[idx];
    if (value > T(0)) {
        float valf = float(value);
        if (valf < threshold) {
            atomicAdd(s0, (valf - m0) * (valf - m0));
        }
        else {
            atomicAdd(s1, (valf - m0) * (valf - m0));
        }
    }
}

enum AutoThresholdVar { THRESHOLD = 0, OLD_THRESHOLD = 1, M0 = 2, M1 = 3, COUNT0 = 4, COUNT1 = 5, S0 = 6, S1 = 7, SIGMA = 8, COUNT = 9 };

template <class T>
float getAutoThreshold(const T* image, const uint32_t _w, const uint32_t _h, const uint32_t _d)
{
    auto start = std::chrono::high_resolution_clock::now();
    uint32_t numel = _w * _h * _d;
    thrust::device_vector <T> d_image(image, image + numel);
    thrust::device_vector <float> d_values(10);//0 -> 9: {threshold, oldThreshold, m0, m1, count0, count1, count, s0, s1, sigma}
    uint32_t cpt = 30;
    float threshold = 0.f, oldThreshold = 1e10;
    d_values[OLD_THRESHOLD] = oldThreshold;

    dim3 threads = _d == 1 ? dim3(BLOCKDIM_X, BLOCKDIM_Y) : dim3(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
    dim3 grid = _d == 1 ? dim3((unsigned int)ceil((float)_w / (float)BLOCKDIM_X), (unsigned int)ceil((float)_h / (float)BLOCKDIM_Y)) : dim3((unsigned int)ceil((float)_w / (float)BLOCKDIM_X), (unsigned int)ceil((float)_h / (float)BLOCKDIM_Y), (unsigned int)ceil((float)_d / (float)BLOCKDIM_Z));

    float* d_values_ptr = thrust::raw_pointer_cast(d_values.data());
    init_auto_threshold << <grid, threads >> > (thrust::raw_pointer_cast(d_image.data()), d_values_ptr + COUNT, d_values_ptr + THRESHOLD, _w, _h, _d);
    d_values[THRESHOLD] = d_values[THRESHOLD] / d_values[COUNT];
    threshold = d_values[THRESHOLD];
    while ((fabs(oldThreshold - threshold) > 1e-12) && (cpt > 0)) {
        auto start2 = std::chrono::high_resolution_clock::now(), start3 = start2;
        thrust::fill(d_values.begin() + M0, d_values.begin() + SIGMA, 0.f);
        auto_threshold_step1 << <grid, threads >> > (thrust::raw_pointer_cast(d_image.data()), d_values[THRESHOLD], d_values_ptr + COUNT0, d_values_ptr + COUNT1, d_values_ptr + M0, d_values_ptr + M1, _w, _h, _d);
        printf("step 1: %lld ms\n", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start3).count());
        start3 = std::chrono::high_resolution_clock::now();
        d_values[M0] = d_values[M0] / d_values[COUNT0];
        d_values[M1] = d_values[M1] / d_values[COUNT1];
        printf("other: %lld ms\n", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start3).count());
        start3 = std::chrono::high_resolution_clock::now();
        auto_threshold_step2 << <grid, threads >> > (thrust::raw_pointer_cast(d_image.data()), d_values[THRESHOLD], d_values[M0], d_values[M1], d_values_ptr + S0, d_values_ptr + S1, _w, _h, _d);
        printf("step 2: %lld ms\n", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start3).count());
        start3 = std::chrono::high_resolution_clock::now();
        d_values[S0] = sqrt(d_values[S0]);
        d_values[S1] = sqrt(d_values[S1]);
        d_values[SIGMA] = (d_values[S0] + d_values[S1]) / d_values[COUNT];
        d_values[OLD_THRESHOLD] = d_values[THRESHOLD];
        d_values[THRESHOLD] = (d_values[M0] + d_values[M1]) / 2.f + d_values[SIGMA] * d_values[SIGMA] * log(d_values[COUNT0] / d_values[COUNT1]) / (d_values[M1] - d_values[M0]);
        threshold = d_values[THRESHOLD];
        oldThreshold = d_values[OLD_THRESHOLD];
        printf("other: %lld ms\n", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start3).count());
        start3 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::high_resolution_clock::now() - start2;
        long long ms = std::chrono::duration_cast<std::chrono::microseconds>(duration2).count();
        float s = std::chrono::duration_cast<std::chrono::seconds>(duration2).count();
        printf("iteration %u, value %f, oldvalue %f, took %f seconds (%lld microseconds)\n", cpt, threshold, oldThreshold, s, ms);
        cpt--;
        printf("timing: %lld ms\n", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start3).count());
    }
    auto duration = std::chrono::high_resolution_clock::now() - start;
    long long ms = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    float s = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    printf("auto threshold, value %f, took %f seconds (%lld microseconds)\n", threshold, s, ms);
    return threshold;
}
*/
template <typename T>
float getAutoThreshold(const T* image, uint32_t w, uint32_t h, uint32_t d) {
    auto start = std::chrono::high_resolution_clock::now();
    using namespace thrust;

    uint32_t numel = w * h * d;

    // Upload input image to device
    device_vector<T> d_image(image, image + numel);

    // Cast image to float for math
    device_vector<float> d_float_image(numel);
    transform(d_image.begin(), d_image.end(), d_float_image.begin(), thrust::placeholders::_1 * 1.0f);

    // Initial threshold = mean
    // Initial mean of values > 0
    float sum_pos = thrust::transform_reduce(
        d_float_image.begin(), d_float_image.end(),
        [] __device__(float x) {
        return x > 0.0f ? x : 0.0f;
    },
        0.0f, thrust::plus<float>()
    );

    int count_pos = thrust::count_if(
        d_float_image.begin(), d_float_image.end(),
        [] __device__(float x) {
        return x > 0.0f;
    }
    );

    float threshold = (count_pos > 0) ? (sum_pos / count_pos) : 0.0f;
    float oldThreshold = 1e10f;

    int cpt = 30;

    while (fabs(oldThreshold - threshold) > 1e-12f && cpt-- > 0) {
        oldThreshold = threshold;

        auto valid_and_leq_thresh = [threshold] __device__(float x) {
            return x > 0.0f && x <= threshold;
        };

        auto valid_and_gt_thresh = [threshold] __device__(float x) {
            return x > threshold;
        };

        // Count and mean of values in (0, threshold]
        int count0 = count_if(d_float_image.begin(), d_float_image.end(), valid_and_leq_thresh);
        float sum0 = transform_reduce(
            d_float_image.begin(), d_float_image.end(),
            [threshold] __device__(float x) {
            return (x > 0.0f && x <= threshold) ? x : 0.0f;
        },
            0.0f, plus<float>()
        );

        // Count and mean of values > threshold
        int count1 = count_if(d_float_image.begin(), d_float_image.end(), valid_and_gt_thresh);
        float sum1 = transform_reduce(
            d_float_image.begin(), d_float_image.end(),
            [threshold] __device__(float x) {
            return (x > threshold) ? x : 0.0f;
        },
            0.0f, plus<float>()
        );

        float m0 = (count0 > 0) ? (sum0 / count0) : 0.0f;
        float m1 = (count1 > 0) ? (sum1 / count1) : 0.0f;

        // Variance for <= threshold (only > 0 values)
        float s0 = transform_reduce(
            d_float_image.begin(), d_float_image.end(),
            [threshold, m0] __device__(float x) {
            return (x > 0.0f && x <= threshold) ? (x - m0) * (x - m0) : 0.0f;
        },
            0.0f, plus<float>()
        );

        // Variance for > threshold
        float s1 = transform_reduce(
            d_float_image.begin(), d_float_image.end(),
            [threshold, m1] __device__(float x) {
            return (x > threshold) ? (x - m1) * (x - m1) : 0.0f;
        },
            0.0f, plus<float>()
        );

        s0 = sqrtf(s0);
        s1 = sqrtf(s1);
        float sigma = (s0 + s1) / (count0 + count1);

        if (count0 > 0 && count1 > 0 && m1 != m0) {
            threshold = (m0 + m1) / 2.0f +
                sigma * sigma * logf((float)count0 / count1) / (m1 - m0);
        }
        else {
            break;
        }
    }
    auto duration = std::chrono::high_resolution_clock::now() - start;
    long long ms = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    float s = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    printf("auto threshold, value %f, took %f seconds (%lld microseconds)\n", threshold, s, ms);
    return threshold;
}

template poca::core::ImageInterface* convertAndCreateLabelImage<uint32_t, uint8_t>(thrust::device_vector<uint32_t>& d_labels, const uint32_t _w, const uint32_t _h, const uint32_t _d);
template poca::core::ImageInterface* convertAndCreateLabelImage<uint32_t, uint16_t>(thrust::device_vector<uint32_t>& d_labels, const uint32_t _w, const uint32_t _h, const uint32_t _d);
template poca::core::ImageInterface* convertAndCreateLabelImage<uint32_t, uint32_t>(thrust::device_vector<uint32_t>& d_labels, const uint32_t _w, const uint32_t _h, const uint32_t _d);

template float getAutoThreshold(const uint8_t* image, const uint32_t _w, const uint32_t _h, const uint32_t _d);
template float getAutoThreshold(const uint16_t* image, const uint32_t _w, const uint32_t _h, const uint32_t _d);
template float getAutoThreshold(const uint32_t* image, const uint32_t _w, const uint32_t _h, const uint32_t _d);
template float getAutoThreshold(const int32_t* image, const uint32_t _w, const uint32_t _h, const uint32_t _d);
template float getAutoThreshold(const float* image, const uint32_t _w, const uint32_t _h, const uint32_t _d);

template __global__ void kernel_threshold(const uint8_t* image, const uint8_t _thresholdMin, const uint8_t _thresholdMax, uint8_t* thresholdedImage, uint32_t size);
template __global__ void kernel_threshold(const uint16_t* image, const uint16_t _thresholdMin, const uint16_t _thresholdMax, uint8_t* thresholdedImage, uint32_t size);
template __global__ void kernel_threshold(const uint32_t* image, const uint32_t _thresholdMin, const uint32_t _thresholdMax, uint8_t* thresholdedImage, uint32_t size);
template __global__ void kernel_threshold(const int32_t* image, const int32_t _thresholdMin, const int32_t _thresholdMax, uint8_t* thresholdedImage, uint32_t size);
template __global__ void kernel_threshold(const float* image, const float _thresholdMin, const float _thresholdMax, uint8_t* thresholdedImage, uint32_t size);
#endif