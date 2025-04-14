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

#include "BasicOperationsImage.h"

#ifndef NO_CUDA
#define cuda_check(x) if (x!=cudaSuccess) exit(1);
#define IF_VERBOSE(x) //x

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

template poca::core::ImageInterface* convertAndCreateLabelImage<uint32_t, uint8_t>(thrust::device_vector<uint32_t>& d_labels, const uint32_t _w, const uint32_t _h, const uint32_t _d);
template poca::core::ImageInterface* convertAndCreateLabelImage<uint32_t, uint16_t>(thrust::device_vector<uint32_t>& d_labels, const uint32_t _w, const uint32_t _h, const uint32_t _d);
template poca::core::ImageInterface* convertAndCreateLabelImage<uint32_t, uint32_t>(thrust::device_vector<uint32_t>& d_labels, const uint32_t _w, const uint32_t _h, const uint32_t _d);

template __global__ void kernel_threshold(const uint8_t* image, const uint8_t _thresholdMin, const uint8_t _thresholdMax, uint8_t* thresholdedImage, uint32_t size);
template __global__ void kernel_threshold(const uint16_t* image, const uint16_t _thresholdMin, const uint16_t _thresholdMax, uint8_t* thresholdedImage, uint32_t size);
template __global__ void kernel_threshold(const uint32_t* image, const uint32_t _thresholdMin, const uint32_t _thresholdMax, uint8_t* thresholdedImage, uint32_t size);
template __global__ void kernel_threshold(const int32_t* image, const int32_t _thresholdMin, const int32_t _thresholdMax, uint8_t* thresholdedImage, uint32_t size);
template __global__ void kernel_threshold(const float* image, const float _thresholdMin, const float _thresholdMax, uint8_t* thresholdedImage, uint32_t size);
#endif