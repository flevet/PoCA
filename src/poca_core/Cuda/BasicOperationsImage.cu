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
    /*thrust::sort(thrust::device, d_pixels.begin(), d_pixels.end());
    thrust::device_vector<T> d_unique = d_pixels;
    const auto end = thrust::unique(d_unique.begin(), d_unique.end());
    int nbUniqueLabels = thrust::distance(d_unique.begin(), end);
    thrust::device_vector<T> ones(d_pixels.size(), 1);
    d_labels.resize(nbUniqueLabels);
    d_counts.resize(nbUniqueLabels);
    //thrust::copy(d_unique.begin(), end, std::ostream_iterator<T>(std::cout, ","));
    thrust::equal_to<T> binary_pred;
    thrust::plus<T> binary_op;
    auto new_end = thrust::reduce_by_key(thrust::device, d_pixels.begin(), d_pixels.end(), ones.begin(), d_labels.begin(), d_counts.begin(), binary_pred, binary_op);*/
    //std::cout << std::endl;
    //thrust::copy(d_labels.begin(), new_end.first, std::ostream_iterator<T>(std::cout, ","));
    //std::cout << std::endl;
    //thrust::copy(d_counts.begin(), new_end.second, std::ostream_iterator<T>(std::cout, ","));
     
    thrust::device_vector <T> d_pixels_tmp(d_pixels);
    // Sort pixels to prepare for reduce_by_key
    thrust::sort(thrust::device, d_pixels_tmp.begin(), d_pixels_tmp.end());

    // Prepare a vector of ones for counting
    thrust::device_vector<T> ones(d_pixels_tmp.size(), 1);

    // Resize output containers to a maximum possible size
    d_labels.resize(d_pixels_tmp.size());
    d_counts.resize(d_pixels_tmp.size());

    // Reduce by key to count occurrences
    auto new_end = thrust::reduce_by_key(
        thrust::device,
        d_pixels_tmp.begin(), d_pixels_tmp.end(),
        ones.begin(),
        d_labels.begin(),
        d_counts.begin(),
        thrust::equal_to<T>(),
        thrust::plus<T>());

    // Resize output vectors to the actual number of unique labels
    size_t num_unique = thrust::distance(d_labels.begin(), new_end.first);
    d_labels.resize(num_unique);
    d_counts.resize(num_unique);
}

template <class T>
void remove_small_labels_kernel_gpu(thrust::device_vector<T>& d_pixels, T threshold)
{
    thrust::device_vector<T> d_labels(d_pixels.size());
    thrust::device_vector<T> d_counts(d_pixels.size());

    count_occurences_label_kernel_gpu(d_pixels, d_labels, d_counts);

    // 3. Build LUT: set to new consecutive label if count >= threshold, else 0
    T max_label = d_labels.back();
    thrust::device_vector<T> label_lut(max_label + 1, 0);  // initialize with 0 (background)

    // Assign new consecutive labels starting from 1
    T next_label = 1;
    for (size_t i = 0; i < num_unique; ++i)
    {
        if (d_counts[i] >= threshold)
        {
            label_lut[d_labels[i]] = next_label;
            ++next_label;
        }
        else
        {
            label_lut[d_labels[i]] = 0; // Remove label: set to background
        }
    }

    // 4. Remap d_pixels using the LUT
    thrust::transform(
        thrust::device,
        d_pixels.begin(),
        d_pixels.end(),
        d_pixels.begin(),
        [lut = label_lut.data()] __device__(T pixel_label) {
        return (pixel_label < static_cast<T>(thrust::distance(lut, lut + pixel_label + 1)))
            ? lut[pixel_label]
            : static_cast<T>(0);
    });
}

template <class T>
__global__ void kernel_threshold(const T* image, const T _thresholdMin, const T _thresholdMax, uint8_t* thresholdedImage, uint32_t size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 0 || tid >= size) return;
    T value = image[tid];
    thresholdedImage[tid] = value >= _thresholdMin && value <= _thresholdMax ? 255 : 0;
}

template <class T>
__global__ void kernel_threshold32(const T* image, const T _thresholdMin, const T _thresholdMax, int32_t* thresholdedImage, uint32_t size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 0 || tid >= size) return;
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

__global__ void increment_plane_counts_kernel(const uint32_t* unique_labels,
    size_t n_unique,
    uint32_t* planeCounts)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_unique) {
        uint32_t lab = unique_labels[i];
        atomicAdd(&planeCounts[lab], 1u);
    }
}

// Computes: planeCounts[label] = number of z-planes where label appears at least once.
// Assumes labels are dense in [0..maxLabel].
// Output planeCounts size = maxLabel+1.
void count_zplanes_dense_labels_gpu(const thrust::device_vector<uint32_t>& d_pixels,
    thrust::device_vector<uint32_t>& d_planeCounts,
    uint32_t width, uint32_t height, uint32_t depth)
{
    const size_t wh = size_t(width) * size_t(height);
    const size_t n = d_pixels.size(); // should be wh*depth

    // 1) find max label (dense range size)
    uint32_t maxLabel = *thrust::max_element(thrust::device, d_pixels.begin(), d_pixels.end());

    d_planeCounts.resize(size_t(maxLabel) + 1);
    thrust::fill(thrust::device, d_planeCounts.begin(), d_planeCounts.end(), 0u);

    // reusable buffers for one slice
    thrust::device_vector<uint32_t> d_slice(wh);

    const int block = 256;

    for (uint32_t z = 0; z < depth; ++z) {
        // 2) copy slice to temp buffer
        const size_t offset = size_t(z) * wh;
        thrust::copy(thrust::device,
            d_pixels.begin() + offset,
            d_pixels.begin() + offset + wh,
            d_slice.begin());

        // 3) sort and unique to get labels present in this plane
        thrust::sort(thrust::device, d_slice.begin(), d_slice.end());
        auto end_unique = thrust::unique(thrust::device, d_slice.begin(), d_slice.end());
        size_t m = size_t(thrust::distance(d_slice.begin(), end_unique));

        // 4) increment plane count once per unique label
        if (m > 0) {
            int grid = int((m + block - 1) / block);
            increment_plane_counts_kernel << <grid, block >> > (
                thrust::raw_pointer_cast(d_slice.data()),
                m,
                thrust::raw_pointer_cast(d_planeCounts.data()));
        }
    }

    // If you want strict error reporting:
    // cudaError_t err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) { ... }
}

void relabelI32(std::vector <uint32_t>& _labels, std::vector <uint32_t>& _relabels)
{
    _relabels.resize(_labels.size());
    thrust::device_vector<int32_t> d_result(_labels);
    relabel_kernel_gpu(d_result);
    cudaMemcpy(_relabels.data(), thrust::raw_pointer_cast(d_result.data()), _labels.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "Here" << std::endl;
}

template <typename HostT>
thrust::device_vector<HostT> upload_to_device(const std::vector<HostT>& h)
{
    thrust::device_vector<HostT> d(h.size());
    cudaMemcpy(thrust::raw_pointer_cast(d.data()),
        h.data(),
        h.size() * sizeof(HostT),
        cudaMemcpyHostToDevice);
    return d;
}

template <typename InT>
thrust::device_vector<uint32_t> to_u32(const thrust::device_vector<InT>& d_in)
{
    thrust::device_vector<uint32_t> d_out(d_in.size());
    thrust::transform(thrust::device,  d_in.begin(), d_in.end(), d_out.begin(), [] __host__ __device__(InT v) { return static_cast<uint32_t>(v); });
    return d_out;
}

void copy_u32_to_host_float(const thrust::device_vector<uint32_t>& d_u32, std::vector<float>& h_f, size_t offset = 0)
{
    const size_t n = d_u32.size() - offset;
    thrust::device_vector<float> d_f(n);

    thrust::transform(thrust::device,  d_u32.begin() + offset, d_u32.end(), d_f.begin(), [] __host__ __device__(uint32_t v) { return static_cast<float>(v); });

    h_f.resize(n);
    cudaMemcpy(h_f.data(),  thrust::raw_pointer_cast(d_f.data()), n * sizeof(float), cudaMemcpyDeviceToHost);
}


void computeFeaturesLabelImage(poca::core::ImageInterface* _image)
{
    thrust::device_vector<uint32_t> d_labels, d_counts;
    switch (_image->type())
    {
    case poca::core::UINT8:
    {
        poca::core::Image<uint8_t>* casted = static_cast <poca::core::Image<uint8_t>*>(_image);
        auto d_u8 = upload_to_device<uint8_t>(casted->pixels());   // no thrust cross-system copy
        auto d_pixels = to_u32<uint8_t>(d_u8);
        //thrust::device_vector<uint32_t> d_pixels(casted->pixels());
        count_occurences_label_kernel_gpu< uint32_t>(d_pixels, d_labels, d_counts);
    }
    break;
    case poca::core::UINT16:
    {
        poca::core::Image<uint16_t>* casted = static_cast <poca::core::Image<uint16_t>*>(_image);
        auto d_u16 = upload_to_device<uint16_t>(casted->pixels());   // no thrust cross-system copy
        auto d_pixels = to_u32<uint16_t>(d_u16);
        //thrust::device_vector<uint32_t> d_pixels(casted->pixels());
        count_occurences_label_kernel_gpu<uint32_t>(d_pixels, d_labels, d_counts);
    }
    break;
    case poca::core::UINT32:
    {
        poca::core::Image<uint32_t>* casted = static_cast <poca::core::Image<uint32_t>*>(_image);
        auto d_pixels = upload_to_device<uint32_t>(casted->pixels());   // no thrust cross-system copy
        count_occurences_label_kernel_gpu<uint32_t>(d_pixels, d_labels, d_counts);
    }
    break;
    default:
        break;
    }
    std::vector <float>& volume = _image->volumes();
    //volume.resize(d_counts.size() - 1);
    std::vector <float> label;// (d_counts.size() - 1);

    // skip background (index 0) like your original code
    copy_u32_to_host_float(d_labels, label, 1);
    copy_u32_to_host_float(d_counts, volume, 1);

    //cudaMemcpy(label.data(), thrust::raw_pointer_cast(d_labels.data() + 1), label.size() * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(volume.data(), thrust::raw_pointer_cast(d_counts.data() + 1), volume.size() * sizeof(float), cudaMemcpyDeviceToHost);
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
    /*if (newMaxLabel < std::numeric_limits<uint8_t>::max()) {
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
    }*/
    image = convertAndCreateLabelImage<uint32_t, uint32_t>(d_result, _image->width(), _image->height(), _image->depth());
    image->setType(poca::core::UINT32);
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

uint32_t relabel_kernel_uint32t_gpu(thrust::device_vector<uint32_t>& d_labels)
{
    thrust::device_vector<uint32_t> d_data(d_labels);
    // Sort the data copy
    thrust::sort(thrust::device, d_labels.begin(), d_labels.end());
    // Allocate an array to store unique values
    thrust::device_vector<uint32_t> d_unique = d_labels;
    // Compress all duplicates
    const auto end = thrust::unique(d_unique.begin(), d_unique.end());
    // Search for all original labels, in this compressed range, and write their
    // indices back as the result 
    thrust::lower_bound(d_unique.begin(), end, d_data.begin(), d_data.end(), d_labels.begin());
    return thrust::distance(d_unique.begin(), end);
}

template <class T>
__global__ void binarize(const T* image, uint8_t* bimage, uint32_t size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 0 || tid >= size) return;
    T value = image[tid];
    bimage[tid] = value > T(0) ? 255 : 0;
}

template <class T, class M>
__global__ void pad2D(const T* src, M* dest, const uint32_t _w, const uint32_t _h, const uint32_t _pad)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= _w || y >= _h) { return; }

    uint32_t wdest = _w + 2 * _pad;

    T val = src[_w * y + x];
    dest[wdest * (y + _pad) + (x + _pad)] = static_cast<M>(val);
}

template <class T, class M>
__global__ void pad3D(const T* src, M* dest, const uint32_t _w, const uint32_t _h, const uint32_t _d, const uint32_t _pad)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= _w || y >= _h || z >= _d) { return; }

    uint32_t wdest = _w + 2 * _pad, hdest = _h + 2 * _pad;
    uint32_t who = _w * _h, whd = wdest * hdest;
    T val = src[who * z + _w * y + x];
    dest[whd * (z + _pad) + wdest * (y + _pad) + (x + _pad)] = static_cast<M>(val);
}

template <class T, class M>
__global__ void unpad2D(const T* src, M* dest, const uint32_t _w, const uint32_t _h, const uint32_t _pad)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= _w || y >= _h) { return; }

    uint32_t wdest = _w + 2 * _pad;

    T val = src[wdest * (y + _pad) + (x + _pad)];
     dest[_w * y + x] = M(val);
}

template <class T, class M>
__global__ void unpad3D(const T* src, M* dest, const uint32_t _w, const uint32_t _h, const uint32_t _d, const uint32_t _pad)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= _w || y >= _h || z >= _d) { return; }

    uint32_t wdest = _w + 2 * _pad, hdest = _h + 2 * _pad;
    uint32_t who = _w * _h, whd = wdest * hdest;
    T val = src[whd * (z + _pad) + wdest * (y + _pad) + (x + _pad)];
    dest[who * z + _w * y + x] = static_cast<M>(val);
}

template <class T, class M>
void pad(const thrust::device_vector<T>& _source, thrust::device_vector<M>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad)
{
    uint32_t wdest = _w + 2 * _pad, hdest = _h + 2 * _pad, ddest = _d == 1 ? 1 : _d + 2 * _pad;
    _output.resize(wdest * hdest * ddest);
    thrust::fill(_output.begin(), _output.end(), M(0));
    dim3 threads = _d == 1 ? dim3(8, 8) : dim3(8, 8, 8);
    dim3 grid = _d == 1 ? dim3((unsigned int)ceil((float)_w / (float)8), (unsigned int)ceil((float)_h / (float)8)) : dim3((unsigned int)ceil((float)_w / (float)8), (unsigned int)ceil((float)_h / (float)8), (unsigned int)ceil((float)_d / (float)8));
    if (_d == 1) 
        pad2D << <grid, threads >> > (thrust::raw_pointer_cast(_source.data()), thrust::raw_pointer_cast(_output.data()), _w, _h, _pad);
    else
        pad3D << <grid, threads >> > (thrust::raw_pointer_cast(_source.data()), thrust::raw_pointer_cast(_output.data()), _w, _h, _d, _pad);
}

template <class T, class M>
void unpad(const thrust::device_vector<T>& _source, thrust::device_vector<M>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad)
{
    uint32_t wdest = _w - 2 * _pad, hdest = _h - 2 * _pad, ddest = _d == 1 ? 1 : _d - 2 * _pad;
    _output.resize(wdest * hdest * ddest);
    thrust::fill(_output.begin(), _output.end(), M(0));
    dim3 threads = _d == 1 ? dim3(8, 8) : dim3(8, 8, 8);
    dim3 grid = _d == 1 ? dim3((unsigned int)ceil((float)wdest / (float)8), (unsigned int)ceil((float)hdest / (float)8)) : dim3((unsigned int)ceil((float)wdest / (float)8), (unsigned int)ceil((float)hdest / (float)8), (unsigned int)ceil((float)ddest / (float)8));
    if (_d == 1)
        unpad2D << <grid, threads >> > (thrust::raw_pointer_cast(_source.data()), thrust::raw_pointer_cast(_output.data()), wdest, hdest, _pad);
    else
        unpad3D << <grid, threads >> > (thrust::raw_pointer_cast(_source.data()), thrust::raw_pointer_cast(_output.data()), wdest, hdest, ddest, _pad);
}

template <class T>
__global__ void maxProjection_kernel(const T* __restrict__ src, T* __restrict__ dest, uint32_t w, uint32_t h, uint32_t d)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    // starting index for this (x, y) at z = 0
    uint32_t id_src = y * w + x;
    T maxv = src[id_src];

    const uint32_t sliceStride = w * h;

    for (uint32_t z = 1; z < d; ++z) {
        id_src += sliceStride;
        T v = src[id_src];
        if (v > maxv) maxv = v;
    }

    dest[y * w + x] = maxv;
}


template <class T>
void maxProjection(const std::vector<T>& _source, std::vector<T>& _output, uint32_t _w, uint32_t _h, uint32_t _d)
{
    // copy input to device
    thrust::device_vector<T> src(_source);
    thrust::device_vector<T> dest(_w * _h);

    _output.resize(_w * _h);

    dim3 threads(16, 16);
    dim3 grid((_w + threads.x - 1) / threads.x, (_h + threads.y - 1) / threads.y);

    maxProjection_kernel << <grid, threads >> > (thrust::raw_pointer_cast(src.data()), thrust::raw_pointer_cast(dest.data()), _w, _h, _d);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpy(_output.data(), thrust::raw_pointer_cast(dest.data()), _output.size() * sizeof(T), cudaMemcpyDeviceToHost);
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

template __global__ void binarize(const uint8_t* image, uint8_t* bimage, uint32_t size);
template __global__ void binarize(const uint16_t* image, uint8_t* bimage, uint32_t size);
template __global__ void binarize(const uint32_t* image, uint8_t* bimage, uint32_t size);
template __global__ void binarize(const float* image, uint8_t* bimage, uint32_t size);

template void count_occurences_label_kernel_gpu(thrust::device_vector<uint8_t>& d_pixels, thrust::device_vector<uint8_t>& d_labels, thrust::device_vector<uint8_t>& d_counts);
template void count_occurences_label_kernel_gpu(thrust::device_vector<uint16_t>& d_pixels, thrust::device_vector<uint16_t>& d_labels, thrust::device_vector<uint16_t>& d_counts);
template void count_occurences_label_kernel_gpu(thrust::device_vector<uint32_t>& d_pixels, thrust::device_vector<uint32_t>& d_labels, thrust::device_vector<uint32_t>& d_counts);
template void count_occurences_label_kernel_gpu(thrust::device_vector<float>& d_pixels, thrust::device_vector<float>& d_labels, thrust::device_vector<float>& d_counts);

template void pad(const thrust::device_vector<uint8_t>& _source, thrust::device_vector<uint8_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void pad(const thrust::device_vector<uint8_t>& _source, thrust::device_vector<uint16_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void pad(const thrust::device_vector<uint8_t>& _source, thrust::device_vector<uint32_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void pad(const thrust::device_vector<uint8_t>& _source, thrust::device_vector<float>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);

template void pad(const thrust::device_vector<uint16_t>& _source, thrust::device_vector<uint8_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void pad(const thrust::device_vector<uint16_t>& _source, thrust::device_vector<uint16_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void pad(const thrust::device_vector<uint16_t>& _source, thrust::device_vector<uint32_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void pad(const thrust::device_vector<uint16_t>& _source, thrust::device_vector<float>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);

template void pad(const thrust::device_vector<uint32_t>& _source, thrust::device_vector<uint8_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void pad(const thrust::device_vector<uint32_t>& _source, thrust::device_vector<uint16_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void pad(const thrust::device_vector<uint32_t>& _source, thrust::device_vector<uint32_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void pad(const thrust::device_vector<uint32_t>& _source, thrust::device_vector<float>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);

template void pad(const thrust::device_vector<float>& _source, thrust::device_vector<uint8_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void pad(const thrust::device_vector<float>& _source, thrust::device_vector<uint16_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void pad(const thrust::device_vector<float>& _source, thrust::device_vector<uint32_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void pad(const thrust::device_vector<float>& _source, thrust::device_vector<float>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);

template void unpad(const thrust::device_vector<uint16_t>& _source, thrust::device_vector<uint8_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void unpad(const thrust::device_vector<uint16_t>& _source, thrust::device_vector<uint16_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void unpad(const thrust::device_vector<uint16_t>& _source, thrust::device_vector<uint32_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void unpad(const thrust::device_vector<uint16_t>& _source, thrust::device_vector<float>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);

template void unpad(const thrust::device_vector<uint32_t>& _source, thrust::device_vector<uint8_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void unpad(const thrust::device_vector<uint32_t>& _source, thrust::device_vector<uint16_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void unpad(const thrust::device_vector<uint32_t>& _source, thrust::device_vector<uint32_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void unpad(const thrust::device_vector<uint32_t>& _source, thrust::device_vector<float>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);

template void unpad(const thrust::device_vector<float>& _source, thrust::device_vector<uint8_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void unpad(const thrust::device_vector<float>& _source, thrust::device_vector<uint16_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void unpad(const thrust::device_vector<float>& _source, thrust::device_vector<uint32_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void unpad(const thrust::device_vector<float>& _source, thrust::device_vector<float>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);

template void unpad(const thrust::device_vector<uint8_t>& _source, thrust::device_vector<uint8_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void unpad(const thrust::device_vector<uint8_t>& _source, thrust::device_vector<uint16_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void unpad(const thrust::device_vector<uint8_t>& _source, thrust::device_vector<uint32_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);
template void unpad(const thrust::device_vector<uint8_t>& _source, thrust::device_vector<float>& _output, uint32_t _w, uint32_t _h, uint32_t _d, uint32_t _pad);

template void maxProjection(const std::vector<uint8_t>& _source, std::vector<uint8_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d);
template void maxProjection(const std::vector<uint16_t>& _source, std::vector<uint16_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d);
template void maxProjection(const std::vector<uint32_t>& _source, std::vector<uint32_t>& _output, uint32_t _w, uint32_t _h, uint32_t _d);
template void maxProjection(const std::vector<float>& _source, std::vector<float>& _output, uint32_t _w, uint32_t _h, uint32_t _d);
#endif