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

#include <thrust\pair.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\extrema.h>
#include <thrust\sort.h>
#include <thrust\unique.h>
#include <thrust\sequence.h>
#include <thrust\distance.h>
#include <thrust/binary_search.h>

#include "ConnectedComponents.h"

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

template <class T, class M>
void count_occurences_label_kernel_gpu(thrust::device_vector<T>& d_pixels, thrust::device_vector<M>& d_labels, thrust::device_vector<M>& d_counts)
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

template <typename T>
__device__ __forceinline__ unsigned char hasBit(T bitmap, unsigned char pos) {
    return (bitmap >> pos) & 1;
}


__device__ uint32_t find(const uint32_t* s_buf, uint32_t n) {
    while (s_buf[n] != n)
        n = s_buf[n];
    return n;
}

__device__ uint32_t find_n_compress(uint32_t* s_buf, uint32_t n) {
    const uint32_t id = n;
    while (s_buf[n] != n) {
        n = s_buf[n];
        s_buf[id] = n;
    }
    return n;
}

__device__ void union_(uint32_t* s_buf, uint32_t a, uint32_t b)
{
    bool done;
    do
    {
        a = find(s_buf, a);
        b = find(s_buf, b);

        if (a < b) {
            int32_t old = atomicMin(s_buf + b, a);
            done = (old == b);
            b = old;
        }
        else if (b < a) {
            int32_t old = atomicMin(s_buf + a, b);
            done = (old == a);
            a = old;
        }
        else
            done = true;

    } while (!done);
}

__global__ void init_labeling(uint32_t* label, const uint32_t W, const uint32_t H, const uint32_t D) {
    const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const uint32_t z = (blockIdx.z * blockDim.z + threadIdx.z) * 2;

    const uint32_t idx = z * W * H + y * W + x;

    if (x < W && y < H && z < D)
        label[idx] = idx;
}

__global__ void merge(uint8_t* const img, uint32_t* label, uint8_t* last_cube_fg, const uint32_t W, const uint32_t H, const uint32_t D)
{
    const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const uint32_t z = (blockIdx.z * blockDim.z + threadIdx.z) * 2;
    const uint32_t stepz = W * H;
    const uint32_t idx = z * stepz + y * W + x;

    if (x >= W || y >= H || z >= D)
        return;

    uint64_t P = 0;
    uint8_t fg = 0;

    uint16_t buffer;
#define P0 0x77707770777
#define CHECK_BUF(P_shift, fg_shift, P_shift2, fg_shift2)  \
    if (buffer & 1) {                                      \
        P |= P0 << (P_shift);                              \
        fg |= 1 << (fg_shift);                             \
    }                                                      \
    if (buffer & (1 << 8)) {                               \
        P |= P0 << (P_shift2);                             \
        fg |= 1 << (fg_shift2);                            \
    }

    if (x + 1 < W) {
        buffer = *reinterpret_cast<uint16_t*>(img + idx);
        CHECK_BUF(0, 0, 1, 1)

            if (y + 1 < H) {
                buffer = *reinterpret_cast<uint16_t*>(img + idx + W);
                CHECK_BUF(4, 2, 5, 3)
            }

        if (z + 1 < D) {
            buffer = *reinterpret_cast<uint16_t*>(img + idx + stepz);
            CHECK_BUF(16, 4, 17, 5)

                if (y + 1 < H) {
                    buffer = *reinterpret_cast<uint16_t*>(img + idx + stepz + W);
                    CHECK_BUF(20, 6, 21, 7)
                }
        }
    }
#undef CHECK_BUF

    // Store fg voxels bitmask into memory
    if (x + 1 < W)          label[idx + 1] = fg;
    else if (y + 1 < H)     label[idx + W] = fg;
    else if (z + 1 < D)     label[idx + stepz] = fg;
    else                    *last_cube_fg = fg;

    // checks on borders

    if (x == 0)             P &= 0xEEEEEEEEEEEEEEEE;
    if (x + 1 >= W)         P &= 0x3333333333333333;
    else if (x + 2 >= W)    P &= 0x7777777777777777;

    if (y == 0)             P &= 0xFFF0FFF0FFF0FFF0;
    if (y + 1 >= H)         P &= 0x00FF00FF00FF00FF;
    else if (y + 2 >= H)    P &= 0x0FFF0FFF0FFF0FFF;

    if (z == 0)             P &= 0xFFFFFFFFFFFF0000;
    if (z + 1 >= D)         P &= 0x00000000FFFFFFFF;
    // else if (z + 2 >= D)    P &= 0x0000FFFFFFFFFFFF;

    if (P > 0) {
        // Lower plane
        const uint32_t img_idx = idx - stepz;
        const uint32_t label_idx = idx - 2 * stepz;

        if (hasBit(P, 0) && img[img_idx - W - 1])
            union_(label, idx, label_idx - 2 * W - 2);

        if ((hasBit(P, 1) && img[img_idx - W]) || (hasBit(P, 2) && img[img_idx - W + 1]))
            union_(label, idx, label_idx - 2 * W);

        if (hasBit(P, 3) && img[img_idx - W + 2])
            union_(label, idx, label_idx - 2 * W + 2);

        if ((hasBit(P, 4) && img[img_idx - 1]) || (hasBit(P, 8) && img[img_idx + W - 1]))
            union_(label, idx, label_idx - 2);

        if ((hasBit(P, 5) && img[img_idx]) || (hasBit(P, 6) && img[img_idx + 1]) || \
            (hasBit(P, 9) && img[img_idx + W]) || (hasBit(P, 10) && img[img_idx + W + 1]))
            union_(label, idx, label_idx);

        if ((hasBit(P, 7) && img[img_idx + 2]) || (hasBit(P, 11) && img[img_idx + W + 2]))
            union_(label, idx, label_idx + 2);

        if (hasBit(P, 12) && img[img_idx + 2 * W - 1])
            union_(label, idx, label_idx + 2 * W - 2);

        if ((hasBit(P, 13) && img[img_idx + 2 * W]) || (hasBit(P, 14) && img[img_idx + 2 * W + 1]))
            union_(label, idx, label_idx + 2 * W);

        if (hasBit(P, 15) && img[img_idx + 2 * W + 2])
            union_(label, idx, label_idx + 2 * W + 2);

        // Current planes
        if ((hasBit(P, 16) && img[idx - W - 1]) || (hasBit(P, 32) && img[idx + stepz - W - 1]))
            union_(label, idx, idx - 2 * W - 2);

        if ((hasBit(P, 17) && img[idx - W]) || (hasBit(P, 18) && img[idx - W + 1]) || \
            (hasBit(P, 33) && img[idx + stepz - W]) || (hasBit(P, 34) && img[idx + stepz - W + 1]))
            union_(label, idx, idx - 2 * W);

        if ((hasBit(P, 19) && img[idx - W + 2]) || (hasBit(P, 35) && img[idx + stepz - W + 2]))
            union_(label, idx, idx - 2 * W + 2);

        if ((hasBit(P, 20) && img[idx - 1]) || (hasBit(P, 24) && img[idx + W - 1]) || \
            (hasBit(P, 36) && img[idx + stepz - 1]) || (hasBit(P, 40) && img[idx + stepz + W - 1]))
            union_(label, idx, idx - 2);

    }

}

__global__ void compression(uint32_t* label, const uint32_t W, const uint32_t H, const uint32_t D)
{
    const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const uint32_t z = (blockIdx.z * blockDim.z + threadIdx.z) * 2;

    const uint32_t idx = z * W * H + y * W + x;

    if (x < W && y < H && z < D)
        find_n_compress(label, idx);
}


__global__ void final_labeling(uint32_t* label, uint8_t* last_cube_fg, const uint32_t W, const uint32_t H, const uint32_t D)
{
    const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const uint32_t z = (blockIdx.z * blockDim.z + threadIdx.z) * 2;

    const uint32_t idx = z * W * H + y * W + x;

    if (x >= W || y >= H || z >= D)
        return;

    int tmp;
    uint8_t fg;
    uint64_t buf;

    if (x + 1 < W) {
        buf = *reinterpret_cast<uint64_t*>(label + idx);
        tmp = (buf & (0xFFFFFFFF)) + 1;
        fg = (buf >> 32) & 0xFFFFFFFF;
    }
    else {
        tmp = label[idx] + 1;
        if (y + 1 < H)       fg = label[idx + W];
        else if (z + 1 < D)  fg = label[idx + W * H];
        else                 fg = *last_cube_fg;
    }

    if (x + 1 < W) {
        *reinterpret_cast<uint64_t*>(label + idx) =
            (static_cast<uint64_t>(((fg >> 1) & 1) * tmp) << 32) | (((fg >> 0) & 1) * tmp);

        if (y + 1 < H)
            *reinterpret_cast<uint64_t*>(label + idx + W) =
            (static_cast<uint64_t>(((fg >> 3) & 1) * tmp) << 32) | (((fg >> 2) & 1) * tmp);
        if (z + 1 < D) {
            *reinterpret_cast<uint64_t*>(label + idx + W * H) =
                (static_cast<uint64_t>(((fg >> 5) & 1) * tmp) << 32) | (((fg >> 4) & 1) * tmp);

            if (y + 1 < H)
                *reinterpret_cast<uint64_t*>(label + idx + W * H + W) =
                (static_cast<uint64_t>(((fg >> 7) & 1) * tmp) << 32) | (((fg >> 6) & 1) * tmp);
        }
    }
    else {
        label[idx] = ((fg >> 0) & 1) * tmp;
        if (y + 1 < H)
            label[idx + (W)] = ((fg >> 2) & 1) * tmp;

        if (z + 1 < D) {
            label[idx + W * H] = ((fg >> 4) & 1) * tmp;
            if (y + 1 < H)
                label[idx + W * H + W] = ((fg >> 6) & 1) * tmp;
        }
    }
} // final_labeling

template <class T>
__global__ void kernel_threshold(const T* image, T _thresholdMin, const T _thresholdMax, uint8_t* thresholdedImage, uint32_t size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    T value = image[tid];
    thresholdedImage[tid] = value >= _thresholdMin && value <= _thresholdMax ? 255 : 0;
}

template <class T>
__global__ void kernel_threshold32(const T* image, T _thresholdMin, const T _thresholdMax, int32_t* thresholdedImage, uint32_t size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    T value = image[tid];
    thresholdedImage[tid] = value > _thresholdMin && value < _thresholdMax ? 255 : 0;
}

#define BLOCK_X 8
#define BLOCK_Y 4
#define BLOCK_Z 4

void connectedComponnetsLabelingBinary(uint8_t* const _pixels, const size_t _nbValues, const uint32_t W, const uint32_t H, const uint32_t D, uint32_t* _labels) {
    dim3 grid = dim3(((W + 1) / 2 + BLOCK_X - 1) / BLOCK_X, ((H + 1) / 2 + BLOCK_Y - 1) / BLOCK_Y, ((D + 1) / 2 + BLOCK_Z - 1) / BLOCK_Z);
    dim3 block = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);
    //cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    uint8_t* last_cube_fg = NULL;
    bool allocated_last_cude_fg_ = false;
    if ((W % 2 == 1) && (H % 2 == 1) && (D % 2 == 1)) {
        if (W > 1 && H > 1)
            last_cube_fg = reinterpret_cast<uint8_t*>(
                _labels + (D - 1) * W * H + (H - 2) * W
                ) + W - 2;
        else if (W > 1 && D > 1)
            last_cube_fg = reinterpret_cast<uint8_t*>(
                _labels + (D - 2) * W * H + (H - 1) * W
                ) + W - 2;
        else if (H > 1 && D > 1)
            last_cube_fg = reinterpret_cast<uint8_t*>(
                _labels + (D - 2) * W * H + (H - 2) * W
                ) + W - 1;
        else {
            cudaMalloc(&last_cube_fg, sizeof(uint8_t));
            allocated_last_cude_fg_ = true;
        }
    }

    init_labeling << <grid, block >> > (_labels, W, H, D);
    merge << <grid, block >> > (_pixels, _labels, last_cube_fg, W, H, D);
    compression << <grid, block >> > (_labels, W, H, D);
    final_labeling << <grid, block >> > (_labels, last_cube_fg, W, H, D);

    if (allocated_last_cude_fg_)
        cudaFree(last_cube_fg);
}

poca::core::ImageInterface* connectedComponnetsLabeling3dUI8(const uint8_t* _pixels, const size_t _nbValues, const uint8_t _thresholdMin, const uint8_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d) {

    const uint32_t D = _d;
    const uint32_t H = _h;
    const uint32_t W = _w;

    /*AT_ASSERTM((D % 2) == 0, "shape must be a even number");
    AT_ASSERTM((H % 2) == 0, "shape must be a even number");
    AT_ASSERTM((W % 2) == 0, "shape must be a even number");*/

    uint8_t* thresholImage;
    cudaMalloc((void**)&thresholImage, _nbValues * sizeof(uint8_t));
    dim3 block(32);
    dim3 grid((_nbValues + block.x - 1) / block.x);
    JustGPUBuffer<uint8_t> imageToThreshold(_pixels, _nbValues);
    kernel_threshold << <grid, block >> > (imageToThreshold.gpu_data, _thresholdMin, _thresholdMax, thresholImage, _nbValues);


    // label must be uint32_t
   // auto label_options = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    //torch::Tensor label = torch::zeros({ D, H, W }, label_options);
    thrust::device_vector<uint32_t> d_labels(_nbValues);
    connectedComponnetsLabelingBinary(thresholImage, _nbValues, _w, _h, _d, thrust::raw_pointer_cast(d_labels.data()));
    cudaFree(thresholImage);
    relabel_kernel_gpu<uint32_t>(d_labels);
    auto maxLabel = *thrust::max_element(d_labels.begin(), d_labels.end());
    std::cout << "Max lbl = " << maxLabel << std::endl;
    poca::core::ImageInterface* image = NULL;
    if (maxLabel < std::numeric_limits<uint8_t>::max()) {
        image = convertAndCreateLabelImage<uint32_t, uint8_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT8);
    }
    else if (maxLabel < std::numeric_limits<uint16_t>::max()) {
        image = convertAndCreateLabelImage<uint32_t, uint16_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT16);
    }
    else {
        image = convertAndCreateLabelImage<uint32_t, uint32_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT32);
    }

    return image;
}

poca::core::ImageInterface* connectedComponnetsLabeling3dUI16(const uint16_t* _pixels, const size_t _nbValues, const uint16_t _thresholdMin, const uint16_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d) {

    const uint32_t D = _d;
    const uint32_t H = _h;
    const uint32_t W = _w;

    /*AT_ASSERTM((D % 2) == 0, "shape must be a even number");
    AT_ASSERTM((H % 2) == 0, "shape must be a even number");
    AT_ASSERTM((W % 2) == 0, "shape must be a even number");*/

    uint8_t* thresholImage;
    cudaMalloc((void**)&thresholImage, _nbValues * sizeof(uint8_t));
    dim3 block(32);
    dim3 grid((_nbValues + block.x - 1) / block.x);
    JustGPUBuffer<uint16_t> imageToThreshold(_pixels, _nbValues);
    kernel_threshold << <grid, block >> > (imageToThreshold.gpu_data, _thresholdMin, _thresholdMax, thresholImage, _nbValues);

    // label must be uint32_t
   // auto label_options = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    //torch::Tensor label = torch::zeros({ D, H, W }, label_options);

    thrust::device_vector<uint32_t> d_labels(_nbValues);
    connectedComponnetsLabelingBinary(thresholImage, _nbValues, _w, _h, _d, thrust::raw_pointer_cast(d_labels.data()));
    cudaFree(thresholImage);
    relabel_kernel_gpu<uint32_t>(d_labels);
    auto maxLabel = *thrust::max_element(d_labels.begin(), d_labels.end());
    std::cout << "Max lbl = " << maxLabel << std::endl;
    poca::core::ImageInterface* image = NULL;
    if (maxLabel < std::numeric_limits<uint8_t>::max()) {
        image = convertAndCreateLabelImage<uint32_t, uint8_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT8);
    }
    else if (maxLabel < std::numeric_limits<uint16_t>::max()) {
        image = convertAndCreateLabelImage<uint32_t, uint16_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT16);
    }
    else {
        image = convertAndCreateLabelImage<uint32_t, uint32_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT32);
    }
    return image;
}

poca::core::ImageInterface* connectedComponnetsLabeling3dUI32(const uint32_t* _pixels, const size_t _nbValues, const uint32_t _thresholdMin, const uint32_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d) {

    const uint32_t D = _d;
    const uint32_t H = _h;
    const uint32_t W = _w;

    uint8_t* thresholImage;
    cudaMalloc((void**)&thresholImage, _nbValues * sizeof(uint8_t));
    dim3 block(32);
    dim3 grid((_nbValues + block.x - 1) / block.x);
    JustGPUBuffer<uint32_t> imageToThreshold(_pixels, _nbValues);
    kernel_threshold << <grid, block >> > (imageToThreshold.gpu_data, _thresholdMin, _thresholdMax, thresholImage, _nbValues);

    thrust::device_vector<uint32_t> d_labels(_nbValues);
    connectedComponnetsLabelingBinary(thresholImage, _nbValues, _w, _h, _d, thrust::raw_pointer_cast(d_labels.data()));
    cudaFree(thresholImage);
    relabel_kernel_gpu<uint32_t>(d_labels);
    auto maxLabel = *thrust::max_element(d_labels.begin(), d_labels.end());
    std::cout << "Max lbl = " << maxLabel << std::endl;
    poca::core::ImageInterface* image = NULL;
    if (maxLabel < std::numeric_limits<uint8_t>::max()) {
        image = convertAndCreateLabelImage<uint32_t, uint8_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT8);
    }
    else if (maxLabel < std::numeric_limits<uint16_t>::max()) {
        image = convertAndCreateLabelImage<uint32_t, uint16_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT16);
    }
    else {
        image = convertAndCreateLabelImage<uint32_t, uint32_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT32);
    }
    return image;
}

poca::core::ImageInterface* connectedComponnetsLabeling3dI32(const int32_t* _pixels, const size_t _nbValues, const int32_t _thresholdMin, const int32_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d) {

    const uint32_t D = _d;
    const uint32_t H = _h;
    const uint32_t W = _w;

    /*AT_ASSERTM((D % 2) == 0, "shape must be a even number");
    AT_ASSERTM((H % 2) == 0, "shape must be a even number");
    AT_ASSERTM((W % 2) == 0, "shape must be a even number");*/

    uint8_t* thresholImage;
    cudaMalloc((void**)&thresholImage, _nbValues * sizeof(uint8_t));
    dim3 block(32);
    dim3 grid((_nbValues + block.x - 1) / block.x);
    JustGPUBuffer<int32_t> imageToThreshold(_pixels, _nbValues);
    kernel_threshold << <grid, block >> > (imageToThreshold.gpu_data, _thresholdMin, _thresholdMax, thresholImage, _nbValues);

    thrust::device_vector<uint32_t> d_labels(_nbValues);
    connectedComponnetsLabelingBinary(thresholImage, _nbValues, _w, _h, _d, thrust::raw_pointer_cast(d_labels.data()));
    cudaFree(thresholImage);
    relabel_kernel_gpu<uint32_t>(d_labels);
    auto maxLabel = *thrust::max_element(d_labels.begin(), d_labels.end());
    std::cout << "Max lbl = " << maxLabel << std::endl;
    poca::core::ImageInterface* image = NULL;
    if (maxLabel < std::numeric_limits<uint8_t>::max()) {
        image = convertAndCreateLabelImage<uint32_t, uint8_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT8);
    }
    else if (maxLabel < std::numeric_limits<uint16_t>::max()) {
        image = convertAndCreateLabelImage<uint32_t, uint16_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT16);
    }
    else {
        image = convertAndCreateLabelImage<uint32_t, uint32_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT32);
    }
    return image;
}

poca::core::ImageInterface* connectedComponnetsLabeling3dF(const float* _pixels, const size_t _nbValues, const float _thresholdMin, const float _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d) {

    const uint32_t D = _d;
    const uint32_t H = _h;
    const uint32_t W = _w;

    uint8_t* thresholImage;
    cudaMalloc((void**)&thresholImage, _nbValues * sizeof(uint8_t));
    dim3 block(32);
    dim3 grid((_nbValues + block.x - 1) / block.x);
    JustGPUBuffer<float> imageToThreshold(_pixels, _nbValues);
    kernel_threshold << <grid, block >> > (imageToThreshold.gpu_data, _thresholdMin, _thresholdMax, thresholImage, _nbValues);

    thrust::device_vector<uint32_t> d_labels(_nbValues);
    connectedComponnetsLabelingBinary(thresholImage, _nbValues, _w, _h, _d, thrust::raw_pointer_cast(d_labels.data()));
    cudaFree(thresholImage);
    relabel_kernel_gpu<uint32_t>(d_labels);
    uint32_t maxLabel = *thrust::max_element(d_labels.begin(), d_labels.end());
    std::cout << "Max lbl = " << maxLabel << std::endl;
    poca::core::ImageInterface* image = NULL;
    if (maxLabel < std::numeric_limits<uint8_t>::max()) {
        image = convertAndCreateLabelImage<uint32_t, uint8_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT8);
    }
    else if (maxLabel < std::numeric_limits<uint16_t>::max()) {
        image = convertAndCreateLabelImage<uint32_t, uint16_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT16);
    }
    else {
        image = convertAndCreateLabelImage<uint32_t, uint32_t>(d_labels, _w, _h, _d);
        image->setType(poca::core::UINT32);
    }
    return image;
}

/*__global__ void kernel_relabel(const int32_t* origLabels, int32_t* relabels, int32_t* uniques, int32_t* iota, uint32_t sizeLabels, uint32_t sizeUniques)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= sizeLabels) return;
    int32_t value = origLabels[tid];
    if (value == 0) {
        relabels[tid] = 0;
        return;
    }
    //thresholdedImage[tid] = value > threshold ? 255 : 0;
}*/

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
    
    //relabel_kernel(_labels, _relabels);
    //count_occurences_label_kernel(_labels, _relabels);
    //thrust::device_vector<int32_t> sortLabels(thrust::device_pointer_cast(_labels), thrust::device_pointer_cast(_labels) + _nbValues), values(_nbValues, 1);
    //thrust::sort(thrust::device, sortLabels.begin(), sortLabels.end());


    /*size_t nbValues = _labels.size();
    thrust::device_vector<int32_t> copiedImage(_labels);
    GPUBuffer<int32_t> out_relabel(_relabels);
    cudaMemcpy(out_relabel.gpu_data, thrust::raw_pointer_cast(copiedImage.data()), _labels.size() * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    
    thrust::sort(thrust::device, copiedImage.begin(), copiedImage.end());
    thrust::device_vector<int32_t>::iterator new_end = thrust::unique(thrust::device, copiedImage.begin(), copiedImage.end());
    thrust::device_vector<int32_t> oldLabels(copiedImage.begin(), new_end);
    thrust::copy(oldLabels.begin(), oldLabels.end(), std::ostream_iterator<int>(std::cout, ","));
    int nbUniqueLabels = thrust::distance(copiedImage.begin(), new_end);
    thrust::device_vector<int32_t> newLabels(nbUniqueLabels);
    thrust::sequence(thrust::device, newLabels.begin(), newLabels.end(), 0, 1);
    thrust::copy(newLabels.begin(), newLabels.end(), std::ostream_iterator<int>(std::cout, ","));


    thrust::copy(thrust::make_permutation_iterator(oldLabels.begin(), newLabels.begin()), thrust::make_permutation_iterator(oldLabels.begin(), newLabels.end()), thrust::device_pointer_cast(out_relabel.gpu_data));
    out_relabel.gpu2cpu();*/
    
    /*std::vector <int32_t> copied(_labels);
    GPUBuffer<int32_t> origLabels(copied);
    thrust::sort(thrust::device_pointer_cast(origLabels.gpu_data), thrust::device_pointer_cast(origLabels.gpu_data) + nbValues);
    int32_t* new_end = thrust::unique(origLabels.gpu_data, origLabels.gpu_data + nbValues);
    auto nbUniqueLabels = new_end - origLabels.gpu_data;
    std::vector <int32_t> iota(nbUniqueLabels);
    GPUBuffer<int32_t> iotaLabels(iota);
    thrust::sequence(thrust::device, iotaLabels.gpu_data, iotaLabels.gpu_data + nbUniqueLabels, 1, 1);

    dim3 block(32);
    dim3 grid((nbValues + block.x - 1) / block.x);
    JustGPUBuffer<uint16_t> imageToThreshold(_pixels, _nbValues);
    kernel_threshold << <grid, block >> > (imageToThreshold.gpu_data, _threshold, thresholImage, _nbValues);*/
}

void computeFeaturesLabelImage(poca::core::ImageInterface* _image)
{
    thrust::device_vector<float> d_labels, d_counts;
    switch (_image->type())
    {
    case poca::core::UINT8:
    {
        poca::core::Image<uint8_t>* casted = static_cast <poca::core::Image<uint8_t>*>(_image);
        thrust::device_vector<uint8_t> d_pixels(casted->pixels());
        count_occurences_label_kernel_gpu<uint8_t, float>(d_pixels, d_labels, d_counts);
    }
    break;
    case poca::core::UINT16:
    {
        poca::core::Image<uint16_t>* casted = static_cast <poca::core::Image<uint16_t>*>(_image);
        thrust::device_vector<uint16_t> d_pixels(casted->pixels());
        count_occurences_label_kernel_gpu<uint16_t, float>(d_pixels, d_labels, d_counts);
    }
    break;
    case poca::core::UINT32:
    {
        poca::core::Image<uint32_t>* casted = static_cast <poca::core::Image<uint32_t>*>(_image);
        thrust::device_vector<uint32_t> d_pixels(casted->pixels());
        count_occurences_label_kernel_gpu<uint32_t, float>(d_pixels, d_labels, d_counts);
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

#endif