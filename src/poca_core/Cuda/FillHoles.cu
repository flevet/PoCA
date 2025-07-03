// fill_image_holes.cu
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <cuda_runtime.h>
#include <cuda/std/limits>
#include <queue>
#include <vector>
#include <iostream>

#include <tinytiffwriter.h>

#include "BasicOperationsImage.h"
#include "ConnectedComponents.h"

#define IDX(x, y, width) ((y) * (width) + (x))

template <class T>
__global__ void init_and_invert_mask_kernel(const T* labels, T* mask, uint32_t label, uint32_t size) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;
    mask[idx] = (labels[idx] == label) ? 0 : 1; // Inverted: 1 where not label
}

template <class T>
__global__ void flood_fill_kernel(T* mask, uint32_t width, uint32_t height, bool* changed) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint32_t idx = IDX(x, y, width);
    if (mask[idx] != 2) return; // Border-connected pixels are marked as 2

    int dirs[4][2] = { {1,0},{-1,0},{0,1},{0,-1} };
    for (int i = 0; i < 4; ++i) {
        int nx = x + dirs[i][0];
        int ny = y + dirs[i][1];
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        uint32_t nidx = IDX(nx, ny, width);
        if (mask[nidx] == 1) {
            mask[nidx] = 2;
            *changed = true;
        }
    }
}

template <class T>
__global__ void fill_holes_kernel(T* labels, T* mask, uint32_t label, uint32_t size) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;
    if (mask[idx] == 1) {
        labels[idx] = label;
    }
}

template <class T>
void fill_holes_for_label(thrust::device_vector<T>& d_labels, uint32_t label, uint32_t width, uint32_t height) {
    uint32_t size = width * height;
    thrust::device_vector<T> d_mask(size);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    init_and_invert_mask_kernel << <blocks, threads >> > (thrust::raw_pointer_cast(d_labels.data()),
        thrust::raw_pointer_cast(d_mask.data()),
        label, size);

    // Mark borders with 2 where the pixel is 1 (inverted mask)
    for (int x = 0; x < width; ++x) {
        if (d_mask[IDX(x, 0, width)] == 1) d_mask[IDX(x, 0, width)] = 2;
        if (d_mask[IDX(x, height - 1, width)] == 1) d_mask[IDX(x, height - 1, width)] = 2;
    }
    for (int y = 0; y < height; ++y) {
        if (d_mask[IDX(0, y, width)] == 1) d_mask[IDX(0, y, width)] = 2;
        if (d_mask[IDX(width - 1, y, width)] == 1) d_mask[IDX(width - 1, y, width)] = 2;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

    bool h_changed;
    bool* d_changed;
    cudaMalloc(&d_changed, sizeof(bool));

    do {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
        flood_fill_kernel << <numBlocks, threadsPerBlock >> > (thrust::raw_pointer_cast(d_mask.data()), width, height, d_changed);
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (h_changed);

    cudaFree(d_changed);

    fill_holes_kernel << <blocks, threads >> > (thrust::raw_pointer_cast(d_labels.data()),
        thrust::raw_pointer_cast(d_mask.data()), label, size);
}

template <class T>
void fill_all_holes(thrust::device_vector<T>& d_labels, uint32_t width, uint32_t height) {
    thrust::device_vector<T> d_copy = d_labels;
    thrust::sort(d_copy.begin(), d_copy.end());
    auto end = thrust::unique(d_copy.begin(), d_copy.end());
    thrust::host_vector<int> h_labels(d_copy.begin(), end);

    for (int label : h_labels) {
        if (label == 0) continue; // skip background
        fill_holes_for_label(d_labels, label, width, height);
    }
}

template <class T>
void run_fill_holes(std::vector<T>& _pixels, const uint32_t _width, const uint32_t _height) {
    thrust::device_vector<T> d_image(_pixels);

    fill_all_holes(d_image, _width, _height);

    cudaMemcpy(_pixels.data(), thrust::raw_pointer_cast(d_image.data()), _width * _height * sizeof(T), cudaMemcpyDeviceToHost);
}

template <class T, class M>
__global__ void copy_transfer(T* array1, M* array2, uint32_t size) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;
    array2[idx] = M(array1[idx]);
}

template <class T>
void run_fill_holes_2(std::vector<T>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth) {
    uint32_t numel = _width * _height * _depth;
    int threadsSingle = 256;
    int blocks = (numel + threadsSingle - 1) / threadsSingle;
    uint32_t padVal = 1;
    uint32_t wpad = _width + 2 * padVal, hpad = _height + 2 * padVal, dpad = _depth == 1 ? 1 : _depth + 2 * padVal;

    thrust::device_vector<T> d_image(_pixels);
    thrust::device_vector<uint8_t> d_bimage(numel), d_padded_image;
    thrust::device_vector<uint32_t> d_changed(1), d_labels(numel), d_holes(numel);

    binarize << <blocks, threadsSingle >> > (thrust::raw_pointer_cast(d_image.data()), thrust::raw_pointer_cast(d_bimage.data()), numel);

    pad<uint8_t>(d_bimage, d_padded_image, _width, _height, _depth, padVal);

    dim3 threads = _depth == 1 ? dim3(8, 8) : dim3(8, 8, 8);
    dim3 grid = _depth == 1 ? dim3((unsigned int)ceil((float)wpad / (float)8), (unsigned int)ceil((float)hpad / (float)8)) : dim3((unsigned int)ceil((float)wpad / (float)8), (unsigned int)ceil((float)hpad / (float)8), (unsigned int)ceil((float)dpad / (float)8));
    dim3 gridOrig = _depth == 1 ? dim3((unsigned int)ceil((float)_width / (float)8), (unsigned int)ceil((float)_height / (float)8)) : dim3((unsigned int)ceil((float)_width / (float)8), (unsigned int)ceil((float)_height / (float)8), (unsigned int)ceil((float)_depth / (float)8));

    thrust::transform(thrust::device, d_padded_image.begin(), d_padded_image.end(), d_padded_image.begin(), [] __device__(uint8_t v) { return (v == 0) ? 255 : 0; });

    /* {
        std::string name("d:/test_binary.tif");
        TinyTIFFWriterSampleFormat sampleF;
        uint16_t bitPerSample;
        auto w = wpad, h = hpad;
        poca::core::ImageType type = poca::core::UINT8;
        switch (type)
        {
        case poca::core::UINT8:
            sampleF = TinyTIFFWriter_UInt;
            bitPerSample = 8;
            break;
        case poca::core::UINT16:
            sampleF = TinyTIFFWriter_UInt;
            bitPerSample = 16;
            break;
        case poca::core::UINT32:
            sampleF = TinyTIFFWriter_UInt;
            bitPerSample = 32;
            break;
        case poca::core::INT32:
            sampleF = TinyTIFFWriter_Int;
            bitPerSample = 32;
            break;
        case poca::core::FLOAT:
            sampleF = TinyTIFFWriter_Float;
            bitPerSample = 32;
            break;
        default:
            break;
        }
        uint32_t numel2 = w * h;
        int threadsSingle2 = 256;
        int blocks2 = (numel2 + threadsSingle2 - 1) / threadsSingle2;
        TinyTIFFWriterFile* tif = TinyTIFFWriter_open(name.c_str(), bitPerSample, sampleF, 1, w, h, TinyTIFFWriter_Greyscale);
        if (tif) {
            std::vector <uint8_t> pixImage(wpad * hpad);
            cudaMemcpy(pixImage.data(), thrust::raw_pointer_cast(d_padded_image.data()), w * h * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            const void* data = (void*)pixImage.data();
            TinyTIFFWriter_writeImage(tif, data);
            TinyTIFFWriter_close(tif);
            std::cout << "Image " << name << std::endl;
        }
    }*/

    face_connected_component(d_padded_image, d_labels, wpad, hpad, 1);

    //thrust::transform(thrust::device, d_labels.begin(), d_labels.end(), d_labels.begin(), [] __device__(uint32_t v) { return (v == 1) ? 0 : v; });

    relabel_kernel_uint32t_gpu(d_labels);

    {
        std::string name("d:/test.tif");
        TinyTIFFWriterSampleFormat sampleF;
        uint16_t bitPerSample;
        auto w = wpad, h = hpad;
        poca::core::ImageType type = poca::core::UINT16;
        switch (type)
        {
        case poca::core::UINT8:
            sampleF = TinyTIFFWriter_UInt;
            bitPerSample = 8;
            break;
        case poca::core::UINT16:
            sampleF = TinyTIFFWriter_UInt;
            bitPerSample = 16;
            break;
        case poca::core::UINT32:
            sampleF = TinyTIFFWriter_UInt;
            bitPerSample = 32;
            break;
        case poca::core::INT32:
            sampleF = TinyTIFFWriter_Int;
            bitPerSample = 32;
            break;
        case poca::core::FLOAT:
            sampleF = TinyTIFFWriter_Float;
            bitPerSample = 32;
            break;
        default:
            break;
        }
        uint32_t numel2 = w * h;
        int threadsSingle2 = 256;
        int blocks2 = (numel2 + threadsSingle2 - 1) / threadsSingle2;
        TinyTIFFWriterFile* tif = TinyTIFFWriter_open(name.c_str(), bitPerSample, sampleF, 1, w, h, TinyTIFFWriter_Greyscale);
        if (tif) {
            std::vector <uint16_t> pixImage(wpad * hpad);
            thrust::device_vector<uint16_t> dtmp(numel2);
            copy_transfer<uint32_t, uint16_t> << <blocks2, threadsSingle2 >> > (thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(dtmp.data()), numel2);
            cudaMemcpy(pixImage.data(), thrust::raw_pointer_cast(dtmp.data()), w * h * sizeof(uint16_t), cudaMemcpyDeviceToHost);
            const void* data = (void*)pixImage.data();
            TinyTIFFWriter_writeImage(tif, data);
            TinyTIFFWriter_close(tif);
            std::cout << "Image " << name << std::endl;
        }
    }



    

    unpad<uint32_t>(d_labels, d_holes, wpad, hpad, dpad, padVal);

    /*thrust::transform(
        thrust::device,
        d_holes.begin(), d_holes.end(),
        d_labels.begin(),
        d_labels.begin(),
        [] __device__(uint32_t val2, T val1) -> T {
        if (val2 > 0) {
            return static_cast<T>(::cuda::std::numeric_limits<uint32_t>::max());
        }
        else {
            return val1 + val2;
        }
    }
    );*/

    /*int iteration = 1;
    uint32_t changed = 1;
    auto start2 = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();
    while (changed != 0) {
        thrust::fill(d_changed.begin(), d_changed.end(), 0);
        if (_depth == 1)
            face_cc_kernel_2d_iteration << < grid, threads >> > (thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_changed.data()), _width, _height);
        else
            face_cc_kernel_3d_iteration << < grid, threads >> > (thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_changed.data()), _width, _height, _depth);
        cudaDeviceSynchronize();
        cudaMemcpy(&changed, thrust::raw_pointer_cast(d_changed.data()), 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        auto duration2 = std::chrono::high_resolution_clock::now() - start2;
        long long ms = std::chrono::duration_cast<std::chrono::microseconds>(duration2).count();
        float s = std::chrono::duration_cast<std::chrono::seconds>(duration2).count();
        printf("face-connected component, iteration %u, sum %u, took %f seconds (%lld microseconds)\n", iteration, changed, s, ms);
        iteration++;
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
    }*/

    /*thrust::transform(
        thrust::device,
        d_labels.begin(), d_labels.end(),
        d_image.begin(),
        [] __device__(uint32_t v) { return static_cast<T>(v); }
    );*/

    //unpad<uint8_t>(d_padded_image, d_bimage, wpad, hpad, dpad, padVal);
    copy_transfer<uint32_t, T> << <blocks, threadsSingle >> > (thrust::raw_pointer_cast(d_holes.data()), thrust::raw_pointer_cast(d_image.data()), numel);

    cudaMemcpy(_pixels.data(), thrust::raw_pointer_cast(d_image.data()), _width * _height * _depth * sizeof(T), cudaMemcpyDeviceToHost);
}

template <class T>
void run_fill_holes_3(const std::vector<T>& _pixels, std::vector <uint32_t>& _results, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint32_t _pad, uint32_t& wdest, uint32_t& hdest, uint32_t& ddest)
{
    uint32_t numel = _width * _height * _depth;
    int threadsSingle = 256;
    int blocks = (numel + threadsSingle - 1) / threadsSingle;
    uint32_t padVal = 1;

    thrust::device_vector<T> d_image(_pixels);
    thrust::device_vector<uint8_t> d_bimage(numel), d_padded_image;
    thrust::device_vector<uint32_t> d_changed(1), d_labels(numel), d_holes(numel);

    binarize << <blocks, threadsSingle >> > (thrust::raw_pointer_cast(d_image.data()), thrust::raw_pointer_cast(d_bimage.data()), numel);

    pad<uint8_t>(d_bimage, d_padded_image, _width, _height, _depth, padVal);

    wdest = _width + 2 * _pad;
    hdest = _height + 2 * _pad;
    ddest = _depth == 1 ? 1 : _depth + 2 * _pad;
}

template void fill_all_holes(thrust::device_vector<uint8_t>& d_labels, uint32_t width, uint32_t height);
template void fill_all_holes(thrust::device_vector<uint16_t>& d_labels, uint32_t width, uint32_t height);
template void fill_all_holes(thrust::device_vector<uint32_t>& d_labels, uint32_t width, uint32_t height);
template void fill_all_holes(thrust::device_vector<float>& d_labels, uint32_t width, uint32_t height);

template void run_fill_holes(std::vector<uint8_t>& _pixels, const uint32_t _width, const uint32_t _height);
template void run_fill_holes(std::vector<uint16_t>& _pixels, const uint32_t _width, const uint32_t _height);
template void run_fill_holes(std::vector<uint32_t>& _pixels, const uint32_t _width, const uint32_t _height);
template void run_fill_holes(std::vector<float>& _pixels, const uint32_t _width, const uint32_t _height);

template void run_fill_holes_2(std::vector<uint8_t>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth);
template void run_fill_holes_2(std::vector<uint16_t>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth);
template void run_fill_holes_2(std::vector<uint32_t>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth);
template void run_fill_holes_2(std::vector<float>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth);

template void run_fill_holes_3(const std::vector<uint8_t>& _pixels, std::vector <uint32_t>& _results, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint32_t _pad, uint32_t& wdest, uint32_t& hdest, uint32_t& ddest);
template void run_fill_holes_3(const std::vector<uint16_t>& _pixels, std::vector <uint32_t>& _results, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint32_t _pad, uint32_t& wdest, uint32_t& hdest, uint32_t& ddest);
template void run_fill_holes_3(const std::vector<uint32_t>& _pixels, std::vector <uint32_t>& _results, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint32_t _pad, uint32_t& wdest, uint32_t& hdest, uint32_t& ddest);
template void run_fill_holes_3(const std::vector<float>& _pixels, std::vector <uint32_t>& _results, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint32_t _pad, uint32_t& wdest, uint32_t& hdest, uint32_t& ddest);