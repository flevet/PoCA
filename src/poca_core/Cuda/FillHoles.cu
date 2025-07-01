// fill_image_holes.cu
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <cuda_runtime.h>
#include <queue>
#include <vector>
#include <iostream>

#include "BasicOperationsImage.h"

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

template <class T>
void run_fill_holes_2(std::vector<T>& _pixels, const uint32_t _width, const uint32_t _height) {
    thrust::device_vector<T> d_image(_pixels), d_padded_image;

    pad(d_image, d_padded_image, _width, _height, 1, 1);

    fill_all_holes(d_image, _width, _height);

    cudaMemcpy(_pixels.data(), thrust::raw_pointer_cast(d_image.data()), _width * _height * sizeof(T), cudaMemcpyDeviceToHost);
}


template void fill_all_holes(thrust::device_vector<uint8_t>& d_labels, uint32_t width, uint32_t height);
template void fill_all_holes(thrust::device_vector<uint16_t>& d_labels, uint32_t width, uint32_t height);
template void fill_all_holes(thrust::device_vector<uint32_t>& d_labels, uint32_t width, uint32_t height);
template void fill_all_holes(thrust::device_vector<float>& d_labels, uint32_t width, uint32_t height);

template void run_fill_holes(std::vector<uint8_t>& _pixels, const uint32_t _width, const uint32_t _height);
template void run_fill_holes(std::vector<uint16_t>& _pixels, const uint32_t _width, const uint32_t _height);
template void run_fill_holes(std::vector<uint32_t>& _pixels, const uint32_t _width, const uint32_t _height);
template void run_fill_holes(std::vector<float>& _pixels, const uint32_t _width, const uint32_t _height);