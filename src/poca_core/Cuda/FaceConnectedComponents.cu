// watershed.cu
#include <cuda_runtime.h>
#include <cuda/std/limits>
#include <thrust\device_vector.h>
#include <thrust\unique.h>
#include <thrust/binary_search.h>

#include <iostream>
#include <cmath>
#include <queue>
#include <vector>

#include <Cuda/ConnectedComponents.h>

#define IDX2(i, j, width) ((i) * (width) + (j))
#define IDX3(i, j, k, width, height) (((k) * (height) + (i)) * (width) + (j))
#define WATERSHED_UNLABELED 0xFFFFFFFFu

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8
#define BLOCKDIM_Z 8

//--------------------------------------------------------------
// Watershed Kernels (GPU Iterative)
//--------------------------------------------------------------
template <class T>
__global__ void face_cc_kernel_2d_init(const uint8_t* binary, T* labels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = IDX2(y, x, width);

    if (binary[idx] == 0) return;

    labels[idx] = T(idx);
}

template <class T>
__global__ void face_cc_kernel_2d_iteration(T* cclabels, uint32_t* changed, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = IDX2(y, x, width);
    T curVal = cclabels[idx];
    if (curVal == 0) return;

    T minLabel = ::cuda::std::numeric_limits<T>::max();
    for (auto n = 0; n < 4; n++) {
        int x2 = x + NEIGHBOR_OFFSET_2D_X[n], y2 = y + NEIGHBOR_OFFSET_2D_Y[n];
        if (x2 >= width || y2 >= height) continue;
        int idxNeigh = IDX2(y2, x2, width);
        T val = cclabels[idxNeigh];
        if (val != 0) {
            minLabel = min(minLabel, val);
        }
    }
    if (minLabel < curVal) {
        cclabels[idx] = minLabel;
        atomicAdd(changed, 1);
    }
}

template <class T>
__global__ void face_cc_kernel_3d_init(const uint8_t* binary, T* labels, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    int idx = IDX3(y, x, z, width, height);

    if (binary[idx] == 0) return;

    labels[idx] = T(idx);
}

template <class T>
__global__ void face_cc_kernel_3d_iteration(T* cclabels, uint32_t* changed, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    int idx = IDX3(y, x, z, width, height);

    T curVal = cclabels[idx];
    if (curVal == 0) return;

    T minLabel = ::cuda::std::numeric_limits<T>::max();
    for (auto n = 0; n < 6; n++) {
        int x2 = x + NEIGHBOR_OFFSET_3D_X[n], y2 = y + NEIGHBOR_OFFSET_3D_Y[n], z2 = z + NEIGHBOR_OFFSET_3D_Z[n];
        if (x2 >= width || y2 >= height || z2 >= depth) continue;
        int idxNeigh = IDX3(y2, x2, z2, width, height);
        T val = cclabels[idxNeigh];
        if (val != 0) {
            minLabel = min(minLabel, val);
        }
    }
    if (minLabel < curVal) {
        cclabels[idx] = minLabel;
        atomicAdd(changed, 1);
    }
}

void face_connected_component(thrust::device_vector<uint8_t>& d_binary, thrust::device_vector<uint32_t>& d_labels, int width, int height, int depth)
{
    auto start = std::chrono::high_resolution_clock::now();
    dim3 threads = depth == 1 ? dim3(BLOCKDIM_X, BLOCKDIM_Y) : dim3(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
    dim3 grid = depth == 1 ? dim3((unsigned int)ceil((float)width / (float)BLOCKDIM_X), (unsigned int)ceil((float)height / (float)BLOCKDIM_Y)) : dim3((unsigned int)ceil((float)width / (float)BLOCKDIM_X), (unsigned int)ceil((float)height / (float)BLOCKDIM_Y), (unsigned int)ceil((float)depth / (float)BLOCKDIM_Z));
    thrust::device_vector<uint32_t> d_changed(1);

    int iteration = 1;
    uint32_t changed = 1;
    //auto start2 = std::chrono::high_resolution_clock::now();
    if (depth == 1)
        face_cc_kernel_2d_init<uint32_t> << < grid, threads >> > (thrust::raw_pointer_cast(d_binary.data()), thrust::raw_pointer_cast(d_labels.data()), width, height);
    else
        face_cc_kernel_3d_init<uint32_t> << < grid, threads >> > (thrust::raw_pointer_cast(d_binary.data()), thrust::raw_pointer_cast(d_labels.data()), width, height, depth);
    cudaDeviceSynchronize();
    while (changed != 0) {
        thrust::fill(d_changed.begin(), d_changed.end(), 0);
        if (depth == 1)
            face_cc_kernel_2d_iteration<uint32_t> << < grid, threads >> > (thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_changed.data()), width, height);
        else
            face_cc_kernel_3d_iteration<uint32_t> << < grid, threads >> > (thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_changed.data()), width, height, depth);
        cudaDeviceSynchronize();
        cudaMemcpy(&changed, thrust::raw_pointer_cast(d_changed.data()), 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        //auto duration2 = std::chrono::high_resolution_clock::now() - start2;
        //long long ms = std::chrono::duration_cast<std::chrono::microseconds>(duration2).count();
        //float s = std::chrono::duration_cast<std::chrono::seconds>(duration2).count();
        //printf("face-connected component, iteration %u, sum %u, took %f seconds (%lld microseconds)\n", iteration, changed, s, ms);
        iteration++;
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
    }

    uint32_t nbLabels = relabel_kernel_uint32t_gpu(d_labels);
    auto duration2 = std::chrono::high_resolution_clock::now() - start;
    long long ms = std::chrono::duration_cast<std::chrono::microseconds>(duration2).count();
    float s = std::chrono::duration_cast<std::chrono::seconds>(duration2).count();
    printf("total time face connected components took %f seconds (%lld microseconds), with number of iterations: %u\n", s, ms, iteration);
}

//--------------------------------------------------------------
// Watershed Algorithm Host Code
//--------------------------------------------------------------
void run_face_connected_component_pipeline(uint8_t* binary, uint32_t* output_labels, int width, int height, int depth) {
    size_t numel = width * height * depth;

    thrust::device_vector<uint8_t> d_binary(binary, binary + numel);
    thrust::device_vector<uint32_t> d_labels(numel), d_changed(1);

    face_connected_component(d_binary, d_labels, width, height, depth);

    cudaMemcpy(output_labels, thrust::raw_pointer_cast(d_labels.data()), numel * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

template __global__ void face_cc_kernel_2d_iteration(uint8_t* cclabels, uint32_t* changed, int width, int height);
template __global__ void face_cc_kernel_2d_iteration(uint16_t* cclabels, uint32_t* changed, int width, int height);
template __global__ void face_cc_kernel_2d_iteration(uint32_t* cclabels, uint32_t* changed, int width, int height);
template __global__ void face_cc_kernel_2d_iteration(float* cclabels, uint32_t* changed, int width, int height);

template __global__ void face_cc_kernel_3d_iteration(uint8_t* cclabels, uint32_t* changed, int width, int height, int depth);
template __global__ void face_cc_kernel_3d_iteration(uint16_t* cclabels, uint32_t* changed, int width, int height, int depth);
template __global__ void face_cc_kernel_3d_iteration(uint32_t* cclabels, uint32_t* changed, int width, int height, int depth);
template __global__ void face_cc_kernel_3d_iteration(float* cclabels, uint32_t* changed, int width, int height, int depth);
