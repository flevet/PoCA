// Face-connected components on CUDA.
#include <cuda_runtime.h>
#include <cuda/std/limits>
#include <thrust/device_vector.h>

#include <chrono>
#include <cmath>
#include <iostream>

#include <Cuda/ConnectedComponents.h>

#define IDX3(x, y, z, width, height) (((z) * (height) + (y)) * (width) + (x))

#define FACE_CC_BLOCK_X 8
#define FACE_CC_BLOCK_Y 8
#define FACE_CC_BLOCK_Z 4

template <class T>
__global__ void face_cc_kernel_2d_iteration(T* cclabels, uint32_t* changed, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;

	int idx = y * width + x;
	T curVal = cclabels[idx];
	if (curVal == 0)
		return;

	T minLabel = ::cuda::std::numeric_limits<T>::max();
	for (auto n = 0; n < 4; n++) {
		int x2 = x + NEIGHBOR_OFFSET_2D_X[n], y2 = y + NEIGHBOR_OFFSET_2D_Y[n];
		if (x2 >= 0 && y2 >= 0 && x2 < width && y2 < height) {
			int idxNeigh = y2 * width + x2;
			T val = cclabels[idxNeigh];
			if (val != 0)
				minLabel = min(minLabel, val);
		}
	}
	if (minLabel < curVal) {
		cclabels[idx] = minLabel;
		atomicAdd(changed, 1);
	}
}

template <class T>
__global__ void face_cc_kernel_3d_iteration(T* cclabels, uint32_t* changed, int width, int height, int depth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= width || y >= height || z >= depth)
		return;

	int idx = IDX3(x, y, z, width, height);
	T curVal = cclabels[idx];
	if (curVal == 0)
		return;

	T minLabel = ::cuda::std::numeric_limits<T>::max();
	for (auto n = 0; n < 6; n++) {
		int x2 = x + NEIGHBOR_OFFSET_3D_X[n], y2 = y + NEIGHBOR_OFFSET_3D_Y[n], z2 = z + NEIGHBOR_OFFSET_3D_Z[n];
		if (x2 >= 0 && y2 >= 0 && z2 >= 0 && x2 < width && y2 < height && z2 < depth) {
			int idxNeigh = IDX3(x2, y2, z2, width, height);
			T val = cclabels[idxNeigh];
			if (val != 0)
				minLabel = min(minLabel, val);
		}
	}
	if (minLabel < curVal) {
		cclabels[idx] = minLabel;
		atomicAdd(changed, 1);
	}
}

namespace {
	__device__ uint32_t findRoot(const uint32_t* _labels, uint32_t _value)
	{
		uint32_t value = _value;
		while (_labels[value - 1] != value)
			value = _labels[value - 1];
		return value;
	}

	__device__ uint32_t compressRoot(uint32_t* _labels, uint32_t _value)
	{
		uint32_t value = findRoot(_labels, _value);

		uint32_t cur = _value;
		while (_labels[cur - 1] != cur) {
			uint32_t next = _labels[cur - 1];
			_labels[cur - 1] = value;
			cur = next;
		}
		return value;
	}

	__device__ void unionRoots(uint32_t* _labels, uint32_t _a, uint32_t _b)
	{
		bool done;
		do {
			_a = findRoot(_labels, _a);
			_b = findRoot(_labels, _b);

			if (_a < _b) {
				uint32_t old = atomicMin(_labels + _b - 1, _a);
				done = old == _b;
				_b = old;
			}
			else if (_b < _a) {
				uint32_t old = atomicMin(_labels + _a - 1, _b);
				done = old == _a;
				_a = old;
			}
			else
				done = true;
		} while (!done);
	}

	__device__ void unionIfForeground(const uint8_t* _binary, uint32_t* _labels, uint32_t _idx, uint32_t _neighbor)
	{
		if (_binary[_idx] == 0 || _binary[_neighbor] == 0)
			return;
		unionRoots(_labels, _idx + 1, _neighbor + 1);
	}

	__global__ void initFaceLabels(const uint8_t* _binary, uint32_t* _labels, uint32_t _size)
	{
		const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= _size)
			return;
		_labels[idx] = _binary[idx] == 0 ? 0 : idx + 1;
	}

	__global__ void mergeFace2D(const uint8_t* _binary, uint32_t* _labels, int _width, int _height)
	{
		const int bx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
		const int by = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
		if (bx >= _width || by >= _height)
			return;

		for (int ly = 0; ly < 2; ly++) {
			const int y = by + ly;
			if (y >= _height)
				continue;
			for (int lx = 0; lx < 2; lx++) {
				const int x = bx + lx;
				if (x >= _width)
					continue;

				const uint32_t idx = y * _width + x;
				if (x > 0)
					unionIfForeground(_binary, _labels, idx, idx - 1);
				if (y > 0)
					unionIfForeground(_binary, _labels, idx, idx - _width);
			}
		}
	}

	__global__ void mergeFace3D(const uint8_t* _binary, uint32_t* _labels, int _width, int _height, int _depth)
	{
		const int bx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
		const int by = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
		const int bz = (blockIdx.z * blockDim.z + threadIdx.z) * 2;
		if (bx >= _width || by >= _height || bz >= _depth)
			return;

		const uint32_t plane = _width * _height;
		for (int lz = 0; lz < 2; lz++) {
			const int z = bz + lz;
			if (z >= _depth)
				continue;
			for (int ly = 0; ly < 2; ly++) {
				const int y = by + ly;
				if (y >= _height)
					continue;
				for (int lx = 0; lx < 2; lx++) {
					const int x = bx + lx;
					if (x >= _width)
						continue;

					const uint32_t idx = IDX3(x, y, z, _width, _height);
					if (x > 0)
						unionIfForeground(_binary, _labels, idx, idx - 1);
					if (y > 0)
						unionIfForeground(_binary, _labels, idx, idx - _width);
					if (z > 0)
						unionIfForeground(_binary, _labels, idx, idx - plane);
				}
			}
		}
	}

	__global__ void compressFaceLabels(uint32_t* _labels, uint32_t _size)
	{
		const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= _size || _labels[idx] == 0)
			return;
		_labels[idx] = compressRoot(_labels, _labels[idx]);
	}
}

void face_connected_component(thrust::device_vector<uint8_t>& d_binary, thrust::device_vector<uint32_t>& d_labels, int width, int height, int depth)
{
	poca::core::Engine* engine = poca::core::Engine::instance();
	auto start = std::chrono::high_resolution_clock::now();

	const uint32_t numel = width * height * depth;
	const dim3 initBlock(256);
	const dim3 initGrid((numel + initBlock.x - 1) / initBlock.x);
	initFaceLabels << <initGrid, initBlock >> > (thrust::raw_pointer_cast(d_binary.data()), thrust::raw_pointer_cast(d_labels.data()), numel);

	if (depth == 1) {
		const dim3 threads(FACE_CC_BLOCK_X, FACE_CC_BLOCK_Y);
		const dim3 bins((width + 1) / 2, (height + 1) / 2);
		const dim3 grid((bins.x + FACE_CC_BLOCK_X - 1) / FACE_CC_BLOCK_X, (bins.y + FACE_CC_BLOCK_Y - 1) / FACE_CC_BLOCK_Y);
		mergeFace2D << <grid, threads >> > (thrust::raw_pointer_cast(d_binary.data()), thrust::raw_pointer_cast(d_labels.data()), width, height);
	}
	else {
		const dim3 threads(FACE_CC_BLOCK_X, FACE_CC_BLOCK_Y, FACE_CC_BLOCK_Z);
		const dim3 bins((width + 1) / 2, (height + 1) / 2, (depth + 1) / 2);
		const dim3 grid(
			(bins.x + FACE_CC_BLOCK_X - 1) / FACE_CC_BLOCK_X,
			(bins.y + FACE_CC_BLOCK_Y - 1) / FACE_CC_BLOCK_Y,
			(bins.z + FACE_CC_BLOCK_Z - 1) / FACE_CC_BLOCK_Z);
		mergeFace3D << <grid, threads >> > (thrust::raw_pointer_cast(d_binary.data()), thrust::raw_pointer_cast(d_labels.data()), width, height, depth);
	}

	compressFaceLabels << <initGrid, initBlock >> > (thrust::raw_pointer_cast(d_labels.data()), numel);
	const uint32_t nbLabels = relabel_kernel_uint32t_gpu(d_labels);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;

	if (engine->verbose()) {
		auto duration = std::chrono::high_resolution_clock::now() - start;
		long long us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
		float s = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
		printf("face connected components took %f seconds (%lld microseconds), labels: %u\n", s, us, nbLabels);
	}
}

template <class T>
void run_face_connected_component_pipeline(T* binary, uint32_t* output_labels, int width, int height, int depth)
{
	size_t numel = width * height * depth;
	thrust::device_vector<uint8_t> d_binary(binary, binary + numel);
	thrust::device_vector<uint32_t> d_labels(numel);

	face_connected_component(d_binary, d_labels, width, height, depth);
	cudaMemcpy(output_labels, thrust::raw_pointer_cast(d_labels.data()), numel * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

template void run_face_connected_component_pipeline(uint8_t* binary, uint32_t* output_labels, int width, int height, int depth);
template void run_face_connected_component_pipeline(uint16_t* binary, uint32_t* output_labels, int width, int height, int depth);
template void run_face_connected_component_pipeline(uint32_t* binary, uint32_t* output_labels, int width, int height, int depth);
template void run_face_connected_component_pipeline(float* binary, uint32_t* output_labels, int width, int height, int depth);

template __global__ void face_cc_kernel_2d_iteration(uint8_t* cclabels, uint32_t* changed, int width, int height);
template __global__ void face_cc_kernel_2d_iteration(uint16_t* cclabels, uint32_t* changed, int width, int height);
template __global__ void face_cc_kernel_2d_iteration(uint32_t* cclabels, uint32_t* changed, int width, int height);
template __global__ void face_cc_kernel_2d_iteration(float* cclabels, uint32_t* changed, int width, int height);

template __global__ void face_cc_kernel_3d_iteration(uint8_t* cclabels, uint32_t* changed, int width, int height, int depth);
template __global__ void face_cc_kernel_3d_iteration(uint16_t* cclabels, uint32_t* changed, int width, int height, int depth);
template __global__ void face_cc_kernel_3d_iteration(uint32_t* cclabels, uint32_t* changed, int width, int height, int depth);
template __global__ void face_cc_kernel_3d_iteration(float* cclabels, uint32_t* changed, int width, int height, int depth);
