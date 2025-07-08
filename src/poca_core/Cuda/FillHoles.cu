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

template <class T, class M>
__global__ void copy_transfer(T* array1, M* array2, uint32_t size) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;
    array2[idx] = M(array1[idx]);
}

template <class T>
void identify_holes(const thrust::device_vector <T>& _pixels, thrust::device_vector <T>& _holes, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    uint32_t padVal = 1;
    uint32_t wpad = _width + 2 * padVal, hpad = _height + 2 * padVal, dpad = _depth == 1 ? 1 : _depth + 2 * padVal;
    uint32_t numel = _width * _height * _depth, numelpad = wpad * hpad * dpad;
    int threadsSingle = 256;
    int blocks = (numel + threadsSingle - 1) / threadsSingle;

    thrust::device_vector<uint8_t> d_padded_binarized_image(numelpad, 0);
    thrust::device_vector<uint32_t> d_labels(numelpad), d_holes(numelpad);

    pad<T, uint32_t>(_pixels, d_labels, _width, _height, _depth, padVal);

    binarize << <blocks, threadsSingle >> > (thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_padded_binarized_image.data()), numel);


    dim3 threads = _depth == 1 ? dim3(8, 8) : dim3(8, 8, 8);
    dim3 grid = _depth == 1 ? dim3((unsigned int)ceil((float)wpad / (float)8), (unsigned int)ceil((float)hpad / (float)8)) : dim3((unsigned int)ceil((float)wpad / (float)8), (unsigned int)ceil((float)hpad / (float)8), (unsigned int)ceil((float)dpad / (float)8));
    dim3 gridOrig = _depth == 1 ? dim3((unsigned int)ceil((float)_width / (float)8), (unsigned int)ceil((float)_height / (float)8)) : dim3((unsigned int)ceil((float)_width / (float)8), (unsigned int)ceil((float)_height / (float)8), (unsigned int)ceil((float)_depth / (float)8));

    thrust::transform(thrust::device, d_padded_binarized_image.begin(), d_padded_binarized_image.end(), d_padded_binarized_image.begin(), [] __device__(uint8_t v) { return (v == 0) ? 255 : 0; });

    face_connected_component(d_padded_binarized_image, d_holes, wpad, hpad, 1);

    //After inversion, what was originally the background will now be the
    //first foreground label encountered.This is ensured due to the
    //single voxel padding done above and the fact that the face connected initialize each pixel with its index in the array
    //We therefor only keeps labels > 2 and give them the cuda max value to initialize the face connected flood filling

    thrust::transform(thrust::device, d_holes.begin(), d_holes.end(), d_holes.begin(), [] __device__(uint32_t v) { return (v == 1) ? 0 : v; });

    unpad<uint32_t, T>(d_holes, _holes, wpad, hpad, dpad, padVal);

     /*{
        std::string name("d:/test.tif");
        TinyTIFFWriterSampleFormat sampleF;
        uint16_t bitPerSample;
        auto w = _width, h = _height;
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
            std::vector <uint16_t> pixImage(w * h);
            thrust::device_vector<uint16_t> dtmp(numel2);
            copy_transfer<T, uint16_t> << <blocks2, threadsSingle2 >> > (thrust::raw_pointer_cast(_holes.data()), thrust::raw_pointer_cast(dtmp.data()), numel2);
            cudaMemcpy(pixImage.data(), thrust::raw_pointer_cast(dtmp.data()), w * h * sizeof(uint16_t), cudaMemcpyDeviceToHost);
            const void* data = (void*)pixImage.data();
            TinyTIFFWriter_writeImage(tif, data);
            TinyTIFFWriter_close(tif);
            std::cout << "Image " << name << std::endl;
        }
    }*/
}

template <class T>
void map_holes_to_labels(const thrust::device_vector <T>& _pixels, thrust::device_vector <T>& _holes, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    dim3 threads = _depth == 1 ? dim3(8, 8) : dim3(8, 8, 8);
    dim3 grid = _depth == 1 ? dim3((unsigned int)ceil((float)_width / (float)8), (unsigned int)ceil((float)_height / (float)8)) : dim3((unsigned int)ceil((float)_width / (float)8), (unsigned int)ceil((float)_height / (float)8), (unsigned int)ceil((float)_depth / (float)8));
    
    thrust::device_vector<uint32_t> d_changed(1);

    //Add label to allow filling holes with the correct label value
    thrust::transform(
        thrust::device,
        _pixels.begin(), _pixels.end(),
        _holes.begin(),
        _holes.begin(),
        [] __device__(T labelValue, T holeValue) -> T {
        if (holeValue > 0) {
            return ::cuda::std::numeric_limits<T>::max();
        }
        else {
            return labelValue;
        }
    }
    );

    int iteration = 1;
    uint32_t changed = 1;
    auto start2 = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();
    while (changed != 0) {
        thrust::fill(d_changed.begin(), d_changed.end(), 0);
        if (_depth == 1)
            face_cc_kernel_2d_iteration << < grid, threads >> > (thrust::raw_pointer_cast(_holes.data()), thrust::raw_pointer_cast(d_changed.data()), _width, _height);
        else
            face_cc_kernel_3d_iteration << < grid, threads >> > (thrust::raw_pointer_cast(_holes.data()), thrust::raw_pointer_cast(d_changed.data()), _width, _height, _depth);
        cudaDeviceSynchronize();
        cudaMemcpy(&changed, thrust::raw_pointer_cast(d_changed.data()), 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        iteration++;
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
    }

    //Remove labels to only keep holes with correct label value
    thrust::transform(
        thrust::device,
        _pixels.begin(), _pixels.end(),
        _holes.begin(),
        _holes.begin(),
        [] __device__(T labelValue, T holeValue) -> T {
        return holeValue - labelValue;
    }
    );

    auto duration2 = std::chrono::high_resolution_clock::now() - start2;
    long long ms = std::chrono::duration_cast<std::chrono::microseconds>(duration2).count();
    float s = std::chrono::duration_cast<std::chrono::seconds>(duration2).count();
    printf("Remap holes took %f seconds (%lld microseconds), and %u iterations\n", s, ms, iteration);
}

template <class T>
void fill_holes_gpu(thrust::device_vector<T>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const T _threshold)
{
    uint32_t numel = _width * _height * _depth;
    thrust::device_vector<T> d_holes(numel, 0);

    identify_holes(_pixels, d_holes, _width, _height, _depth);

    thrust::device_vector<T> d_holes_id(d_holes.size());
    thrust::device_vector<T> d_counts(d_holes.size());
    count_occurences_label_kernel_gpu(d_holes, d_holes_id, d_counts);
    // Build a keep/remove LUT
    T max_label = d_holes_id.back();
    thrust::device_vector<uint8_t> label_keep_lut(max_label + 1, 0);  // 0: remove, 1: keep

    for (size_t i = 0; i < d_holes_id.size(); ++i)
        if (d_counts[i] < _threshold)
            label_keep_lut[d_holes_id[i]] = 1;  // Keep this label
        else
            label_keep_lut[d_holes_id[i]] = 0;  // Remove this label (set to background)

    //thrust::copy(label_keep_lut.begin(), label_keep_lut.end(), std::ostream_iterator<uint32_t>(std::cout, ","));

    // Remove labels by setting them to 0 if they're not kept
    thrust::transform(
        thrust::device,
        d_holes.begin(),
        d_holes.end(),
        d_holes.begin(),
        [lut = label_keep_lut.data()] __device__(T label) {
        return lut[label] ? label : static_cast<T>(0);
    });

    map_holes_to_labels(_pixels, d_holes, _width, _height, _depth);

    thrust::transform(
        thrust::device,
        d_holes.begin(), d_holes.end(),
        _pixels.begin(),
        _pixels.begin(),
        [] __device__(T holeValue, T imageValue) -> T {
        if (holeValue > 0) {
            return holeValue;
        }
        else {
            return imageValue;
        }
    }
    );

}

template <class T>
void run_fill_holes_2(std::vector<T>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const T _threshold) {
    uint32_t numel = _width * _height * _depth;
    int threadsSingle = 256;
    int blocks = (numel + threadsSingle - 1) / threadsSingle;

    thrust::device_vector<T> d_image(_pixels);

    fill_holes_gpu(d_image, _width, _height, _depth, _threshold);

    cudaMemcpy(_pixels.data(), thrust::raw_pointer_cast(d_image.data()), _width * _height * _depth * sizeof(T), cudaMemcpyDeviceToHost);
}

template void fill_holes_gpu(thrust::device_vector<uint8_t>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint8_t _threshold);
template void fill_holes_gpu(thrust::device_vector<uint16_t>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint16_t _threshold);
template void fill_holes_gpu(thrust::device_vector<uint32_t>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint32_t _threshold);
template void fill_holes_gpu(thrust::device_vector<float>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const float _threshold);

template void run_fill_holes_2(std::vector<uint8_t>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint8_t _threshold);
template void run_fill_holes_2(std::vector<uint16_t>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint16_t _threshold);
template void run_fill_holes_2(std::vector<uint32_t>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const uint32_t _threshold);
template void run_fill_holes_2(std::vector<float>& _pixels, const uint32_t _width, const uint32_t _height, const uint32_t _depth, const float _threshold);