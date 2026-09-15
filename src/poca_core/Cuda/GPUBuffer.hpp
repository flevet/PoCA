#pragma once

#ifndef NO_CUDA
#define cuda_check(x) if (x!=cudaSuccess) exit(1);
#define IF_VERBOSE(x) //x

template <class T>
struct GPUBuffer
{
    void init(T* data)
    {
        IF_VERBOSE(
            std::cerr << "GPU: "
            << size * sizeof(T) / 1048576
            << " Mb used" << std::endl
        );

        cpu_data = data;
        cuda_check(cudaMalloc((void**)&gpu_data, size * sizeof(T)));
        cpu2gpu();
    }

    void init(std::vector<T>& data)
    {
        size = data.size();
        cpu_data = data.data();

        cuda_check(cudaMalloc((void**)&gpu_data, size * sizeof(T)));
        cpu2gpu();
    }

    GPUBuffer()
        : cpu_data(nullptr), gpu_data(nullptr), size(0)
    {}

    GPUBuffer(std::vector<T>& v)
        : cpu_data(nullptr), gpu_data(nullptr), size(v.size())
    {
        init(v.data());
    }

    GPUBuffer(T* v, size_t count)
        : cpu_data(nullptr), gpu_data(nullptr), size(count)
    {
        init(v);
    }

    ~GPUBuffer()
    {
        if (gpu_data != nullptr)
            cuda_check(cudaFree(gpu_data));
    }

    void cpu2gpu()
    {
        cuda_check(cudaMemcpy(
            gpu_data, cpu_data,
            size * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    void gpu2cpu()
    {
        cuda_check(cudaMemcpy(
            cpu_data, gpu_data,
            size * sizeof(T),
            cudaMemcpyDeviceToHost));
    }

    T* cpu_data;
    T* gpu_data;
    size_t size;
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

#endif