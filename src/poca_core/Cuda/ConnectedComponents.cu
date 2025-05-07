/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      3DConnectedComponents.cu
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
#include "BasicOperationsImage.h"

#ifndef NO_CUDA

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

// 2d
#define BLOCK_ROWS 16
#define BLOCK_COLS 16

namespace cc2d_stack
{
    __global__ void init_labeling(uint32_t* label, const uint32_t W, const uint32_t H, const uint32_t D)
    {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t depth = blockIdx.z * blockDim.z + threadIdx.z;
        const uint32_t idx = depth * W * H + row * W + col;

        if (row < H && col < W && depth < D)
            label[idx] = idx;
    }


    __global__ void merge(uint8_t* img, uint32_t* label, const uint32_t W, const uint32_t H, const uint32_t D)
    {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t depth = blockIdx.z * blockDim.z + threadIdx.z;
        const uint32_t idx = depth * W * H + row * W + col;

        if (row >= H || col >= W || depth >= D)
            return;

        uint32_t P = 0;

        // NOTE : Original Codes, but occurs silent error
        // NOTE : Programs keep runnig, but now showing printf logs, and the result is weird
        // uint8_t buffer[4] = {0};
        // if (col + 1 < W) {
        //     *(reinterpret_cast<uint16_t*>(buffer)) = *(reinterpret_cast<uint16_t*>(img + idx));
        //     if (row + 1 < H) {
        //         *(reinterpret_cast<uint16_t*>(buffer + 2)) = *(reinterpret_cast<uint16_t*>(img + idx + W));
        //     }
        // }
        // else {
        //     buffer[0] = img[idx];
        //     if (row + 1 < H)
        //         buffer[2] = img[idx + W];
        // }
        // if (buffer[0])              P |= 0x777;
        // if (buffer[1])              P |= (0x777 << 1);
        // if (buffer[2])              P |= (0x777 << 4);

        if (img[idx])                      P |= 0x777;
        if (row + 1 < H && img[idx + W])   P |= 0x777 << 4;
        if (col + 1 < W && img[idx + 1])   P |= 0x777 << 1;

        if (col == 0)               P &= 0xEEEE;
        if (col + 1 >= W)           P &= 0x3333;
        else if (col + 2 >= W)      P &= 0x7777;

        if (row == 0)               P &= 0xFFF0;
        if (row + 1 >= H)           P &= 0xFF;

        if (P > 0)
        {
            // If need check about top-left pixel(if flag the first bit) and hit the top-left pixel
            if (hasBit(P, 0) && img[idx - W - 1]) {
                union_(label, idx, idx - 2 * W - 2); // top left block
            }

            if ((hasBit(P, 1) && img[idx - W]) || (hasBit(P, 2) && img[idx - W + 1]))
                union_(label, idx, idx - 2 * W); // top bottom block

            if (hasBit(P, 3) && img[idx + 2 - W])
                union_(label, idx, idx - 2 * W + 2); // top right block

            if ((hasBit(P, 4) && img[idx - 1]) || (hasBit(P, 8) && img[idx + W - 1]))
                union_(label, idx, idx - 2); // just left block
        }
    }

    __global__ void compression(uint32_t* label, const int32_t W, const int32_t H, const uint32_t D)
    {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t depth = blockIdx.z * blockDim.z + threadIdx.z;
        const uint32_t idx = depth * W * H + row * W + col;


        if (row < H && col < W && depth < D)
            find_n_compress(label, idx);
    }

    __global__ void final_labeling(const uint8_t* img, uint32_t* label, const int32_t W, const int32_t H, const uint32_t D)
    {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t depth = blockIdx.z * blockDim.z + threadIdx.z;
        const uint32_t idx = depth * W * H + row * W + col;

        if (row >= H || col >= W || depth >= D)
            return;

        int32_t y = label[idx] + 1;

        if (img[idx])
            label[idx] = y;
        else
            label[idx] = 0;

        if (col + 1 < W)
        {
            if (img[idx + 1])
                label[idx + 1] = y;
            else
                label[idx + 1] = 0;

            if (row + 1 < H)
            {
                if (img[idx + W + 1])
                    label[idx + W + 1] = y;
                else
                    label[idx + W + 1] = 0;
            }
        }

        if (row + 1 < H)
        {
            if (img[idx + W])
                label[idx + W] = y;
            else
                label[idx + W] = 0;
        }
    }

} // namespace cc2d_stack

namespace cc2d
{
    __global__ void init_labeling(uint32_t* label, const uint32_t W, const uint32_t H)
    {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t idx = row * W + col;

        if (row < H && col < W)
            label[idx] = idx;
    }


    __global__ void merge(uint8_t* img, uint32_t* label, const uint32_t W, const uint32_t H)
    {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t idx = row * W + col;

        if (row >= H || col >= W)
            return;

        uint32_t P = 0;

        // NOTE : Original Codes, but occurs silent error
        // NOTE : Programs keep runnig, but now showing printf logs, and the result is weird
        // uint8_t buffer[4] = {0};
        // if (col + 1 < W) {
        //     *(reinterpret_cast<uint16_t*>(buffer)) = *(reinterpret_cast<uint16_t*>(img + idx));
        //     if (row + 1 < H) {
        //         *(reinterpret_cast<uint16_t*>(buffer + 2)) = *(reinterpret_cast<uint16_t*>(img + idx + W));
        //     }
        // }
        // else {
        //     buffer[0] = img[idx];
        //     if (row + 1 < H)
        //         buffer[2] = img[idx + W];
        // }
        // if (buffer[0])              P |= 0x777;
        // if (buffer[1])              P |= (0x777 << 1);
        // if (buffer[2])              P |= (0x777 << 4);

        if (img[idx])                      P |= 0x777;
        if (row + 1 < H && img[idx + W])   P |= 0x777 << 4;
        if (col + 1 < W && img[idx + 1])   P |= 0x777 << 1;

        if (col == 0)               P &= 0xEEEE;
        if (col + 1 >= W)           P &= 0x3333;
        else if (col + 2 >= W)      P &= 0x7777;

        if (row == 0)               P &= 0xFFF0;
        if (row + 1 >= H)           P &= 0xFF;

        if (P > 0)
        {
            // If need check about top-left pixel(if flag the first bit) and hit the top-left pixel
            if (hasBit(P, 0) && img[idx - W - 1]) {
                union_(label, idx, idx - 2 * W - 2); // top left block
            }

            if ((hasBit(P, 1) && img[idx - W]) || (hasBit(P, 2) && img[idx - W + 1]))
                union_(label, idx, idx - 2 * W); // top bottom block

            if (hasBit(P, 3) && img[idx + 2 - W])
                union_(label, idx, idx - 2 * W + 2); // top right block

            if ((hasBit(P, 4) && img[idx - 1]) || (hasBit(P, 8) && img[idx + W - 1]))
                union_(label, idx, idx - 2); // just left block
        }
    }

    __global__ void compression(uint32_t* label, const int32_t W, const int32_t H)
    {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t idx = row * W + col;


        if (row < H && col < W)
            find_n_compress(label, idx);
    }

    __global__ void final_labeling(const uint8_t* img, uint32_t* label, const int32_t W, const int32_t H)
    {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t idx = row * W + col;

        if (row >= H || col >= W)
            return;

        int32_t y = label[idx] + 1;

        if (img[idx])
            label[idx] = y;
        else
            label[idx] = 0;

        if (col + 1 < W)
        {
            if (img[idx + 1])
                label[idx + 1] = y;
            else
                label[idx + 1] = 0;

            if (row + 1 < H)
            {
                if (img[idx + W + 1])
                    label[idx + W + 1] = y;
                else
                    label[idx + W + 1] = 0;
            }
        }

        if (row + 1 < H)
        {
            if (img[idx + W])
                label[idx + W] = y;
            else
                label[idx + W] = 0;
        }
    }

} // namespace cc2d

namespace cc3d
{
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
}

#define BLOCK_X 8
#define BLOCK_Y 4
#define BLOCK_Z 4

void connectedComponnets2DLabelingBinary(uint8_t* const _pixels, const uint32_t W, const uint32_t H, uint32_t* _labels) {
    dim3 grid = dim3(((W + 1) / 2 + BLOCK_COLS - 1) / BLOCK_COLS, ((H + 1) / 2 + BLOCK_ROWS - 1) / BLOCK_ROWS);
    dim3 block = dim3(BLOCK_COLS, BLOCK_ROWS);

    cc2d::init_labeling << <grid, block >> > (_labels, W, H);
    cc2d::merge << <grid, block >> > (_pixels, _labels, W, H);
    cc2d::compression << <grid, block >> > (_labels, W, H);
    cc2d::final_labeling << <grid, block >> > (_pixels, _labels, W, H);
}

void connectedComponnets2DLabelingStackBinary(uint8_t* const _pixels, const uint32_t W, const uint32_t H, const uint32_t D, uint32_t* _labels) {
    dim3 grid = dim3(((W + 1) / 2 + BLOCK_X - 1) / BLOCK_X, ((H + 1) / 2 + BLOCK_Y - 1) / BLOCK_Y, (unsigned int)ceil((float)D / 8.f));
    dim3 block = dim3(BLOCK_X, BLOCK_Y, 8);

    cc2d_stack::init_labeling << <grid, block >> > (_labels, W, H, D);
    cc2d_stack::merge << <grid, block >> > (_pixels, _labels, W, H, D);
    cc2d_stack::compression << <grid, block >> > (_labels, W, H, D);
    cc2d_stack::final_labeling << <grid, block >> > (_pixels, _labels, W, H, D);
}

void connectedComponnets3DLabelingBinary(uint8_t* const _pixels, const size_t _nbValues, const uint32_t W, const uint32_t H, const uint32_t D, uint32_t* _labels) {
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

    cc3d::init_labeling << <grid, block >> > (_labels, W, H, D);
    cc3d::merge << <grid, block >> > (_pixels, _labels, last_cube_fg, W, H, D);
    cc3d::compression << <grid, block >> > (_labels, W, H, D);
    cc3d::final_labeling << <grid, block >> > (_labels, last_cube_fg, W, H, D);

    if (allocated_last_cude_fg_)
        cudaFree(last_cube_fg);
}

template <class T>
poca::core::ImageInterface* connectedComponnetsLabelingGPU(const T* _pixels, const T _thresholdMin, const T _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d) {

    const uint32_t D = _d;
    const uint32_t H = _h;
    const uint32_t W = _w;
    uint32_t nbValues = D * H * W;

    //TODO: add test about even number and pad if required
    /*AT_ASSERTM((D % 2) == 0, "shape must be a even number");
    AT_ASSERTM((H % 2) == 0, "shape must be a even number");
    AT_ASSERTM((W % 2) == 0, "shape must be a even number");*/

    uint8_t* thresholImage;
    cudaMalloc((void**)&thresholImage, nbValues * sizeof(uint8_t));
    dim3 block(32);
    dim3 grid((nbValues + block.x - 1) / block.x);
    thrust::device_vector<T> imageToThreshold(_pixels, _pixels + nbValues);
    kernel_threshold << <grid, block >> > (thrust::raw_pointer_cast(imageToThreshold.data()), _thresholdMin, _thresholdMax, thresholImage, nbValues);

    thrust::device_vector<uint32_t> d_labels(nbValues);
    if (D > 1)
        connectedComponnets3DLabelingBinary(thresholImage, nbValues, _w, _h, _d, thrust::raw_pointer_cast(d_labels.data()));
    else
        connectedComponnets2DLabelingBinary(thresholImage, _w, _h, thrust::raw_pointer_cast(d_labels.data()));
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

template poca::core::ImageInterface* connectedComponnetsLabelingGPU(const uint8_t* _pixels, const uint8_t _thresholdMin, const uint8_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
template poca::core::ImageInterface* connectedComponnetsLabelingGPU(const uint16_t* _pixels, const uint16_t _thresholdMin, const uint16_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
template poca::core::ImageInterface* connectedComponnetsLabelingGPU(const uint32_t* _pixels, const uint32_t _thresholdMin, const uint32_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
template poca::core::ImageInterface* connectedComponnetsLabelingGPU(const int32_t* _pixels, const int32_t _thresholdMin, const int32_t _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
template poca::core::ImageInterface* connectedComponnetsLabelingGPU(const float* _pixels, const float _thresholdMin, const float _thresholdMax, const uint32_t _w, const uint32_t _h, const uint32_t _d);
#endif