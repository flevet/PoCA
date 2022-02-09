#ifndef H_VORONOI_H
#define H_VORONOI_H

#include "params.h"

/*#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "voronoi_defs.h"
#include "params.h"

#define cuda_check(x) if (x!=cudaSuccess) exit(1);

#define FOR(I,UPPERBND) for(int I = 0; I<int(UPPERBND); ++I)

typedef unsigned char uchar;  // local indices with special values



static const uchar END_OF_LIST = 255;

struct PointCH {
	double x, y, z;
	PointCH *prev, *next;

	__device__ void act();
};

struct PointCHQ { double x, y, z; };

struct ConvexCell {
    __device__ ConvexCell(int p_seed, float* p_pts, Status* p_status, float *);
    __device__ void clip_by_plane(int vid);
    __device__ float4 compute_triangle_point(uchar3 t, bool persp_divide=true) const;
    __device__ inline  uchar& ith_plane(uchar t, int i);
    __device__ int new_point(int vid);
    __device__ void new_triangle(uchar i, uchar j, uchar k);
    __device__ void compute_boundary();
    __device__ bool is_security_radius_reached(float4 last_neig);
    
    __device__ bool triangle_is_in_conflict(uchar3 t, float4 eqn) const {
//        return triangle_is_in_conflict_double(t, eqn);
        return triangle_is_in_conflict_float(t, eqn);
    }

    __device__ bool triangle_is_in_conflict_float(uchar3 t, float4 eqn) const;
    __device__ bool triangle_is_in_conflict_double(uchar3 t, float4 eqn) const;

    
    Status* status;
    uchar nb_t;
    uchar nb_r;
    float* pts;
    int voro_id;
    float4 voro_seed;
    uchar nb_v;
    uchar first_boundary_;     
};

void compute_voro_diagram_GPU(
    std::vector<float>& pts, std::vector<Status> &stat, std::vector<float>& bary, 
    std::vector<int>* KNN = NULL, // Optional: readback K nearest neighbors.
    int nb_Lloyd_iter = 0         // Optional: Lloyd iterations (not implemented ? TO BE CHECKED)
);

void computeVoronoi(std::vector <float> &, std::vector <unsigned int> &, std::vector <float> &);

void computeKNeighbors(std::vector <unsigned int> &, std::vector <unsigned int> &, std::vector <unsigned int> &);
*/
void computeVoronoiFirstRing(std::vector <float> &, std::vector <unsigned int> &, std::vector <unsigned int> &, std::vector <float> &, std::vector <float> &, std::vector <unsigned int> &, bool, unsigned int);
#endif // __VORONOI_H__

