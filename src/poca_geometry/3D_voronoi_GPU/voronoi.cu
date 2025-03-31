/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      voronoi.cu
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

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "voronoi.h"

/*#define VORO_BLOCK_SIZE 16
#define KNN_BLOCK_SIZE  32
#define _K_             200
#define _MAX_P_         64
#define _MAX_T_         96

#define IF_VERBOSE(x) x*/

enum Status {
	triangle_overflow = 0,
	vertex_overflow = 1,
	inconsistent_boundary = 2,
	security_radius_not_reached = 3,
	success = 4,
	needs_exact_predicates = 5
};

#define cuda_check(x) if (x!=cudaSuccess) exit(1);

#define FOR(I,UPPERBND) for(int I = 0; I<int(UPPERBND); ++I)

typedef unsigned char uchar;  // local indices with special values

static const uchar END_OF_LIST = 255;

__shared__ uchar3 tr_data[VORO_BLOCK_SIZE * _MAX_T_]; // memory pool for chained lists of triangles
__shared__ uchar boundary_next_data[VORO_BLOCK_SIZE * _MAX_P_];
__shared__ float4 clip_data[VORO_BLOCK_SIZE * _MAX_P_]; // clipping planes

inline  __device__ uchar3& tr(int t) { return  tr_data[threadIdx.x*_MAX_T_ + t]; }
inline  __device__ uchar& boundary_next(int v) { return  boundary_next_data[threadIdx.x*_MAX_P_ + v]; }
inline  __device__ float4& clip(int v) { return  clip_data[threadIdx.x*_MAX_P_ + v]; }

__device__ float4 point_from_ptr3(float* f) {
	return make_float4(f[0], f[1], f[2], 1);
}
__device__ float4 minus4(float4 A, float4 B) {
	return make_float4(A.x - B.x, A.y - B.y, A.z - B.z, A.w - B.w);
}
__device__ float4 plus4(float4 A, float4 B) {
	return make_float4(A.x + B.x, A.y + B.y, A.z + B.z, A.w + B.w);
}
__device__ float dot4(float4 A, float4 B) {
	return A.x*B.x + A.y*B.y + A.z*B.z + A.w*B.w;
}
__device__ float dot3(float4 A, float4 B) {
	return A.x*B.x + A.y*B.y + A.z*B.z;
}
__device__ float4 mul3(float s, float4 A) {
	return make_float4(s*A.x, s*A.y, s*A.z, 1.);
}
__device__ float4 div3(float s, float4 A) {
	return make_float4(A.x / s, A.y / s, A.z / s, 1.);
}
__device__ float4 cross3(float4 A, float4 B) {
	return make_float4(A.y*B.z - A.z*B.y, A.z*B.x - A.x*B.z, A.x*B.y - A.y*B.x, 0);
}
__device__ float4 plane_from_point_and_normal(float4 P, float4 n) {
	return  make_float4(n.x, n.y, n.z, -dot3(P, n));
}
__device__ inline float det2x2(float a11, float a12, float a21, float a22) {
	return a11*a22 - a12*a21;
}
__device__ inline float det3x3(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33) {
	return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
}

__device__ inline float det4x4(
	float a11, float a12, float a13, float a14,
	float a21, float a22, float a23, float a24,
	float a31, float a32, float a33, float a34,
	float a41, float a42, float a43, float a44
	) {
	float m12 = a21*a12 - a11*a22;
	float m13 = a31*a12 - a11*a32;
	float m14 = a41*a12 - a11*a42;
	float m23 = a31*a22 - a21*a32;
	float m24 = a41*a22 - a21*a42;
	float m34 = a41*a32 - a31*a42;

	float m123 = m23*a13 - m13*a23 + m12*a33;
	float m124 = m24*a13 - m14*a23 + m12*a43;
	float m134 = m34*a13 - m14*a33 + m13*a43;
	float m234 = m34*a23 - m24*a33 + m23*a43;

	return (m234*a14 - m134*a24 + m124*a34 - m123*a44);
}

__device__ inline double det2x2(double a11, double a12, double a21, double a22) {
	return a11*a22 - a12*a21;
}

__device__ inline double det3x3(double a11, double a12, double a13, double a21, double a22, double a23, double a31, double a32, double a33) {
	return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
}

__device__ inline double det4x4(
	double a11, double a12, double a13, double a14,
	double a21, double a22, double a23, double a24,
	double a31, double a32, double a33, double a34,
	double a41, double a42, double a43, double a44
	) {
	double m12 = a21*a12 - a11*a22;
	double m13 = a31*a12 - a11*a32;
	double m14 = a41*a12 - a11*a42;
	double m23 = a31*a22 - a21*a32;
	double m24 = a41*a22 - a21*a42;
	double m34 = a41*a32 - a31*a42;

	double m123 = m23*a13 - m13*a23 + m12*a33;
	double m124 = m24*a13 - m14*a23 + m12*a43;
	double m134 = m34*a13 - m14*a33 + m13*a43;
	double m234 = m34*a23 - m24*a33 + m23*a43;

	return (m234*a14 - m134*a24 + m124*a34 - m123*a44);
}

__device__ inline float get_tet_volume(float4 A, float4 B, float4 C) {
	return -det3x3(A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z) / 6.;
}
__device__ void get_tet_volume_and_barycenter(float4& bary, float& volume, float4 A, float4 B, float4 C, float4 D) {
	volume = get_tet_volume(minus4(A, D), minus4(B, D), minus4(C, D));
	bary = make_float4(.25*(A.x + B.x + C.x + D.x), .25*(A.y + B.y + C.y + D.y), .25*(A.z + B.z + C.z + D.z), 1);
}
__device__ float4 project_on_plane(float4 P, float4 plane) {
	float4 n = make_float4(plane.x, plane.y, plane.z, 0);
	float lambda = (dot4(n, P) + plane.w) / dot4(n, n);
	//    lambda = (dot3(n, P) + plane.w) / norm23(n);
	return plus4(P, mul3(-lambda, n));
}
template <typename T> __device__ void inline swap(T& a, T& b) { T c(a); a = b; b = c; }

inline __device__ float max4(float a, float b, float c, float d) {
	return fmaxf(fmaxf(a, b), fmaxf(c, d));
}

inline __device__ void get_minmax3(
	float& m, float& M, float x1, float x2, float x3
	) {
	m = fminf(fminf(x1, x2), x3);
	M = fmaxf(fmaxf(x1, x2), x3);
}

inline __device__ double max4(double a, double b, double c, double d) {
	return fmax(fmax(a, b), fmax(c, d));
}

inline __device__ void get_minmax3(
	double& m, double& M, double x1, double x2, double x3
	) {
	m = fmin(fmin(x1, x2), x3);
	M = fmax(fmax(x1, x2), x3);
}

struct ConvexCell {
	Status* status;
	uchar nb_t;
	uchar nb_r;
	float* pts;
	int voro_id;
	float4 voro_seed;
	uchar nb_v;
	uchar first_boundary_;

	__device__ ConvexCell(int p_seed, float* p_pts, Status *p_status, float * p_bbox) {
		float eps = 0.f;
		float xmin = p_bbox[0] - eps;
		float ymin = p_bbox[1] - eps;
		float zmin = p_bbox[2] - eps;
		float xmax = p_bbox[3] + eps;
		float ymax = p_bbox[4] + eps;
		float zmax = p_bbox[5] + eps;
		pts = p_pts;
		first_boundary_ = END_OF_LIST;
		FOR(i, _MAX_P_)
			boundary_next(i) = END_OF_LIST;
		voro_id = p_seed;
		voro_seed = make_float4(pts[3 * voro_id], pts[3 * voro_id + 1], pts[3 * voro_id + 2], 1);
		status = p_status;
		*status = success;

		clip(0) = make_float4(1.0, 0.0, 0.0, -xmin);
		clip(1) = make_float4(-1.0, 0.0, 0.0, xmax);
		clip(2) = make_float4(0.0, 1.0, 0.0, -ymin);
		clip(3) = make_float4(0.0, -1.0, 0.0, ymax);
		clip(4) = make_float4(0.0, 0.0, 1.0, -zmin);
		clip(5) = make_float4(0.0, 0.0, -1.0, zmax);
		nb_v = 6;

		tr(0) = make_uchar3(2, 5, 0);
		tr(1) = make_uchar3(5, 3, 0);
		tr(2) = make_uchar3(1, 5, 2);
		tr(3) = make_uchar3(5, 1, 3);
		tr(4) = make_uchar3(4, 2, 0);
		tr(5) = make_uchar3(4, 0, 3);
		tr(6) = make_uchar3(2, 4, 1);
		tr(7) = make_uchar3(4, 3, 1);
		nb_t = 8;
	}

	__device__  bool is_security_radius_reached(float4 last_neig) {
		// finds furthest voro vertex distance2
		float v_dist = 0;
		FOR(i, nb_t) {
			float4 pc = compute_triangle_point(tr(i));
			float4 diff = minus4(pc, voro_seed);
			float d2 = dot3(diff, diff); // TODO safe to put dot4 here, diff.w = 0
			v_dist = max(d2, v_dist);
		}
		//compare to new neighbors distance2
		float4 diff = minus4(last_neig, voro_seed); // TODO it really should take index of the neighbor instead of the float4, then would be safe to put dot4
		float d2 = dot3(diff, diff);
		return (d2 > 4 * v_dist);
	}

	__device__ inline  uchar& ith_plane(uchar t, int i) {
		return reinterpret_cast<uchar *>(&(tr(t)))[i];
	}

	__device__ float4 compute_triangle_point(uchar3 t/*, bool persp_divide*/) const {
		bool persp_divide = true;
		float4 pi1 = clip(t.x);
		float4 pi2 = clip(t.y);
		float4 pi3 = clip(t.z);
		float4 result;
		result.x = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
		result.y = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
		result.z = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
		result.w = det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
		if (persp_divide) return make_float4(result.x / result.w, result.y / result.w, result.z / result.w, 1);
		return result;
	}

	__device__ bool triangle_is_in_conflict(uchar3 t, float4 eqn) const {
		return triangle_is_in_conflict_float(t, eqn);
	}


	__device__ bool triangle_is_in_conflict_float(uchar3 t, float4 eqn) const {
		float4 pi1 = clip(t.x);
		float4 pi2 = clip(t.y);
		float4 pi3 = clip(t.z);
		float det = det4x4(
			pi1.x, pi2.x, pi3.x, eqn.x,
			pi1.y, pi2.y, pi3.y, eqn.y,
			pi1.z, pi2.z, pi3.z, eqn.z,
			pi1.w, pi2.w, pi3.w, eqn.w
			);

		return (det > 0.0f);
	}

	__device__ bool triangle_is_in_conflict_double(uchar3 t, float4 eqn_f) const {
		float4 pi1_f = clip(t.x);
		float4 pi2_f = clip(t.y);
		float4 pi3_f = clip(t.z);

		double4 eqn = make_double4(eqn_f.x, eqn_f.y, eqn_f.z, eqn_f.w);
		double4 pi1 = make_double4(pi1_f.x, pi1_f.y, pi1_f.z, pi1_f.w);
		double4 pi2 = make_double4(pi2_f.x, pi2_f.y, pi2_f.z, pi2_f.w);
		double4 pi3 = make_double4(pi3_f.x, pi3_f.y, pi3_f.z, pi3_f.w);

		double det = det4x4(
			pi1.x, pi2.x, pi3.x, eqn.x,
			pi1.y, pi2.y, pi3.y, eqn.y,
			pi1.z, pi2.z, pi3.z, eqn.z,
			pi1.w, pi2.w, pi3.w, eqn.w
			);

		return (det > 0.0f);
	}



	__device__ void new_triangle(uchar i, uchar j, uchar k) {
		if (nb_t + 1 >= _MAX_T_) {
			*status = triangle_overflow;
			return;
		}
		tr(nb_t) = make_uchar3(i, j, k);
		nb_t++;
	}

	__device__ int new_point(int vid) {
		if (nb_v >= _MAX_P_) {
			*status = vertex_overflow;
			return -1;
		}

		float4 B = point_from_ptr3(pts + 3 * vid);
		float4 dir = minus4(voro_seed, B);
		float4 ave2 = plus4(voro_seed, B);
		float dot = dot3(ave2, dir); // TODO safe to put dot4 here, dir.w = 0
		clip(nb_v) = make_float4(dir.x, dir.y, dir.z, -dot / 2.f);
		nb_v++;
		return nb_v - 1;
	}

	__device__ void compute_boundary() {
		// clean circular list of the boundary
		FOR(i, _MAX_P_) boundary_next(i) = END_OF_LIST;
		first_boundary_ = END_OF_LIST;

		int nb_iter = 0;
		uchar t = nb_t;
		while (nb_r > 0) {
			if (nb_iter++ > 100) {
				*status = inconsistent_boundary;
				return;
			}
			bool is_in_border[3];
			bool next_is_opp[3];
			FOR(e, 3)   is_in_border[e] = (boundary_next(ith_plane(t, e)) != END_OF_LIST);
			FOR(e, 3)   next_is_opp[e] = (boundary_next(ith_plane(t, (e + 1) % 3)) == ith_plane(t, e));

			bool new_border_is_simple = true;
			// check for non manifoldness
			FOR(e, 3) if (!next_is_opp[e] && !next_is_opp[(e + 1) % 3] && is_in_border[(e + 1) % 3]) new_border_is_simple = false;

			// check for more than one boundary ... or first triangle
			if (!next_is_opp[0] && !next_is_opp[1] && !next_is_opp[2]) {
				if (first_boundary_ == END_OF_LIST) {
					FOR(e, 3) boundary_next(ith_plane(t, e)) = ith_plane(t, (e + 1) % 3);
					first_boundary_ = tr(t).x;
				}
				else new_border_is_simple = false;
			}

			if (!new_border_is_simple) {
				t++;
				if (t == nb_t + nb_r) t = nb_t;
				continue;
			}

			// link next
			FOR(e, 3) if (!next_is_opp[e]) boundary_next(ith_plane(t, e)) = ith_plane(t, (e + 1) % 3);

			// destroy link from removed vertices
			FOR(e, 3)  if (next_is_opp[e] && next_is_opp[(e + 1) % 3]) {
				if (first_boundary_ == ith_plane(t, (e + 1) % 3)) first_boundary_ = boundary_next(ith_plane(t, (e + 1) % 3));
				boundary_next(ith_plane(t, (e + 1) % 3)) = END_OF_LIST;
			}

			//remove triangle from R, and restart iterating on R
			swap(tr(t), tr(nb_t + nb_r - 1));
			t = nb_t;
			nb_r--;
		}
	}

	__device__ void  clip_by_plane(int vid) {
		int cur_v = new_point(vid); // add new plane equation
		if (*status == vertex_overflow) return;
		float4 eqn = clip(cur_v);
		nb_r = 0;

		int i = 0;
		while (i < nb_t) { // for all vertices of the cell
			if (triangle_is_in_conflict(tr(i), eqn)) {
				nb_t--;
				swap(tr(i), tr(nb_t));
				nb_r++;
			}
			else i++;
		}

		if (*status == needs_exact_predicates) {
			return;
		}

		if (nb_r == 0) { // if no clips, then remove the plane equation
			nb_v--;
			return;
		}

		// Step 2: compute cavity boundary
		compute_boundary();
		if (*status != success) return;
		if (first_boundary_ == END_OF_LIST) return;

		// Step 3: Triangulate cavity
		uchar cir = first_boundary_;
		do {
			new_triangle(cur_v, cir, boundary_next(cir));
			if (*status != success) return;
			cir = boundary_next(cir);
		} while (cir != first_boundary_);
	}
};

//----------------------------------WRAPPER
template <class T> struct GPUBuffer {
	void init(T* data) {
		IF_VERBOSE(std::cerr << "GPU: " << size * sizeof(T) / 1048576 << " Mb used" << std::endl);
		cpu_data = data;
		cuda_check(cudaMalloc((void**)& gpu_data, size * sizeof(T)));
		cpu2gpu();
	}
	GPUBuffer(std::vector<T>& v) { size = v.size(); init(v.data()); }
	GPUBuffer(T * _v, int _size) { size = _size; init(_v); }
	~GPUBuffer() { cuda_check(cudaFree(gpu_data)); }

	void cpu2gpu() { cuda_check(cudaMemcpy(gpu_data, cpu_data, size * sizeof(T), cudaMemcpyHostToDevice)); }
	void gpu2cpu() { cuda_check(cudaMemcpy(cpu_data, gpu_data, size * sizeof(T), cudaMemcpyDeviceToHost)); }

	T* cpu_data;
	T* gpu_data;
	int size;
};

char StatusStr[6][128] = {
	"triangle_overflow", "vertex_overflow", "inconsistent_boundary", "security_radius_not_reached", "success", "needs_exact_predicates"
};

void show_status_stats(std::vector<Status> &stat) {
	IF_VERBOSE(std::cerr << " \n\n\n---------Summary of success/failure------------\n");
	std::vector<int> nb_statuss(6, 0);
	FOR(i, stat.size()) nb_statuss[stat[i]]++;
	IF_VERBOSE(FOR(r, 6) std::cerr << " " << StatusStr[r] << "   " << nb_statuss[r] << "\n";)
		std::cerr << " " << StatusStr[4] << "   " << nb_statuss[4] << " /  " << stat.size() << "\n";
}

void cuda_check_error() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) { fprintf(stderr, "Failed (1) (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
}


void printDevProp() {

	int devCount; // Number of CUDA devices
	cudaError_t err = cudaGetDeviceCount(&devCount);
	if (err != cudaSuccess) {
		std::cerr << "Failed to initialize CUDA / failed to count CUDA devices (error code << "
			<< cudaGetErrorString(err) << ")! [File:      voronoi.cu " << __FILE__ << ", line: " << __LINE__ << "]" << std::endl;
		exit(1);
	}

	printf("CUDA Device Query...\n");
	printf("There are %d CUDA devices.\n", devCount);

	// Iterate through devices
	for (int i = 0; i<devCount; ++i) {
		// Get device properties
		printf("\nCUDA Device #%d\n", i);
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		printf("Major revision number:         %d\n", devProp.major);
		printf("Minor revision number:         %d\n", devProp.minor);
		printf("Name:                          %s\n", devProp.name);
		printf("Total global memory:           %lu\n", devProp.totalGlobalMem);
		printf("Total shared memory per block: %lu\n", devProp.sharedMemPerBlock);
		printf("Total registers per block:     %d\n", devProp.regsPerBlock);
		printf("Warp size:                     %d\n", devProp.warpSize);
		printf("Maximum memory pitch:          %lu\n", devProp.memPitch);
		printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
		for (int i = 0; i < 3; ++i)
			printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
		for (int i = 0; i < 3; ++i)
			printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
		printf("Clock rate:                    %d\n", devProp.clockRate);
		printf("Total constant memory:         %lu\n", devProp.totalConstMem);
		printf("Texture alignment:             %lu\n", devProp.textureAlignment);
		printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
		printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
		printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	}
}

__device__ void get_tet_decomposition_of_vertex(ConvexCell& cc, int t, float4* P) {
	float4 C = cc.voro_seed;
	float4 A = cc.compute_triangle_point(tr(t));
	FOR(i, 3)  P[2 * i] = project_on_plane(C, clip(cc.ith_plane(t, i)));
	FOR(i, 3) P[2 * i + 1] = project_on_plane(A, plane_from_point_and_normal(C, cross3(minus4(P[2 * i], C), minus4(P[(2 * (i + 1)) % 6], C))));
}

__device__ void export_bary_and_volume(ConvexCell& cc, float* out_pts, int seed) {
	float4 tet_bary;
	float tet_vol;
	float4 bary_sum = make_float4(0, 0, 0, 0);
	float cell_vol = 0;
	float4 P[6];
	float4 C = cc.voro_seed;

	FOR(t, cc.nb_t) {
		float4 A = cc.compute_triangle_point(tr(t));
		get_tet_decomposition_of_vertex(cc, t, P);
		FOR(i, 6) {
			get_tet_volume_and_barycenter(tet_bary, tet_vol, P[i], P[(i + 1) % 6], C, A);
			bary_sum = plus4(bary_sum, mul3(tet_vol, tet_bary));
			cell_vol += tet_vol;
		}
	}
	out_pts[3 * seed] = bary_sum.x / cell_vol;
	out_pts[3 * seed + 1] = bary_sum.y / cell_vol;
	out_pts[3 * seed + 2] = bary_sum.z / cell_vol;
}

__device__ void export_volume(ConvexCell& cc, float* _volumes, int seed, float * cells, int * _nbTCells) {
	float4 tet_bary;
	float tet_vol;
	float4 bary_sum = make_float4(0, 0, 0, 0);
	float cell_vol = 0;
	float4 P[6];
	float4 C = cc.voro_seed;

	int realNbTet = 0; //nb_connected_faces_dual
	FOR(t, cc.nb_t) {
		int index = seed * _MAX_T_ * 3 + realNbTet * 3;
		float4 tmp = cc.compute_triangle_point(tr(t));
		cells[index + 0] = tmp.x;
		cells[index + 1] = tmp.y;
		cells[index + 2] = tmp.z;
		realNbTet++;
	}

	FOR(t, cc.nb_t) {
		float4 A = cc.compute_triangle_point(tr(t));
		get_tet_decomposition_of_vertex(cc, t, P);
		FOR(i, 6) {
			get_tet_volume_and_barycenter(tet_bary, tet_vol, P[i], P[(i + 1) % 6], C, A);
			bary_sum = plus4(bary_sum, mul3(tet_vol, tet_bary));
			cell_vol += tet_vol;
		}
	}
	_volumes[seed] = cell_vol;
	_nbTCells[seed] = realNbTet;
}

//###################  KERNEL   ######################
__device__ void compute_voro_cell_first_ring(float * pts, int nbpts, uint32_t* ci, uint32_t* ro, Status * gpu_stat, float* volumes, uint8_t* borders, float * bbox, /*float * _cells, unsigned int * _nbTCells, */int seed) {
	//create BBox
	volumes[seed] = -1;

	ConvexCell cc(seed, pts, &(gpu_stat[seed]), bbox);

	for (int v = ro[seed]; v < ro[seed + 1]; v++) {
		uint32_t z = ci[v];
		cc.clip_by_plane(z);
		if (gpu_stat[seed] != success) {
			return;
		}
	}

	volumes[seed] = -2;

	float4 tet_bary;
	float tet_vol;
	float4 bary_sum = make_float4(0, 0, 0, 0);
	float cell_vol = 0;
	float4 P[6];
	float4 C = cc.voro_seed;

	uint8_t border = 0;
	FOR(t, cc.nb_t) {
		float4 A = cc.compute_triangle_point(tr(t));
		get_tet_decomposition_of_vertex(cc, t, P);
		FOR(i, 6) {
			get_tet_volume_and_barycenter(tet_bary, tet_vol, P[i], P[(i + 1) % 6], C, A);
			bary_sum = plus4(bary_sum, mul3(tet_vol, tet_bary));
			cell_vol += tet_vol;
		}

		uchar3 ts = tr(t);
		if (ts.x < 6) border = 1;
		if (ts.y < 6) border = 1;
		if (ts.z < 6) border = 1;
	}
	volumes[seed] = cell_vol;
	borders[seed] = border;
}

//----------------------------------KERNEL
__global__ void voro_cell_test_GPU_param_first_ring(float * pts, int nbpts, uint32_t* ci, uint32_t* ro, Status * gpu_stat, float* _volumes, uint8_t* _borders, float * bbox) {
	int seed = blockIdx.x * blockDim.x + threadIdx.x;
	if (seed >= nbpts) return;
	gpu_stat[seed] = security_radius_not_reached;
	compute_voro_cell_first_ring(pts, nbpts, ci, ro, gpu_stat, _volumes, _borders, bbox, seed);
}

//###################  KERNEL   ######################
__device__ void compute_voro_cell_first_ring_with_construction_cells(float * pts, int nbpts, uint32_t* ci, uint32_t* ro, Status* gpu_stat, float* volumes, uint8_t* borders, float * bbox, float * _cells, unsigned int * _nbTCells, int seed, unsigned int numPointsChunkIndex) {
	//create BBox
	ConvexCell cc(seed, pts, &(gpu_stat[seed]), bbox);

	for (int v = ro[seed]; v < ro[seed + 1]; v++) {
		uint32_t z = ci[v];
		cc.clip_by_plane(z);
		if (gpu_stat[seed] != success) {
			return;
		}
	}

	float4 tet_bary;
	float tet_vol;
	float4 bary_sum = make_float4(0, 0, 0, 0);
	float cell_vol = 0;
	float4 P[6];
	float4 C = cc.voro_seed;

	int realNbTet = 0;
	FOR(t, cc.nb_t) {
		unsigned int index = /*seed*/numPointsChunkIndex * _MAX_T_ * 3 + realNbTet * 3;
		float4 tmp = cc.compute_triangle_point(tr(t));
		_cells[index + 0] = tmp.x;
		_cells[index + 1] = tmp.y;
		_cells[index + 2] = tmp.z;
		realNbTet++;
	}

	uint8_t border = 0;
	FOR(t, cc.nb_t) {
		float4 A = cc.compute_triangle_point(tr(t));
		get_tet_decomposition_of_vertex(cc, t, P);
		FOR(i, 6) {
			get_tet_volume_and_barycenter(tet_bary, tet_vol, P[i], P[(i + 1) % 6], C, A);
			bary_sum = plus4(bary_sum, mul3(tet_vol, tet_bary));
			cell_vol += tet_vol;
		}
		uchar3 ts = tr(t);
		if (ts.x < 6) border = 1;
		if (ts.y < 6) border = 1;
		if (ts.z < 6) border = 1;
	}
	volumes[seed] = cell_vol;
	_nbTCells[seed] = realNbTet;
	borders[seed] = border;
}

//----------------------------------KERNEL
__global__ void voro_cell_test_GPU_param_first_ring_with_construction_cells(float * pts, int nbpts, uint32_t* ci, uint32_t* ro, Status* gpu_stat, float* _volumes, uint8_t* _borders, float * bbox, float * _cells, unsigned int * _nbTCells, unsigned int _numPointChunkIndex, unsigned int _chunkNbPoints) {
	int seed = _numPointChunkIndex + (blockIdx.x * blockDim.x + threadIdx.x);
	if (seed >= _chunkNbPoints) return;
	gpu_stat[seed] = security_radius_not_reached;
	compute_voro_cell_first_ring_with_construction_cells(pts, nbpts, ci, ro, gpu_stat, _volumes, _borders, bbox, _cells, _nbTCells, seed, seed - _numPointChunkIndex);
}

void computeVoronoiFirstRing(std::vector <float> & _pts, std::vector <uint32_t> & _ci, std::vector <uint32_t> & _ro, std::vector <float> & _volumes, std::vector <uint8_t>& _borders, std::vector <float> & _cells, std::vector <unsigned int> & _nbTCells, bool _no_ConstructCells, unsigned int _sizeBuffer)
{
	int nb_pts = _pts.size() / 3;
	std::vector<float> bary(_pts.size(), 0);
	std::vector<Status> stat(nb_pts, security_radius_not_reached);

	std::vector <float> bbox(6);
	bbox[0] = bbox[3] = _pts[0];
	bbox[1] = bbox[4] = _pts[1];
	bbox[2] = bbox[5] = _pts[2];

	for (unsigned int n = 1; n < nb_pts; n++){
		float * pt = &_pts[n * 3];
		if (pt[0] < bbox[0])
			bbox[0] = pt[0];
		if (pt[1] < bbox[1])
			bbox[1] = pt[1];
		if (pt[2] < bbox[2])
			bbox[2] = pt[2];

		if (pt[0] > bbox[3])
			bbox[3] = pt[0];
		if (pt[1] > bbox[4])
			bbox[4] = pt[1];
		if (pt[2] > bbox[5])
			bbox[5] = pt[2];
	}

	float w = bbox[3] - bbox[0], h = bbox[4] - bbox[1], t = bbox[5] - bbox[2];
	float maxD = w < h ? w : h;
	maxD = maxD < t ? maxD : t;
	float eps = maxD / 100.f;
	FOR(n, 3) bbox[n] = bbox[n] - eps;
	FOR(n, 3) bbox[n + 3] = bbox[n + 3] + eps;

	GPUBuffer<float> out_pts(_pts);
	GPUBuffer<uint32_t> out_ci(_ci);
	GPUBuffer<uint32_t> out_ro(_ro);
	GPUBuffer<float> out_volumes(_volumes);
	GPUBuffer<uint8_t> out_borders(_borders);
	GPUBuffer<Status> gpu_stat(stat);
	GPUBuffer<float> out_bbox(bbox);
	GPUBuffer<unsigned int> out_nbTCells(_nbTCells);


	if (_no_ConstructCells){

		voro_cell_test_GPU_param_first_ring << < nb_pts / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE >> > (
			out_pts.gpu_data, nb_pts, out_ci.gpu_data, out_ro.gpu_data, gpu_stat.gpu_data, out_volumes.gpu_data, out_borders.gpu_data, out_bbox.gpu_data/*, out_cells.gpu_data, out_nbTCells.gpu_data*/
			);

	}
	else{
		int numChunks = ceil((float)_cells.size() / (float)_sizeBuffer), chunkIndex = 0;
		int numPointsChuncks = ceil((float)nb_pts / (float)numChunks), numPointsChunkIndex = 0;
		for (unsigned int n = 0; n < numChunks; n++){
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);

			unsigned int bufferLength = (n < numChunks - 1) ? _sizeBuffer : _cells.size() - n * _sizeBuffer;
			unsigned int numPoints = (n < numChunks - 1) ? numPointsChuncks : nb_pts - n * numPointsChuncks;

			GPUBuffer<float> out_cells(&_cells[chunkIndex], bufferLength);
			voro_cell_test_GPU_param_first_ring_with_construction_cells << < numPoints / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE >> > (
				out_pts.gpu_data, nb_pts, out_ci.gpu_data, out_ro.gpu_data, gpu_stat.gpu_data, out_volumes.gpu_data, out_borders.gpu_data, out_bbox.gpu_data, out_cells.gpu_data, out_nbTCells.gpu_data, numPointsChunkIndex, numPointsChunkIndex + numPoints
				);
			cuda_check_error();

			out_cells.gpu2cpu();
			chunkIndex += bufferLength;
			numPointsChunkIndex += numPoints;

			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			IF_VERBOSE(std::cerr << "GPU voro construction - step " << (n + 1) << ": " << milliseconds << " msec" << std::endl);
		}
	}

	out_volumes.gpu2cpu();
	out_borders.gpu2cpu();
	out_nbTCells.gpu2cpu();
	gpu_stat.gpu2cpu();
	show_status_stats(stat);
}

