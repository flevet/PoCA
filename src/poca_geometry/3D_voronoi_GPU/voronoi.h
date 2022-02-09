/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      voronoi.h
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

#ifndef H_VORONOI_H
#define H_VORONOI_H

#if 0==PRESET               // conservative settings (white noise)
#define VORO_BLOCK_SIZE 16
#define KNN_BLOCK_SIZE  32
#define _K_             200
#define _MAX_P_         64
#define _MAX_T_         96//200
#elif 1==PRESET            // perturbed grid settings
#define VORO_BLOCK_SIZE 16
#define KNN_BLOCK_SIZE  32
#define _K_             90
#define _MAX_P_         50
#define _MAX_T_         96
#elif 2==PRESET            // blue noise settings
#define VORO_BLOCK_SIZE 16
#define KNN_BLOCK_SIZE  64
#define _K_             35
#define _MAX_P_         32
#define _MAX_T_         96
#endif

// Uncomment to activate arithmetic filters.
//   If arithmetic filters are activated,
//   status is set to needs_exact_predicates
//   whenever predicates could not be evaluated
//   using floating points on the GPU
// #define USE_ARITHMETIC_FILTER

#define IF_VERBOSE(x) //x

#ifndef NO_CUDA
void computeVoronoiFirstRing(std::vector <float> &, std::vector <uint32_t> &, std::vector <uint32_t> &, std::vector <float> &, std::vector <float> &, std::vector <uint32_t> &, bool, uint32_t);
#endif

#endif // __VORONOI_H__


