/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      UMAP.hpp
*
* Copyright: Florian Levet (2020-2026)
*
* License:   LGPL v3
*/

#ifndef UMAP_hpp__
#define UMAP_hpp__

#include <cstddef>
#include <cstdint>
#include <vector>

namespace poca::geometry {
	class UMAP {
	public:
		struct Options {
			std::size_t dimensions = 2;
			int numberOfNeighbors = 15;
			double minDistance = 0.1;
			double spread = 1.0;
			std::uint64_t seed = 42;
			int numberOfThreads = 1;
			bool standardize = true;
		};

		struct Result {
			std::vector<std::vector<float>> embedding; // [sample][component]
			std::vector<float> means;
			std::vector<float> scales;
			int numberOfNeighbors = 0; // actual value used after clamping to N - 1
		};

		// Input layout: one row per sample, one column per feature.
		// The default uses exact Euclidean kNN (knncolle VP-tree), spectral
		// initialization and the pinned libscran/umappp implementation.
		static Result compute(const std::vector<std::vector<float>>& data);
		static Result compute(const std::vector<std::vector<float>>& data, const Options& options);

	private:
		UMAP() = delete;
	};
}

#endif
