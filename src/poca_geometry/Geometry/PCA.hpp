/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PCA.hpp
*
* Copyright: Florian Levet (2020-2026)
*
* License:   LGPL v3
*/

#ifndef PCA_hpp__
#define PCA_hpp__

#include <cstddef>
#include <vector>

namespace poca::geometry {
	class PCA {
	public:
		struct Result {
			std::vector<std::vector<float>> embedding; // [sample][component]
			std::vector<std::vector<float>> loadings;  // [component][input feature]
			std::vector<float> explainedVariance;
			std::vector<float> explainedVarianceRatio;
			std::vector<float> means;
			std::vector<float> scales;
		};

		// Input layout: one row per sample, one column per feature.
		// dimensions must be 2 or 3. When standardize is true, each feature is
		// centered and divided by its sample standard deviation before PCA.
		static Result compute(const std::vector<std::vector<float>>& data,
			std::size_t dimensions = 2, bool standardize = true);

	private:
		PCA() = delete;
	};
}

#endif
