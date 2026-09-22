/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      UMAP.cpp
*
* Copyright: Florian Levet (2020-2026)
*
* License:   LGPL v3
*/

#include "UMAP.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <umappp/umappp.hpp>

namespace poca::geometry {
	namespace {
		void validateInput(const std::vector<std::vector<float>>& data, const UMAP::Options& options)
		{
			if (options.dimensions != 2 && options.dimensions != 3)
				throw std::invalid_argument("UMAP output dimensions must be 2 or 3.");
			if (data.size() < 3)
				throw std::invalid_argument("UMAP requires at least three samples.");
			if (data.front().empty())
				throw std::invalid_argument("UMAP requires at least one input feature.");
			if (options.numberOfNeighbors < 2)
				throw std::invalid_argument("UMAP numberOfNeighbors must be at least 2.");
			if (!std::isfinite(options.minDistance) || options.minDistance < 0.0)
				throw std::invalid_argument("UMAP minDistance must be finite and non-negative.");
			if (!std::isfinite(options.spread) || options.spread <= 0.0)
				throw std::invalid_argument("UMAP spread must be finite and strictly positive.");
			if (options.minDistance > options.spread)
				throw std::invalid_argument("UMAP minDistance must not exceed spread.");
			if (options.numberOfThreads < 1)
				throw std::invalid_argument("UMAP numberOfThreads must be at least 1.");

			const std::size_t featureCount = data.front().size();
			for (std::size_t sample = 0; sample < data.size(); ++sample) {
				if (data[sample].size() != featureCount)
					throw std::invalid_argument("UMAP input rows must all have the same number of features.");
				for (const float value : data[sample])
					if (!std::isfinite(value))
						throw std::invalid_argument("UMAP input contains a non-finite value.");
			}
		}
	}

	UMAP::Result UMAP::compute(const std::vector<std::vector<float>>& data)
	{
		return compute(data, Options{});
	}

	UMAP::Result UMAP::compute(const std::vector<std::vector<float>>& data, const Options& options)
	{
		validateInput(data, options);

		const std::size_t sampleCount = data.size();
		const std::size_t featureCount = data.front().size();
		const int usedNeighbors = (std::min)(options.numberOfNeighbors, static_cast<int>(sampleCount - 1));

		std::vector<double> means(featureCount, 0.0);
		std::vector<double> scales(featureCount, 1.0);
		for (std::size_t feature = 0; feature < featureCount; ++feature) {
			double sum = 0.0;
			for (std::size_t sample = 0; sample < sampleCount; ++sample)
				sum += static_cast<double>(data[sample][feature]);
			means[feature] = sum / static_cast<double>(sampleCount);
		}

		if (options.standardize) {
			const double denominator = static_cast<double>(sampleCount - 1);
			for (std::size_t feature = 0; feature < featureCount; ++feature) {
				double sumSquared = 0.0;
				for (std::size_t sample = 0; sample < sampleCount; ++sample) {
					const double centered = static_cast<double>(data[sample][feature]) - means[feature];
					sumSquared += centered * centered;
				}
				const double scale = std::sqrt((std::max)(0.0, sumSquared / denominator));
				if (std::isfinite(scale) && scale > std::numeric_limits<double>::epsilon())
					scales[feature] = scale;
			}
		}

		// umappp/knncolle expect a column-major matrix with input dimensions as
		// rows and observations as columns. Each observation is therefore stored
		// contiguously here as [feature0, feature1, ...].
		std::vector<double> input(featureCount * sampleCount, 0.0);
		for (std::size_t sample = 0; sample < sampleCount; ++sample) {
			for (std::size_t feature = 0; feature < featureCount; ++feature) {
				double value = static_cast<double>(data[sample][feature]) - means[feature];
				if (options.standardize)
					value /= scales[feature];
				input[sample * featureCount + feature] = value;
			}
		}

		knncolle::VptreeBuilder<int, double, double> builder(
			std::make_shared<knncolle::EuclideanDistance<double, double>>());

		umappp::Options umapOptions;
		umapOptions.num_neighbors = usedNeighbors;
		umapOptions.min_dist = options.minDistance;
		umapOptions.spread = options.spread;
		umapOptions.initialize_method = umappp::InitializeMethod::SPECTRAL;
		umapOptions.initialize_seed = static_cast<umappp::RngEngine::result_type>(options.seed);
		umapOptions.optimize_seed = static_cast<umappp::RngEngine::result_type>(options.seed);
		umapOptions.num_threads = options.numberOfThreads;

		// Keep spectral initialization and layout optimization deterministic.
		// For the organoid-scale datasets targeted here, parallelizing these two
		// steps is not useful; num_threads still applies to neighbor-related work.
		umapOptions.num_threads_spectral = 1;
		umapOptions.num_threads_optimize = 1;

		std::vector<double> embedding(sampleCount * options.dimensions, 0.0);
		try {
			auto status = umappp::initialize<int, double>(
				featureCount,
				static_cast<int>(sampleCount),
				input.data(),
				builder,
				options.dimensions,
				embedding.data(),
				umapOptions);
			status.run(embedding.data());
		}
		catch (const std::exception& error) {
			throw std::runtime_error(std::string("UMAP computation failed: ") + error.what());
		}

		Result result;
		result.embedding.assign(sampleCount, std::vector<float>(options.dimensions, 0.f));
		result.means.resize(featureCount, 0.f);
		result.scales.resize(featureCount, 1.f);
		result.numberOfNeighbors = usedNeighbors;

		for (std::size_t feature = 0; feature < featureCount; ++feature) {
			result.means[feature] = static_cast<float>(means[feature]);
			result.scales[feature] = static_cast<float>(scales[feature]);
		}

		for (std::size_t sample = 0; sample < sampleCount; ++sample)
			for (std::size_t component = 0; component < options.dimensions; ++component)
				result.embedding[sample][component] = static_cast<float>(embedding[sample * options.dimensions + component]);

		return result;
	}
}
