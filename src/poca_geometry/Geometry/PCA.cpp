/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PCA.cpp
*
* Copyright: Florian Levet (2020-2026)
*
* License:   LGPL v3
*/

#include "PCA.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace poca::geometry {
	namespace {
		void validateInput(const std::vector<std::vector<float>>& data, const std::size_t dimensions)
		{
			if (dimensions != 2 && dimensions != 3)
				throw std::invalid_argument("PCA output dimensions must be 2 or 3.");
			if (data.size() < 2)
				throw std::invalid_argument("PCA requires at least two samples.");
			if (data.front().empty())
				throw std::invalid_argument("PCA requires at least one input feature.");

			const std::size_t featureCount = data.front().size();
			for (std::size_t sample = 0; sample < data.size(); ++sample) {
				if (data[sample].size() != featureCount)
					throw std::invalid_argument("PCA input rows must all have the same number of features.");
				for (const float value : data[sample])
					if (!std::isfinite(value))
						throw std::invalid_argument("PCA input contains a non-finite value.");
			}

			const std::size_t maxUsefulComponents = (std::min)(featureCount, data.size() - 1);
			if (dimensions > maxUsefulComponents)
				throw std::invalid_argument("PCA output dimensionality exceeds the rank supported by the input matrix.");
		}
	}

	PCA::Result PCA::compute(const std::vector<std::vector<float>>& data,
		const std::size_t dimensions, const bool standardize)
	{
		validateInput(data, dimensions);

		const std::size_t sampleCount = data.size();
		const std::size_t featureCount = data.front().size();

		Eigen::MatrixXd matrix(static_cast<Eigen::Index>(sampleCount), static_cast<Eigen::Index>(featureCount));
		for (std::size_t sample = 0; sample < sampleCount; ++sample)
			for (std::size_t feature = 0; feature < featureCount; ++feature)
				matrix(static_cast<Eigen::Index>(sample), static_cast<Eigen::Index>(feature)) = static_cast<double>(data[sample][feature]);

		const Eigen::RowVectorXd means = matrix.colwise().mean();
		matrix.rowwise() -= means;

		Eigen::RowVectorXd scales = Eigen::RowVectorXd::Ones(static_cast<Eigen::Index>(featureCount));
		if (standardize) {
			const double denominator = static_cast<double>(sampleCount - 1);
			for (std::size_t feature = 0; feature < featureCount; ++feature) {
				const Eigen::Index column = static_cast<Eigen::Index>(feature);
				const double variance = matrix.col(column).squaredNorm() / denominator;
				const double scale = std::sqrt((std::max)(0.0, variance));
				if (std::isfinite(scale) && scale > std::numeric_limits<double>::epsilon()) {
					scales(column) = scale;
					matrix.col(column) /= scale;
				}
				else {
					// Constant features contain no PCA information. Keeping a scale of
					// one leaves their centered column equal to zero.
					scales(column) = 1.0;
				}
			}
		}

		const Eigen::MatrixXd covariance = (matrix.transpose() * matrix) / static_cast<double>(sampleCount - 1);
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance);
		if (solver.info() != Eigen::Success)
			throw std::runtime_error("PCA eigendecomposition failed.");

		const Eigen::VectorXd eigenvalues = solver.eigenvalues(); // ascending
		const Eigen::MatrixXd eigenvectors = solver.eigenvectors();

		double totalVariance = 0.0;
		for (Eigen::Index i = 0; i < eigenvalues.size(); ++i)
			totalVariance += (std::max)(0.0, eigenvalues(i));

		Eigen::MatrixXd selected(static_cast<Eigen::Index>(featureCount), static_cast<Eigen::Index>(dimensions));
		for (std::size_t component = 0; component < dimensions; ++component) {
			const Eigen::Index sourceColumn = static_cast<Eigen::Index>(featureCount - 1 - component);
			selected.col(static_cast<Eigen::Index>(component)) = eigenvectors.col(sourceColumn);
		}
		const Eigen::MatrixXd scores = matrix * selected;

		Result result;
		result.embedding.assign(sampleCount, std::vector<float>(dimensions, 0.f));
		result.loadings.assign(dimensions, std::vector<float>(featureCount, 0.f));
		result.explainedVariance.resize(dimensions, 0.f);
		result.explainedVarianceRatio.resize(dimensions, 0.f);
		result.means.resize(featureCount, 0.f);
		result.scales.resize(featureCount, 1.f);

		for (std::size_t feature = 0; feature < featureCount; ++feature) {
			const Eigen::Index index = static_cast<Eigen::Index>(feature);
			result.means[feature] = static_cast<float>(means(index));
			result.scales[feature] = static_cast<float>(scales(index));
		}

		for (std::size_t component = 0; component < dimensions; ++component) {
			const Eigen::Index sourceColumn = static_cast<Eigen::Index>(featureCount - 1 - component);
			const double variance = (std::max)(0.0, eigenvalues(sourceColumn));
			result.explainedVariance[component] = static_cast<float>(variance);
			result.explainedVarianceRatio[component] = totalVariance > 0.0 ? static_cast<float>(variance / totalVariance) : 0.f;
			for (std::size_t feature = 0; feature < featureCount; ++feature)
				result.loadings[component][feature] = static_cast<float>(selected(static_cast<Eigen::Index>(feature), static_cast<Eigen::Index>(component)));
		}

		for (std::size_t sample = 0; sample < sampleCount; ++sample)
			for (std::size_t component = 0; component < dimensions; ++component)
				result.embedding[sample][component] = static_cast<float>(scores(static_cast<Eigen::Index>(sample), static_cast<Eigen::Index>(component)));

		return result;
	}
}
