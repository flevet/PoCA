/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#ifndef HistogramStorageStatistics_hpp__
#define HistogramStorageStatistics_hpp__

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include "ArrayStatistics.hpp"

namespace poca::core {
	inline void storageDisplayInterval(float& _min, float& _max)
	{
		if (!std::isfinite(_min) || !std::isfinite(_max) || _min > _max)
			throw std::invalid_argument("Invalid storage-backed display interval");
		if (_min != _max) return;
		const double value = _min;
		const double margin = std::max(1., std::abs(value) * 1.e-3);
		const double limit = (std::numeric_limits<float>::max)();
		_min = static_cast<float>(std::max(-limit, value - margin));
		_max = static_cast<float>(std::min(limit, value + margin));
	}
	// Small coarse-level samples are independent of the full-resolution values.
	// This initialization is CPU-only, including on systems without CUDA.
	template <class T>
	ArrayStatistics storageSampleStatistics(const std::vector<T>& _sample)
	{
		std::vector<float> values(STATS_NB_PARAMS);
		// Preserve the existing storage-sample even-median convention.
		computeStats_CPU(_sample, values, true);
		return ArrayStatistics(values);
	}

	template <class T>
	void storageSampleBins(const std::vector<T>& _sample, std::vector<float>& _bins,
		std::vector<float>& _ts, std::size_t _count, float _min, float _max, float& _step, float& _maxY)
	{
		if (_count < 2 || !std::isfinite(_min) || !std::isfinite(_max) || _min > _max)
			throw std::invalid_argument("Invalid storage-backed histogram bounds or bin count");
		_bins.assign(_count, 0.f);
		_ts.resize(_count);
		const double step = (static_cast<double>(_max) - _min) / (_count - 1);
		_step = static_cast<float>(step);
		for (T value : _sample) {
			const double v = static_cast<double>(value);
			if (!std::isfinite(v)) continue;
			if (v < _min || v > _max) continue;
			const std::size_t bin = step == 0. ? 0 : std::min(_count - 1, static_cast<std::size_t>((v - _min) / step));
			_bins[bin] += 1.f;
		}
		_maxY = 0.f;
		for (std::size_t i = 0; i < _count; ++i) {
			_ts[i] = static_cast<float>(std::min(static_cast<double>(_max), _min + i * step + 0.5 * step));
			_maxY = std::max(_maxY, _bins[i]);
		}
	}
}
#endif
