/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "DetectionSet.hpp"
#include <limits>
#include <memory>
#include <stdexcept>

namespace poca::geometry {
	void DetectionSet::initializeStorageBacked(size_t _count, uint32_t _dimension, size_t _slices,
		const poca::core::BoundingBox& _bbox, std::map<std::string, std::unique_ptr<poca::core::MyData>> _features)
	{
		if (!m_data.empty() || m_kdTree || !_count || _count > (std::numeric_limits<uint32_t>::max)() ||
			(_dimension != 2 && _dimension != 3) || !_slices)
			throw std::invalid_argument("Invalid storage-backed DetectionSet initialization");
		if (!_features.count("x") || !_features.count("y") || (_features.count("z") != (_dimension == 3)))
			throw std::invalid_argument("DetectionSet coordinate manifest does not match dimension");
		for (const auto& feature : _features)
			if (!feature.second || feature.second->nbElements() != _count)
				throw std::invalid_argument("DetectionSet feature count mismatch: " + feature.first);
		adoptPersistedFeatures(std::move(_features),_count,"x");
		m_nbPoints = _count;
		m_nbSlices = _slices;
		m_nbSelection = static_cast<unsigned int>(_count);
		m_bbox = _bbox;
		m_currentHistogram = "x";
		m_storageBacked = true;
	}

	void DetectionSet::ensureSpatialIndex() const
	{
		std::lock_guard<std::mutex> lock(m_spatialIndexMutex);
		if (m_kdTree) return;
		if (!m_nbPoints) throw std::runtime_error("Cannot index an empty DetectionSet");
		const auto& xs = getOriginalData<float>("x");
		const auto& ys = getOriginalData<float>("y");
		const auto* zs = hasData("z") ? &getOriginalData<float>("z") : nullptr;
		if (xs.size() != m_nbPoints || ys.size() != m_nbPoints || (zs && zs->size() != m_nbPoints))
			throw std::runtime_error("DetectionSet coordinate count mismatch while building spatial index");
		m_pointCloud.resize(m_nbPoints);
		for (size_t n = 0; n < m_nbPoints; ++n) m_pointCloud.m_pts[n].set(xs[n], ys[n], zs ? (*zs)[n] : 0.);
		auto tree = std::make_unique<KdTree_DetectionPoint>(3, m_pointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
		tree->buildIndex();
		m_kdTree = tree.release();
	}
}
