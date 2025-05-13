/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Skeleton.cpp
*
* Copyright: Florian Levet (2020-2025)
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

#include <General/MyData.hpp>
#include <General/Misc.h>
#include <Fit/lmcurve.h>

#include <Geometry/dbscan.h>
#include <Geometry/nanoflann.hpp>

#include "Skeleton.hpp"

namespace poca::geometry {

	float Skeleton::MERGE_DISTANCE = 0.001f;

	Skeleton::Skeleton(const std::vector <uint32_t>& _firsts, const std::vector <poca::core::Vec3mf>& _points) :poca::core::BasicComponent("Skeleton"), m_segments(_points, _firsts)
	{
		float firstZ = _points[0].z();
		bool allSames = std::all_of(_points.begin(), _points.end(), [firstZ](poca::core::Vec3mf p) { return p.z() == firstZ; });
		m_dimension = allSames ? 2 : 3;

		std::vector <float> lengthes(m_segments.nbElements()), nbLocs(m_segments.nbElements()), ids(m_segments.nbElements());

		for (uint32_t n = 0; n < m_segments.nbElements(); n++) {
			float l = 0.f;
			for (uint32_t i = _firsts[n]; i < _firsts[n + 1] - 1; i++) {
				const poca::core::Vec3mf& p1 = _points[i], & p2 = _points[i + 1];
				auto d = p1.distance(p2);
				l += d;
			}
			lengthes[n] = l;
			nbLocs[n] = _firsts[n + 1] - _firsts[n];
			ids[n] = n + 1;

		}

		mergeSegmentsToSkeletons();
		computeBBoxes();

		m_data["lengthes"] = poca::core::generateDataWithLog(lengthes);
		m_data["ids"] = poca::core::generateDataWithLog(ids);
		m_data["skeletonIds"] = poca::core::generateDataWithLog(m_segmentToSkeleton);
		m_data["nb"] = poca::core::generateDataWithLog(nbLocs);
		m_selection.resize(lengthes.size());
		setCurrentHistogramType("skeletonIds");
		forceRegenerateSelection();
	}

	Skeleton::~Skeleton()
	{

	}

	void Skeleton::mergeSegmentsToSkeletons()
	{
		//Merge all segments that are connected and part of the same skeleton
		const std::vector <poca::core::Vec3mf>& points = m_segments.getData();
		const std::vector <uint32_t>& firsts = m_segments.getFirstElements();
		KdPointCloud_3D_D cloud;
		cloud.m_pts.resize(m_segments.nbElements() * 2);
		for (auto n = 0, cptt = 0; n < m_segments.nbElements(); n++) {
			auto ids = { firsts[n], firsts[n + 1] - 1 };
			for (auto id : ids) {
				cloud.m_pts[cptt].m_x = points[id].x();
				cloud.m_pts[cptt].m_y = points[id].y();
				cloud.m_pts[cptt++].m_z = points[id].z();
			}
		}
		const double DOUBLE_EPSILON = 0.001;
		KdTree_3D_double kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
		kdtree.buildIndex();
		const double search_radius = static_cast<double>(DOUBLE_EPSILON * DOUBLE_EPSILON);
		std::vector< std::pair< std::size_t, double > > ret_matches;
		nanoflann::SearchParams params;
		std::size_t nMatches;
		std::vector<bool> done(cloud.m_pts.size(), false);
		std::vector <uint32_t> queue = { 0 };
		bool allDone = std::count(done.begin(), done.end(), true) == done.size();
		uint32_t currentSkeleton = 1;
		m_segmentToSkeleton.resize(m_segments.nbElements());
		m_skeletons.push_back(std::set <uint32_t>());
		do {
			std::vector <uint32_t> queue;
			size_t sizeQueue = 0, current = 0;
			for (auto n = 0; n < done.size() && queue.empty(); n++) {
				if (!done[n]) {
					queue.push_back(n);
					sizeQueue = queue.size();
					done[n] = true;
				}
			}
			while (current < sizeQueue) {
				auto id = queue[current];
				uint32_t idSegment = floor(id / 2);
				m_segmentToSkeleton[idSegment] = currentSkeleton;
				m_skeletons.back().insert(idSegment);
				uint32_t ptsSegment[] = { 2 * idSegment , 2 * idSegment + 1 };
				for (auto ptSeg : ptsSegment) {
					if (!done[ptSeg]) {
						queue.push_back(ptSeg);
						done[ptSeg] = true;
					}
				}

				const double queryPt[3] = { cloud.m_pts[id].m_x, cloud.m_pts[id].m_y, cloud.m_pts[id].m_z };
				nMatches = kdtree.radiusSearch(&queryPt[0], search_radius, ret_matches, params);
				for (size_t n = 0; n < nMatches; n++) {
					if (!done[ret_matches[n].first])
						queue.push_back(ret_matches[n].first);
					done[ret_matches[n].first] = true;
				}
				sizeQueue = queue.size();
				current++;
			}
			allDone = std::count(done.begin(), done.end(), true) == done.size();
			if (!allDone) {
				currentSkeleton++;
				m_skeletons.push_back(std::set <uint32_t>());
			}
		} while (!allDone);
	}

	void Skeleton::computeBBoxes()
	{
		m_bboxSegments.resize(m_segments.nbElements());
		m_bboxSkeletons.resize(m_skeletons.size());
		std::fill(m_bboxSegments.begin(), m_bboxSegments.end(), poca::core::BoundingBox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX));
		std::fill(m_bboxSkeletons.begin(), m_bboxSkeletons.end(), poca::core::BoundingBox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX));
		const auto& firsts = m_segments.getFirstElements();
		const auto& points = m_segments.getData();
		for (auto cur = 0; cur < m_segments.nbElements(); cur++) {
			poca::core::BoundingBox& bbox = m_bboxSegments[cur];
			for (unsigned int n = firsts[cur]; n < firsts[cur + 1]; n++) {
				bbox[0] = points[n].x() < bbox[0] ? points[n].x() : bbox[0];
				bbox[1] = points[n].y() < bbox[1] ? points[n].y() : bbox[1];
				bbox[2] = points[n].z() < bbox[2] ? points[n].z() : bbox[2];

				bbox[3] = points[n].x() > bbox[3] ? points[n].x() : bbox[3];
				bbox[4] = points[n].y() > bbox[4] ? points[n].y() : bbox[4];
				bbox[5] = points[n].z() > bbox[5] ? points[n].z() : bbox[5];
			}
		}

		for (auto n = 0; n < m_skeletons.size(); n++) {
			poca::core::BoundingBox& bbox = m_bboxSkeletons[n];
			for (auto idSegment : m_skeletons[n]) {
				const poca::core::BoundingBox& bboxSegment = m_bboxSegments[idSegment];
				bbox[0] = bboxSegment[0] < bbox[0] ? bboxSegment[0] : bbox[0];
				bbox[1] = bboxSegment[1] < bbox[1] ? bboxSegment[1] : bbox[1];
				bbox[2] = bboxSegment[2] < bbox[2] ? bboxSegment[2] : bbox[2];

				bbox[3] = bboxSegment[3] > bbox[3] ? bboxSegment[3] : bbox[3];
				bbox[4] = bboxSegment[4] > bbox[4] ? bboxSegment[4] : bbox[4];
				bbox[5] = bboxSegment[5] > bbox[5] ? bboxSegment[5] : bbox[5];
			}
		}
	}

	poca::core::BasicComponentInterface* Skeleton::copy()
	{
		return new Skeleton(*this);
	}

	void Skeleton::generateSegments(std::vector <poca::core::Vec3mf>& _segments) const
	{
		_segments.clear();
		const auto& firsts = m_segments.getFirstElements();
		const auto& points = m_segments.getData();
		for (auto n = 0; n < m_segments.nbElements(); n++) {
			_segments.push_back(points[firsts[n]]);
			for (uint32_t i = firsts[n] + 1; i < firsts[n + 1] - 1; i++) {
				_segments.push_back(points[i]);
				_segments.push_back(points[i]);
			}
			_segments.push_back(points[firsts[n + 1] - 1]);
		}
	}

	void Skeleton::generatePickingIndicesSegments(std::vector <float>& _ids) const
	{
		const std::vector <float>& ids = this->getData<float>("ids");
		generatePickingIndices(_ids, ids);
	}

	void Skeleton::generatePickingIndicesSkeletons(std::vector <float>& _ids) const
	{
		const std::vector <float>& ids = this->getData<float>("skeletonIds");
		generatePickingIndices(_ids, ids);
	}

	void Skeleton::generatePickingIndices(std::vector <float>& _ids, const std::vector <float>& _idsFeature) const
	{
		_ids.clear();
		const auto& firsts = m_segments.getFirstElements();
		const auto& points = m_segments.getData();
		for (auto n = 0; n < m_segments.nbElements(); n++) {
			auto trackId = _idsFeature[n];
			_ids.push_back(trackId);
			for (uint32_t i = firsts[n] + 1; i < firsts[n + 1] - 1; i++) {
				_ids.push_back(trackId);
				_ids.push_back(trackId);
			}
			_ids.push_back(trackId);
		}
	}

	void Skeleton::getFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		_features.clear();

		const auto& firsts = m_segments.getFirstElements();
		const auto& points = m_segments.getData();
		for (auto n = 0; n < m_segments.nbElements(); n++) {
			float val = _selection[n] ? _values[n] : _notSelectedValue;
			_features.push_back(val);
			for (uint32_t i = firsts[n] + 1; i < firsts[n + 1] - 1; i++) {
				_features.push_back(val);
				_features.push_back(val);
			}
			_features.push_back(val);
		}
	}

	void Skeleton::saveAsSkel(const std::string& _filename)
	{
		std::ofstream fs(_filename, std::ifstream::binary);
		poca::core::MyArrayVec3mf& segments = getSegments();
		std::vector <poca::core::Vec3mf>& points = segments.getData();
		std::vector <uint32_t>& firsts = segments.getFirstElements();
		size_t nb = firsts.size();
		fs.write(reinterpret_cast<char*>(&nb), sizeof(size_t));
		fs.write(reinterpret_cast<char*>(firsts.data()), nb * sizeof(uint32_t));
		nb = points.size();
		fs.write(reinterpret_cast<char*>(&nb), sizeof(size_t));
		fs.write(reinterpret_cast<char*>(points.data()), nb * sizeof(poca::core::Vec3mf));
		fs.close();
	}
}

