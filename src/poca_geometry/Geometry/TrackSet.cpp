/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      TrackSet.cpp
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

#include <General/MyData.hpp>

#include "TrackSet.hpp"

namespace poca::geometry {

	TrackSet::TrackSet(const std::vector <uint32_t>& _firsts, const std::vector <poca::core::Vec3mf>& _points, const std::vector <uint32_t>& _planes):poca::core::BasicComponent("TrackSet"), m_tracks(_points, _firsts), m_planes(_planes)
	{
		std::vector <float> lengthes(m_tracks.nbElements());
		for (uint32_t n = 0; n < m_tracks.nbElements(); n++) {
			float l = 0.f;
			for (uint32_t i = _firsts[n]; i < _firsts[n + 1] - 1; i++) {
				const poca::core::Vec3mf& p1 = _points[i], & p2 = _points[i + 1];
				l += p1.distance(p2);
			}
			lengthes[n] = l;
		}
		m_data["lengthes"] = new poca::core::MyData(lengthes);
		m_selection.resize(lengthes.size());
		setCurrentHistogramType("lengthes");
		forceRegenerateSelection();
	}

	TrackSet::~TrackSet()
	{

	}

	poca::core::BasicComponent* TrackSet::copy()
	{
		return new TrackSet(*this);
	}

	void TrackSet::generateTracks(std::vector <poca::core::Vec3mf>& _tracks) const
	{
		_tracks.clear();
		const auto& firsts = m_tracks.getFirstElements();
		const auto& points = m_tracks.getData();
		for (auto n = 0; n < m_tracks.nbElements(); n++) {
			_tracks.push_back(points[firsts[n]]);
			for (uint32_t i = firsts[n] + 1; i < firsts[n + 1] - 1; i++) {
				_tracks.push_back(points[i]);
				_tracks.push_back(points[i]);
			}
			_tracks.push_back(points[firsts[n + 1] - 1]);
		}
	}

	void TrackSet::generatePickingIndices(std::vector <float>& _ids) const
	{
		_ids.clear();
		const auto& firsts = m_tracks.getFirstElements();
		const auto& points = m_tracks.getData();
		for (auto n = 0; n < m_tracks.nbElements(); n++) {
			_ids.push_back(n + 1);
			for (uint32_t i = firsts[n] + 1; i < firsts[n + 1] - 1; i++) {
				_ids.push_back(n + 1);
				_ids.push_back(n + 1);
			}
			_ids.push_back(n + 1);
		}
	}

	void TrackSet::getFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		_features.clear();

		const auto& firsts = m_tracks.getFirstElements();
		const auto& points = m_tracks.getData();
		for (auto n = 0; n < m_tracks.nbElements(); n++) {
			float val = _selection[n] ? _values[n] : _notSelectedValue;
			_features.push_back(val);
			for (uint32_t i = firsts[n] + 1; i < firsts[n + 1] - 1; i++) {
				_features.push_back(val);
				_features.push_back(val);
			}
			_features.push_back(val);
		}
	}

	poca::core::BoundingBox TrackSet::computeBoundingBoxElement(const uint32_t _idx) const
	{
		poca::core::BoundingBox bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
		const auto& firsts = m_tracks.getFirstElements();
		const auto& points = m_tracks.getData();
		if (points.empty()) return bbox;
		for (unsigned int n = firsts[_idx]; n < firsts[_idx + 1]; n++) {
			bbox[0] = points[n].x() < bbox[0] ? points[n].x() : bbox[0];
			bbox[1] = points[n].y() < bbox[1] ? points[n].y() : bbox[1];
			bbox[2] = points[n].z() < bbox[2] ? points[n].z() : bbox[2];

			bbox[3] = points[n].x() > bbox[3] ? points[n].x() : bbox[3];
			bbox[4] = points[n].y() > bbox[4] ? points[n].y() : bbox[4];
			bbox[5] = points[n].z() > bbox[5] ? points[n].z() : bbox[5];
		}
		return bbox;
	}
}