/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      TrackSet.cpp
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

#include "TrackSet.hpp"

namespace poca::geometry {

	TrackSet::TrackSet(const std::vector <uint32_t>& _firsts, const std::vector <poca::core::Vec3mf>& _points) :poca::core::BasicComponent("TrackSet"), m_tracks(_points, _firsts)
	{
		float firstZ = _points[0].z();
		bool allSames = std::all_of(_points.begin(), _points.end(), [firstZ](poca::core::Vec3mf p) { return p.z() == firstZ; });
		m_dimension = allSames ? 2 : 3;

		std::vector <float> lengthes(m_tracks.nbElements()), nbLocs(m_tracks.nbElements()), ids(m_tracks.nbElements());

		for (uint32_t n = 0; n < m_tracks.nbElements(); n++) {
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
		m_data["lengthes"] = poca::core::generateDataWithLog(lengthes);
		m_data["ids"] = poca::core::generateDataWithLog(ids);
		m_data["nb"] = poca::core::generateDataWithLog(nbLocs);
		m_selection.resize(lengthes.size());
		setCurrentHistogramType("lengthes");
		forceRegenerateSelection();
	}

	TrackSet::TrackSet(const std::vector <uint32_t>& _firsts, const std::vector <poca::core::Vec3mf>& _points, const std::vector <float>& _planes, const std::vector <uint32_t>& _locsToTrackID) :poca::core::BasicComponent("TrackSet"), m_tracks(_points, _firsts), m_planes(_planes), m_locsToTrackID(_locsToTrackID)
	{
		float firstZ = _points[0].z();
		bool allSames = std::all_of(_points.begin(), _points.end(), [firstZ](poca::core::Vec3mf p) { return p.z() == firstZ; });
		m_dimension = allSames ? 2 : 3;

		float dxyz = 0.f, dxy = 0.f;
		std::vector <float> lengthes(m_tracks.nbElements()), time(m_tracks.nbElements()), nbLocs(m_tracks.nbElements()), msds(m_tracks.nbElements());
		std::vector <float> ids(m_tracks.nbElements()), diffusionCoeff(m_tracks.nbElements());
		m_msds.resize(m_tracks.nbElements());
		for (uint32_t n = 0; n < m_tracks.nbElements(); n++) {
			float l = 0.f;
			for (uint32_t i = _firsts[n]; i < _firsts[n + 1] - 1; i++) {
				const poca::core::Vec3mf& p1 = _points[i], & p2 = _points[i + 1];
				auto d = p1.distance(p2);
				l += d;

				auto dx = p2[0] - p1[0], dy = p2[1] - p1[1], dz = p2[2] - p1[2];
				auto d2 = sqrt(dx * dx + dy * dy);
				if (d > dxyz)
					dxyz = d;
				if (d2 > dxy)
					dxy = d2;
			}
			lengthes[n] = l;
			nbLocs[n] = _firsts[n + 1] - _firsts[n];
			time[n] = _planes[_firsts[n]];
			ids[n] = m_locsToTrackID[_firsts[n]];

			//Compute msd
			m_msds[n].resize(nbLocs[n] - 1);
			uint32_t inter = 1;
			for (uint32_t j = 0; j < m_msds[n].size(); j++) {
				m_msds[n][j] = 0.f;
				int iprec = 0;
				float cpt = 0.f;
				uint32_t startId = _firsts[n];
				for (uint32_t i = inter; i < nbLocs[n]; i++, iprec++) {
					const poca::core::Vec3mf& p1 = _points[startId + i], & p2 = _points[startId + iprec];
					auto dx = p2[0] - p1[0], dy = p2[1] - p1[1], dz = p2[2] - p1[2];
					float d = sqrt(dx * dx + dy * dy + dz * dz);
					m_msds[n][j] += d;
					cpt = cpt + 1.f;
				}
				inter++;
				m_msds[n][j] /= cpt;
			}

			/*float msd = 0.f, nbs = _firsts[n + 1] - _firsts[n];
			const auto& p0 = _points[_firsts[n]];
			for (uint32_t i = _firsts[n] + 1; i < _firsts[n + 1] - 1; i++) {
				const auto& p = _points[i];
				auto d = p.distance(p0);
				msd += d / nbs;
			}
			msds[n] = msd;
			meanDistance[n] = l / nbs;*/

			uint32_t stop = m_msds[n].size() > 4 ? 4 : m_msds[n].size();
			std::vector <double> v(stop), x(stop);
			for (uint32_t i = 0; i < stop; i++) {
				v[i] = m_msds[n][i];
				x[i] = i + 1;
			}
			lm_control_struct control = lm_control_double;
			lm_status_struct status;
			control.verbosity = 9;
			int nbParamEqn = 2;
			std::vector<double> paramsEqn(nbParamEqn);
			paramsEqn[0] = m_msds[n][m_msds[n].size()/2];
			lmcurve(nbParamEqn, paramsEqn.data(), x.size(), x.data(), v.data(), &poca::core::linear, &control, &status);
			diffusionCoeff[n] = paramsEqn[0];

		}
		m_data["lengthes"] = poca::core::generateDataWithLog(lengthes);
		m_data["ids"] = poca::core::generateDataWithLog(ids);
		m_data["appearing time"] = poca::core::generateDataWithLog(time);
		m_data["nb"] = poca::core::generateDataWithLog(nbLocs);
		m_data["diffusion coefficient"] = poca::core::generateDataWithLog(diffusionCoeff);
		m_selection.resize(lengthes.size());
		setCurrentHistogramType("lengthes");
		forceRegenerateSelection();

		std::cout << "max dxyz = " << dxyz << ", max dxy = " << dxy << std::endl;
		for (const auto& p : _points)
			m_bbox.addPointBBox(p.x(), p.y(), p.z());
	}

	TrackSet::~TrackSet()
	{

	}

	poca::core::BasicComponentInterface* TrackSet::copy()
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
		const auto& ids = this->getData<float>("ids");
		const auto& firsts = m_tracks.getFirstElements();
		const auto& points = m_tracks.getData();
		for (auto n = 0; n < m_tracks.nbElements(); n++) {
			auto trackId = ids[n];
			_ids.push_back(trackId);
			for (uint32_t i = firsts[n] + 1; i < firsts[n + 1] - 1; i++) {
				_ids.push_back(trackId);
				_ids.push_back(trackId);
			}
			_ids.push_back(trackId);
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