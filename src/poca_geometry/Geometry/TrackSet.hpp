/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      TrackSet.hpp
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

#ifndef TrackSet_h__
#define TrackSet_h__

#include <General/BasicComponent.hpp>
#include <General/MyArray.hpp>
#include <General/Vec3.hpp>

namespace poca::geometry {

	class TrackSet : public poca::core::BasicComponent {
	public:
		TrackSet(const std::vector <uint32_t>&, const std::vector <poca::core::Vec3mf>&);
		TrackSet(const std::vector <uint32_t>&, const std::vector <poca::core::Vec3mf>&, const std::vector <float>&, const std::vector <uint32_t>&);
		~TrackSet();

		BasicComponentInterface* copy();

		const poca::core::MyArrayVec3mf& getTracks() const { return m_tracks; }
		const std::vector <float>& getTimes() const { return m_planes; }

		const uint32_t nbTracks() const { return m_tracks.nbElements(); }
		const std::vector <uint32_t>& locsToTrackID() const { return m_locsToTrackID; }

		void generateTracks(std::vector <poca::core::Vec3mf>&) const;
		void generatePickingIndices(std::vector <float>&) const;
		void getFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const;

		poca::core::BoundingBox computeBoundingBoxElement(const uint32_t) const;

		inline const bool hasMsds() const { return !m_msds.empty(); }
		inline const std::vector < std::vector <float>>& getMsds() const { return m_msds; }
		inline const std::vector <float>& getMsd(const uint32_t _index) const { return m_msds[_index]; }
		const uint32_t dimension() const { return m_dimension; }

	protected:
		poca::core::MyArrayVec3mf m_tracks;
		std::vector <float> m_planes;
		std::vector <uint32_t> m_locsToTrackID, m_tracksID;
		std::vector < std::vector <float>> m_msds;
		uint32_t m_dimension;
	};

}

#endif