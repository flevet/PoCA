/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Skeleton.hpp
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

#ifndef Skeleton_h__
#define Skeleton_h__

#include <General/BasicComponent.hpp>
#include <General/MyArray.hpp>
#include <General/Vec3.hpp>

namespace poca::geometry {

	class Skeleton : public poca::core::BasicComponent {
	public:
		static float MERGE_DISTANCE;

		Skeleton(const std::vector <uint32_t>&, const std::vector <poca::core::Vec3mf>&);
		~Skeleton();

		BasicComponentInterface* copy();

		const poca::core::MyArrayVec3mf& getSegments() const { return m_segments; }
		poca::core::MyArrayVec3mf& getSegments() { return m_segments; }

		const uint32_t nbSegments() const { return m_segments.nbElements(); }

		void generateSegments(std::vector <poca::core::Vec3mf>&) const;
		void generatePickingIndicesSegments(std::vector <float>&) const;
		void generatePickingIndicesSkeletons(std::vector <float>&) const;
		void generatePickingIndices(std::vector <float>&, const std::vector<float>&) const;
		void getFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const;

		void saveAsSkel(const std::string& _filename);

		const poca::core::BoundingBox& boundingBoxSegment(const uint32_t _id) const { return m_bboxSegments[_id]; }
		const poca::core::BoundingBox& boundingBoxSkeleton(const uint32_t _id) const { return m_bboxSkeletons[_id]; }

		const uint32_t dimension() const { return m_dimension; }

	protected:
		void mergeSegmentsToSkeletons();
		void computeBBoxes();

	protected:
		poca::core::MyArrayVec3mf m_segments;
		uint32_t m_dimension;

		std::vector <float> m_segmentToSkeleton;
		std::vector <std::set <uint32_t>> m_skeletons;

		std::vector <poca::core::BoundingBox> m_bboxSegments, m_bboxSkeletons;

		//TODO: add graphs
	};

}

#endif