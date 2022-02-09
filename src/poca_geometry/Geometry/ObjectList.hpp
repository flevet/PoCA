/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectList.hpp
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

#ifndef ObjectList_hpp__
#define ObjectList_hpp__

#include <any>

#include <General/BasicComponent.hpp>
#include <General/MyArray.hpp>
#include <General/Vec3.hpp>
#include <Interfaces/MyObjectInterface.hpp>

namespace poca::geometry {
	class ObjectList : public poca::core::BasicComponent {
	public:
		ObjectList(const float* , const float*, const float*, const std::vector <uint32_t>&, const std::vector <uint32_t>&, const std::vector <poca::core::Vec3mf>&, const std::vector <uint32_t>&, const std::vector <poca::core::Vec3mf>&, const std::vector <uint32_t>&, const std::vector <uint32_t>&);
		ObjectList(const float*, const float*, const float*, const std::vector <uint32_t>&, const std::vector <uint32_t>&, const std::vector <poca::core::Vec3mf>&, const std::vector <uint32_t>&, const std::vector <float>&, const std::vector <uint32_t>&, const std::vector <uint32_t>&, const std::vector <uint32_t>&, const std::vector <poca::core::Vec3mf>& = std::vector <poca::core::Vec3mf>());
		~ObjectList();

		poca::core::BasicComponent* copy();

		virtual void generateLocs(std::vector <poca::core::Vec3mf>&);
		virtual void generateNormalLocs(std::vector <poca::core::Vec3mf>&);
		virtual void getLocsFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const;
		virtual void getLocsFeatureInSelectionHiLow(std::vector <float>&, const std::vector <bool>&, const float, const float) const;
		virtual void getOutlinesFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const;
		virtual void getOutlinesFeatureInSelectionHiLow(std::vector <float>&, const std::vector <bool>&, const float, const float) const;
		virtual void generateLocsPickingIndices(std::vector <float>&) const;

		virtual void generateTriangles(std::vector <poca::core::Vec3mf>&);
		virtual void generateNormals(std::vector <poca::core::Vec3mf>&);
		virtual void generateOutlines(std::vector <poca::core::Vec3mf>&);
		virtual void getFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const;
		virtual void getFeatureInSelectionHiLow(std::vector <float>&, const std::vector <bool>&, const float, const float) const;
		virtual void generatePickingIndices(std::vector <float>&) const;
		virtual poca::core::BoundingBox computeBoundingBoxElement(const int) const;
		virtual poca::core::Vec3mf computeBarycenterElement(const int) const;

		inline const poca::core::MyArrayUInt32& getLocsObjects() const { return m_locs; }
		inline const poca::core::MyArrayVec3mf& getTrianglesObjects() const { return m_triangles; }
		inline const poca::core::MyArrayVec3mf& getOutlinesObjects() const { return m_outlines; }
		inline const std::vector <poca::core::Vec3mf>& getNormalOutlineLocs() const { return m_normalOutlineLocs; }
		inline const std::vector <uint32_t>& getLinkTriangulationFacesToObjects() const { return m_linkTriangulationFacesToObjects; }
		inline const float* getXs() const { return m_xs; }
		inline const float* getYs() const { return m_ys; }
		inline const float* getZs() const { return m_zs; }

		inline const uint32_t dimension() const { return m_zs == NULL ? 2 : 3; }
		inline const size_t nbObjects() const { return m_locs.nbElements(); }

		inline void setOutlineLocs(const std::vector <uint32_t>& _locsOutlines, const std::vector <uint32_t>& _firstLocsOutlines) { m_outlineLocs = poca::core::MyArrayUInt32(_locsOutlines, _firstLocsOutlines); }
		inline const poca::core::MyArrayUInt32& getLocOutlines() const { return m_outlineLocs; }
		virtual void generateOutlineLocs(std::vector <poca::core::Vec3mf>&);
		virtual void getOutlineLocsFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const;
		virtual void getOutlineLocsFeatureInSelectionHiLow(std::vector <float>&, const std::vector <bool>&, const float, const float) const;

		virtual const std::vector < std::array<poca::core::Vec3mf, 3>>& getAxisObjects() const { return m_axis; }

	protected:
		const float* m_xs, * m_ys, * m_zs;
		poca::core::MyArrayUInt32 m_locs, m_outlineLocs;
		poca::core::MyArrayVec3mf m_triangles, m_outlines;
		std::vector <poca::core::Vec3mf> m_normalOutlineLocs;
		std::vector <uint32_t> m_linkTriangulationFacesToObjects;
		std::vector < std::array<poca::core::Vec3mf, 3>> m_axis;
	};
}

#endif

