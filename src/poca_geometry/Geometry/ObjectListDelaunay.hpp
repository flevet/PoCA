/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListDelaunay.hpp
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

#ifndef ObjectListDelaunay_hpp__
#define ObjectListDelaunay_hpp__

#include <any>
#include <tuple>

#include <General/BasicComponentList.hpp>
#include <General/MyArray.hpp>
#include <General/Vec3.hpp>
#include <Interfaces/ObjectListInterface.hpp>

namespace poca::geometry {
	class ObjectListDelaunay : public poca::geometry::ObjectListInterface {
	public:
		ObjectListDelaunay(const float* , const float*, const float*, const std::vector <uint32_t>&, const std::vector <uint32_t>&, const std::vector <poca::core::Vec3mf>&, const std::vector <uint32_t>&, const std::vector <poca::core::Vec3mf>&, const std::vector <uint32_t>&, const std::vector <uint32_t>&);
		ObjectListDelaunay(const float*, const float*, const float*, const std::vector <uint32_t>&, const std::vector <uint32_t>&, const std::vector <poca::core::Vec3mf>&, const std::vector <uint32_t>&, const std::vector <float>&, const std::vector <uint32_t>&, const std::vector <uint32_t>&, const std::vector <uint32_t>&, const std::vector <poca::core::Vec3mf>& = std::vector <poca::core::Vec3mf>());
		~ObjectListDelaunay();

		poca::core::BasicComponentInterface* copy();

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

		inline const std::vector <uint32_t>& getLinkTriangulationFacesToObjects() const { return m_linkTriangulationFacesToObjects; }
		const float* getXs() const { return m_xs; }
		const float* getYs() const { return m_ys; }
		const float* getZs() const { return m_zs; }

		inline const uint32_t dimension() const { return m_zs == NULL ? 2 : 3; }
		inline const size_t nbObjects() const { return m_locs.nbElements(); }

		bool hasSkeletons() const { return false; }

		virtual void generateOutlineLocs(std::vector <poca::core::Vec3mf>&);
		virtual void getOutlineLocsFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const;
		virtual void getOutlineLocsFeatureInSelectionHiLow(std::vector <float>&, const std::vector <bool>&, const float, const float) const;

		virtual void setLocs(const float*, const float*, const float*, const std::vector <uint32_t>&, const std::vector <uint32_t>&);

	protected:
		const float* m_xs, * m_ys, * m_zs;
		std::vector <uint32_t> m_linkTriangulationFacesToObjects;
	};
}

#endif

