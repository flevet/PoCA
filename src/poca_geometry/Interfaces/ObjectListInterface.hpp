/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListInterface.hpp
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

#ifndef ObjectListInterface_hpp__
#define ObjectListInterface_hpp__

#include <any>
#include <tuple>

#include <General/BasicComponentList.hpp>
#include <General/MyArray.hpp>
#include <General/Vec3.hpp>
#include <Interfaces/MyObjectInterface.hpp>

namespace poca::geometry {
	class ObjectListInterface : public poca::core::BasicComponent {
	public:
		virtual ~ObjectListInterface() = default;

		virtual poca::core::BasicComponentInterface* copy() = 0;

		virtual void generateLocs(std::vector <poca::core::Vec3mf>&) = 0;
		virtual void generateNormalLocs(std::vector <poca::core::Vec3mf>&) = 0;
		virtual void getLocsFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const = 0;
		virtual void getLocsFeatureInSelectionHiLow(std::vector <float>&, const std::vector <bool>&, const float, const float) const = 0;
		virtual void getOutlinesFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const = 0;
		virtual void getOutlinesFeatureInSelectionHiLow(std::vector <float>&, const std::vector <bool>&, const float, const float) const = 0;
		virtual void generateLocsPickingIndices(std::vector <float>&) const = 0;

		virtual void generateTriangles(std::vector <poca::core::Vec3mf>&) = 0;
		virtual void generateNormals(std::vector <poca::core::Vec3mf>&) = 0;
		virtual void generateOutlines(std::vector <poca::core::Vec3mf>&) = 0;
		virtual void getFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const = 0;
		virtual void getFeatureInSelectionHiLow(std::vector <float>&, const std::vector <bool>&, const float, const float) const = 0;
		virtual void generatePickingIndices(std::vector <float>&) const = 0;
		virtual poca::core::BoundingBox computeBoundingBoxElement(const int) const = 0;
		virtual poca::core::Vec3mf computeBarycenterElement(const int) const = 0;

		inline const poca::core::MyArrayUInt32& getLocsObjects() const { return m_locs; }
		inline const poca::core::MyArrayVec3mf& getTrianglesObjects() const { return m_triangles; }
		inline const poca::core::MyArrayVec3mf& getOutlinesObjects() const { return m_outlines; }
		inline const std::vector <poca::core::Vec3mf>& getNormalOutlineLocs() const { return m_normalOutlineLocs; }

		virtual const float* getXs() const = 0;
		virtual const float* getYs() const = 0;
		virtual const float* getZs() const = 0;

		virtual const uint32_t dimension() const = 0;
		virtual const size_t nbObjects() const = 0;

		virtual bool hasSkeletons() const = 0;

		inline void setOutlineLocs(const std::vector <uint32_t>& _locsOutlines, const std::vector <uint32_t>& _firstLocsOutlines) { m_outlineLocs = poca::core::MyArrayUInt32(_locsOutlines, _firstLocsOutlines); }
		inline const poca::core::MyArrayUInt32& getLocOutlines() const { return m_outlineLocs; }
		virtual void generateOutlineLocs(std::vector <poca::core::Vec3mf>&) = 0;
		virtual void getOutlineLocsFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const = 0;
		virtual void getOutlineLocsFeatureInSelectionHiLow(std::vector <float>&, const std::vector <bool>&, const float, const float) const = 0;

		virtual const std::vector < std::array<poca::core::Vec3mf, 3>>& getAxisObjects() const { return m_axis; }

	protected:
		ObjectListInterface(const std::string& _name) :BasicComponent(_name){}
		ObjectListInterface(const std::string& _name, 
			const std::vector <uint32_t>& _locsAllObjects, 
			const std::vector <uint32_t>& _firstsLocs, 
			const std::vector <poca::core::Vec3mf>& _trianglesAllObjects, 
			const std::vector <uint32_t>& _firstsTriangles,
			const std::vector <poca::core::Vec3mf>& _outlinesAllObjects, 
			const std::vector <uint32_t>& _firstsOutlines) :BasicComponent(_name), m_locs(_locsAllObjects, _firstsLocs), m_triangles(_trianglesAllObjects, _firstsTriangles), m_outlines(_outlinesAllObjects, _firstsOutlines) {}

		ObjectListInterface(const std::string& _name, 
			const std::vector <uint32_t>& _locsAllObjects, 
			const std::vector <uint32_t>& _firstsLocs, 
			const std::vector <poca::core::Vec3mf>& _trianglesAllObjects, 
			const std::vector <uint32_t>& _firstsTriangles,
			const std::vector <uint32_t>& _locsOutlineAllObject,
			const std::vector <uint32_t>& _firstsOutlines, 
			const std::vector <poca::core::Vec3mf>& _normalOutlineLocs) :BasicComponent(_name), m_locs(_locsAllObjects, _firstsLocs), m_triangles(_trianglesAllObjects, _firstsTriangles), m_outlineLocs(_locsOutlineAllObject, _firstsOutlines), m_normalOutlineLocs(_normalOutlineLocs) {}

	protected:
		poca::core::MyArrayUInt32 m_locs, m_outlineLocs;
		poca::core::MyArrayVec3mf m_triangles, m_outlines;
		std::vector <poca::core::Vec3mf> m_normalOutlineLocs;
		std::vector < std::array<poca::core::Vec3mf, 3>> m_axis;
	};
}

#endif

