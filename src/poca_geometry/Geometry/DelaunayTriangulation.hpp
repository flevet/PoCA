/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DelaunayTriangulation.hpp
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

#ifndef DelaunayTriangulation_hpp__
#define DelaunayTriangulation_hpp__

#include <any>

#include <General/BasicComponent.hpp>
#include <General/MyArray.hpp>
#include <General/Vec3.hpp>

#include "../Interfaces/DelaunayTriangulationInterface.hpp"
#include "../Geometry/CGAL_includes.hpp"

namespace delaunator {
	class DelaunayFromDelaunator;
}

namespace poca::geometry {

	class DelaunayTriangulation2D : public DelaunayTriangulationInterface {
	public:
		~DelaunayTriangulation2D();

		void generateTriangles(std::vector <poca::core::Vec3mf>&);
		void getFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const;
		void generatePickingIndices(std::vector <float>&) const;
		poca::core::BoundingBox computeBoundingBoxElement(const int) const;
		poca::core::Vec3mf computeBarycenterElement(const int) const;
		void getVerticesTriangle(const size_t, std::vector <poca::core::Vec3mf>&) const;
		const uint32_t dimension() const { return 2; }
		void generateFaceSelectionFromLocSelection(const std::vector <bool>&, std::vector <bool>&);
		void generateFaceSelectionFromLocSelection(const std::vector <uint32_t>&, std::vector <uint32_t>&);
		void generateFaceSelectionFromLocSelection(const std::vector <uint32_t>&, std::map <uint32_t, std::vector<uint32_t>>&);
		bool isPointInTriangle(const float, const float, const float, const uint32_t) const;

		virtual poca::core::BasicComponentInterface* copy() = 0;
		virtual void getTrianglesNeighboringPoint(uint32_t, std::vector <uint32_t>&) = 0;
		virtual void trianglesAdjacentToTriangle(std::uint32_t, std::vector <std::uint32_t>&) = 0;
		virtual uint32_t indexTriangleOfPoint(const float, const float, const float, const uint32_t = std::numeric_limits<std::uint32_t>::max()) = 0;
		virtual const std::array<size_t, 3> getOutline(const size_t, const size_t) const = 0;
		
	protected:
		DelaunayTriangulation2D(const std::vector <float>&, const std::vector <float>&, const std::vector<size_t>&, const poca::core::MyArrayUInt32&);
		DelaunayTriangulation2D(const std::vector <float>&, const std::vector <float>&, const std::vector <float>&, const std::vector<size_t>&, const poca::core::MyArrayUInt32&);
	};

	class DelaunayTriangulation2DDelaunator : public DelaunayTriangulation2D {
	public:
		DelaunayTriangulation2DDelaunator(const std::vector <float>&, const std::vector <float>&, const std::vector<size_t>&, const poca::core::MyArrayUInt32&, delaunator::DelaunayFromDelaunator*, std::vector <double>*);
		~DelaunayTriangulation2DDelaunator();

		poca::core::BasicComponentInterface* copy();
		void getTrianglesNeighboringPoint(uint32_t, std::vector <uint32_t>&);
		void trianglesAdjacentToTriangle(std::uint32_t, std::vector <std::uint32_t>&);
		uint32_t indexTriangleOfPoint(const float, const float, const float, const uint32_t = std::numeric_limits<std::uint32_t>::max());
		const std::array<size_t, 3> getOutline(const size_t, const size_t) const;

		const size_t nbPoints() const { return m_coords->size() / 2; }
		inline delaunator::DelaunayFromDelaunator* getDelaunator() const { return m_internalDelaunay; }

	protected:
		delaunator::DelaunayFromDelaunator* m_internalDelaunay;
		std::vector <double>* m_coords;
	};

	class DelaunayTriangulation2DOnSphere : public DelaunayTriangulation2D {
	public:
		DelaunayTriangulation2DOnSphere(const std::vector <float>&, const std::vector <float>&, const std::vector <float>&, const std::vector<size_t>&, const poca::core::MyArrayUInt32&, CGALDelaunayOnSphere*, const poca::core::Vec3mf&, const float);
		~DelaunayTriangulation2DOnSphere();

		poca::core::BasicComponentInterface* copy();
		void getTrianglesNeighboringPoint(uint32_t, std::vector <uint32_t>&);
		void trianglesAdjacentToTriangle(std::uint32_t, std::vector <std::uint32_t>&);
		uint32_t indexTriangleOfPoint(const float, const float, const float, const uint32_t = std::numeric_limits<std::uint32_t>::max());
		const std::array<size_t, 3> getOutline(const size_t, const size_t) const;

		inline CGALDelaunayOnSphere* getDelaunay() const { return m_internalDelaunay; }
		inline const poca::core::Vec3mf& getCentroid() const { return m_centroid; }
		inline const float getRadius() const { return m_radius; }

	protected:
		poca::core::Vec3mf m_centroid;
		float m_radius;
		CGALDelaunayOnSphere* m_internalDelaunay;
	};

	class DelaunayTriangulation3D : public DelaunayTriangulationInterface {
	public:
		//DelaunayTriangulation3D(const std::vector <float>&, const std::vector <float>&, const std::vector <float>&, const std::vector <poca::core::Vec3mf>&, const std::vector <float>&, const poca::core::MyArraySizeT&, const size_t);
		DelaunayTriangulation3D(const std::vector <float>&, const std::vector <float>&, const std::vector <float>&, const std::vector <uint32_t>&, const std::vector <float>&, const poca::core::MyArrayUInt32&, const uint32_t, Triangulation_3_inexact*);
		~DelaunayTriangulation3D();

		poca::core::BasicComponentInterface* copy();

		void generateTriangles(std::vector <poca::core::Vec3mf>&);
		void getFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const;
		void generatePickingIndices(std::vector <float>&) const;
		poca::core::BoundingBox computeBoundingBoxElement(const int) const;
		poca::core::Vec3mf computeBarycenterElement(const int) const;
		void getVerticesTriangle(const size_t, std::vector <poca::core::Vec3mf>&) const;
		const uint32_t dimension() const { return 3; }
		void generateFaceSelectionFromLocSelection(const std::vector <bool>&, std::vector <bool>&);
		void generateFaceSelectionFromLocSelection(const std::vector <uint32_t>&, std::vector <uint32_t>&);
		void generateFaceSelectionFromLocSelection(const std::vector <uint32_t>&, std::map <uint32_t, std::vector<uint32_t>>&);

		void getTrianglesNeighboringPoint(uint32_t, std::vector <uint32_t>&) {}
		bool isPointInTriangle(const float, const float, const float, const uint32_t) const { return false; }
		void trianglesAdjacentToTriangle(std::uint32_t, std::vector <std::uint32_t>&) {}
		
		uint32_t indexTriangleOfPoint(const float, const float, const float, const uint32_t = std::numeric_limits<std::uint32_t>::max());

		const std::array<size_t, 3> getOutline(const size_t, const size_t) const { return std::array<size_t, 3>{0, 0, 0}; }
		inline Triangulation_3_inexact* getDelaunay() const { return m_internalDelaunay; }

	protected:
		Triangulation_3_inexact* m_internalDelaunay;
	};
}

#endif

