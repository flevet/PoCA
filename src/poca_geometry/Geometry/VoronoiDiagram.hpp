/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagram.hpp
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

#ifndef VoronoiDiagram_hpp__
#define VoronoiDiagram_hpp__

#include <any>

#include <General/BasicComponent.hpp>
#include <General/MyArray.hpp>
#include <General/Vec3.hpp>

#include "DetectionSet.hpp"
#include "../Interfaces/DelaunayTriangulationInterface.hpp"

namespace poca::geometry {
	class VoronoiDiagram : public poca::core::BasicComponent {
	public:
		virtual ~VoronoiDiagram();

		virtual poca::core::BasicComponentInterface* copy() = 0;

		virtual void generateTriangles(std::vector <poca::core::Vec3mf>&) = 0;
		virtual void generateLines(std::vector <poca::core::Vec3mf>&) = 0;
		virtual void getFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float, const bool = false) const = 0;
		virtual void generatePickingIndices(std::vector <float>&) const = 0;
		virtual poca::core::BoundingBox computeBoundingBoxElement(const int) const = 0;
		virtual poca::core::Vec3mf computeBarycenterElement(const int) const = 0;
		virtual const uint32_t dimension() const = 0;
		virtual const bool hasCells() const = 0;

		virtual void determineEdgeEffectLocalizations(std::vector <bool>&) const = 0;

		virtual const float* getXs() const { return m_xs; }
		virtual const float* getYs() const { return m_ys; }
		virtual const float* getZs() const { return m_zs; }

		virtual const uint32_t nbFaces() const { return m_nbCells; }

		virtual KdTree_DetectionPoint* getKdTree() const { return m_kdTree; }
		virtual DelaunayTriangulationInterface* getDelaunay() const { return m_delaunay; }
		virtual const poca::core::MyArrayUInt32& getNeighbors() const { return m_neighbors; }

		virtual void generateLinesNormals(std::vector <poca::core::Vec3mf>&) {}

		virtual const std::vector <bool>& borderLocalizations() const { return m_borderLocs; }

		virtual const float averageDensity() const;
		virtual const float averageMeanNbLocs() const;

		virtual uint32_t indexTriangleOfPoint(const float, const float, const float) const = 0;

	protected:
		VoronoiDiagram(const float*, const float*, const float*, const uint32_t, const std::vector<uint32_t>&, const std::vector<uint32_t>&, const std::vector <bool>&, KdTree_DetectionPoint* = NULL, DelaunayTriangulationInterface * = NULL);

	protected:
		const float* m_xs, * m_ys, * m_zs;
		uint32_t m_nbCells;
		poca::core::MyArrayUInt32 m_neighbors;
		
		bool m_deleteKdTree;
		KdTree_DetectionPoint* m_kdTree;
		DelaunayTriangulationInterface* m_delaunay;

		std::vector <bool> m_borderLocs;
	};

	class VoronoiDiagram2D : public VoronoiDiagram {
	public:
		VoronoiDiagram2D(const poca::core::Vec3md*, const uint32_t, const std::vector<uint32_t>&, const std::vector<uint32_t>&, const std::vector <bool>&, const float*, const float*, const float*, KdTree_DetectionPoint* = NULL, DelaunayTriangulationInterface* = NULL);
		~VoronoiDiagram2D();

		poca::core::BasicComponentInterface* copy();

		void generateTriangles(std::vector <poca::core::Vec3mf>&);
		void generateLines(std::vector <poca::core::Vec3mf>&);
		void getFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float, const bool = false) const;
		void generatePickingIndices(std::vector <float>&) const;
		poca::core::BoundingBox computeBoundingBoxElement(const int) const;
		poca::core::Vec3mf computeBarycenterElement(const int) const;
		uint32_t indexTriangleOfPoint(const float, const float, const float) const;

		void determineEdgeEffectLocalizations(std::vector <bool>&) const {}

		const uint32_t dimension() const { return 2; }
		const bool hasCells() const { return true; }

	protected:
		std::vector <poca::core::Vec3md> m_cells;
	};

	class VoronoiDiagram2DOnSphere : public VoronoiDiagram2D {
	public:
		VoronoiDiagram2DOnSphere(const std::vector<poca::core::Vec3md>&, const uint32_t, const std::vector<uint32_t>&, const std::vector<uint32_t>&, const std::vector <poca::core::Vec3mf>&, const float*, const float*, const float*, const poca::core::Vec3mf&, const float, KdTree_DetectionPoint* = NULL, DelaunayTriangulationInterface* = NULL);
		~VoronoiDiagram2DOnSphere();

		void generateLinesNormals(std::vector <poca::core::Vec3mf>&);

		const float averageDensity() const;
		const float averageMeanNbLocs() const;

		inline const std::vector <poca::core::Vec3mf>& getNormals() const { return m_normals; }
		inline const poca::core::Vec3mf& getCentroid() const { return m_centroid; }
		inline const float getRadius() const { return m_radius; }

	protected:
		poca::core::Vec3mf m_centroid;
		float m_radius;

		std::vector <poca::core::Vec3mf> m_normals;
	};

	class VoronoiDiagram3D : public VoronoiDiagram {
	public:
		VoronoiDiagram3D(const uint32_t, const std::vector <uint32_t>&, const std::vector <uint32_t>&, const std::vector <float>&, const std::vector <bool>&, const float*, const float*, const float*, KdTree_DetectionPoint* = NULL, DelaunayTriangulationInterface* = NULL);
		VoronoiDiagram3D(const uint32_t, const std::vector <uint32_t>&, const std::vector <uint32_t>&, const std::vector <poca::core::Vec3mf>&,
			const std::vector <uint32_t>&, const std::vector <float>&, const std::vector <bool>&, const float*, const float*, const float*, KdTree_DetectionPoint* = NULL, DelaunayTriangulationInterface* = NULL);
		~VoronoiDiagram3D();

		poca::core::BasicComponentInterface* copy();

		void generateTriangles(std::vector <poca::core::Vec3mf>&);
		void generateLines(std::vector <poca::core::Vec3mf>&);
		void getFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float, const bool = false) const;
		void generatePickingIndices(std::vector <float>&) const;
		poca::core::BoundingBox computeBoundingBoxElement(const int) const;
		poca::core::Vec3mf computeBarycenterElement(const int) const;
		uint32_t indexTriangleOfPoint(const float, const float, const float) const;

		void determineEdgeEffectLocalizations(std::vector <bool>&) const;

		const uint32_t dimension() const { return 3; }
		const bool hasCells() const { return !m_cells.empty(); }

		inline const std::vector <poca::core::Vec3mf>& getCells() const { return m_cells; }
		inline const std::vector <uint32_t>& getFirstCells() const { return m_firstCells; }


	protected:
		std::vector <poca::core::Vec3mf> m_cells;
		std::vector <uint32_t> m_firstCells;
	};
}
#endif

