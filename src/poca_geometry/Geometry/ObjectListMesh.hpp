/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListMesh.hpp
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

#ifndef ObjectListMesh_hpp__
#define ObjectListMesh_hpp__

#include <any>
#include <tuple>

#include <General/BasicComponentList.hpp>
#include <General/MyArray.hpp>
#include <General/Vec3.hpp>
#include <Interfaces/ObjectListInterface.hpp>
#include <Geometry/CGAL_includes.hpp>

namespace poca::geometry {
	class ObjectListMesh : public poca::geometry::ObjectListInterface {
	public:
		ObjectListMesh(std::vector <std::vector <poca::core::Vec3mf>>&, std::vector <std::vector <std::vector <std::size_t>>>&, const bool = true, const bool = false, const double = 1., const uint32_t = 1);
		ObjectListMesh(std::vector <std::vector <Point_3_double>>&, std::vector <std::vector <std::vector <std::size_t>>>&, const bool = true, const bool = false, const double = 1., const uint32_t = 1);
		~ObjectListMesh();

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

		inline const uint32_t dimension() const { return 3; }
		inline const size_t nbObjects() const { return 0; }

		const float* getXs() const { return m_xs.data(); }
		const float* getYs() const { return m_ys.data(); }
		const float* getZs() const { return m_zs.data(); }

		bool hasSkeletons() const { return !m_edgesSkeleton.empty(); }

		virtual void generateOutlineLocs(std::vector <poca::core::Vec3mf>&);
		virtual void getOutlineLocsFeatureInSelection(std::vector <float>&, const std::vector <float>&, const std::vector <bool>&, const float) const;
		virtual void getOutlineLocsFeatureInSelectionHiLow(std::vector <float>&, const std::vector <bool>&, const float, const float) const;

		void computeSkeletons();
		void saveAsOBJ(const std::string&) const;

		inline const poca::core::MyArrayVec3mf& getSkeletons() const { return m_edgesSkeleton; }
		inline const poca::core::MyArrayVec3mf& getLinks() const { return m_linksSkeleton; }

	protected:
		const bool addObjectMesh(std::vector <Point_3_double>&, std::vector<std::vector<std::size_t> >&, 
									std::vector <poca::core::Vec3mf>&, std::vector <std::uint32_t>&, 
									std::vector <poca::core::Vec3mf>&, std::vector <std::uint32_t>&,
									std::vector <poca::core::Vec3mf>&, std::vector <std::uint32_t>&,
									std::vector <float>&);
		const bool processSurfaceMesh(Surface_mesh_3_double&, 
										std::vector <poca::core::Vec3mf>&, std::vector <std::uint32_t>&, 
										std::vector <poca::core::Vec3mf>&, std::vector <std::uint32_t>&,
										std::vector <poca::core::Vec3mf>&, std::vector <std::uint32_t>&,
										std::vector <float>&);

	protected:
		std::vector < Surface_mesh_3_double> m_meshes;
		std::vector <poca::core::Vec3mf> m_centroids;
		std::vector <poca::core::BoundingBox> m_bboxMeshes;

		poca::core::MyArrayVec3mf m_edgesSkeleton, m_linksSkeleton;

		//For now duplicate information about the points for compatibility with ObjectListInterface and existing plugins
		std::vector <float> m_xs, m_ys, m_zs;

		bool m_repair{ true };
		bool m_applyRemeshing{ false };
		double m_targetLength{ 1. };
		int32_t m_iterations{ 1 };
	};
}

#endif

