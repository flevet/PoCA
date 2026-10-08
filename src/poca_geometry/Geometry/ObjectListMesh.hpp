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
#include <mutex>

#include <General/BasicComponentList.hpp>
#include <array>
#include <General/MyArray.hpp>
#include <General/Vec3.hpp>
#include <Interfaces/ObjectListInterface.hpp>
#include <Geometry/CGAL_includes.hpp>

namespace poca::geometry {
	class ObjectListMesh : public poca::geometry::ObjectListInterface {
	public:
		ObjectListMesh(std::vector <std::vector <poca::core::Vec3mf>>&, std::vector <std::vector <std::vector <std::size_t>>>&, const std::vector <poca::core::ROIInterface*>&, const bool = true, const bool = false, const double = 1., const uint32_t = 1);
		ObjectListMesh(std::vector <std::vector <Point_3_double>>&, std::vector <std::vector <std::vector <std::size_t>>>&, const bool = true, const bool = false, const double = 1., const uint32_t = 1);
		ObjectListMesh(const std::vector < Surface_mesh_3_double>&, const bool = false, const float = 0.f, const uint32_t = 0, const bool = true);
		// Lightweight constructor used by diagnostics: one open triangle per object.
		// It intentionally bypasses closed-surface volume/repair assumptions.
		ObjectListMesh(const std::vector < std::array<poca::core::Vec3mf, 3> >&);
		// Backend-independent scientific topology; double coordinates and global face indices.
		class IndexedMeshGeometry {
		public:
			std::vector<std::array<double,3>> vertices;
			std::vector<std::array<uint64_t,3>> faces;
			std::vector<uint64_t> vertexOffsets, faceOffsets;

			size_t nbObjects() const { return vertexOffsets.empty() ? 0 : vertexOffsets.size()-1; }
			void validate() const;
			void generateNormals(std::vector<poca::core::Vec3mf>&, std::vector<poca::core::Vec3mf>&) const;
			std::vector<Surface_mesh_3_double> materialize() const;
			static IndexedMeshGeometry fromMeshes(const std::vector<Surface_mesh_3_double>&);
			size_t memorySize() const;
		};
		// Persistence-only path: no repair, stitching, PCA, remeshing or CGAL construction.
		struct PersistedIndexedMeshes {};
		// Optional diagnostic output; no plugin/backend dependency or retained observer.
		struct PersistedConstructionTiming {
			enum Phase { Validation, Triangles, Normals, Auxiliary, TriangleZ, Adoption, Count };
			std::array<double,Count> seconds{};
		};
		ObjectListMesh(PersistedIndexedMeshes, IndexedMeshGeometry&&,
			std::map<std::string, std::unique_ptr<poca::core::MyData>>, bool = false,
			std::vector<std::array<poca::core::Vec3mf,3>> = {},
			std::optional<poca::core::PersistedHistogramState> = std::nullopt, PersistedConstructionTiming* = nullptr);
		ObjectListMesh(const ObjectListMesh&);
		ObjectListMesh& operator=(const ObjectListMesh&) = delete;
		const unsigned int memorySize() const override;
		~ObjectListMesh();

		poca::core::BasicComponentInterface* copy();
		poca::core::BasicComponentInterface* copy(const std::vector <poca::core::ROIInterface*>&);

		ObjectListInterface* exportFilteredObjects() const;
		ObjectListInterface* exportSelectedObjects(const std::set<int>&) const;

		void remesh(const float, const uint32_t);
		void subdivide(const uint32_t);

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
		inline const size_t nbObjects() const { return m_indexedGeometry ? m_indexedGeometry->nbObjects() : m_meshes.size(); }

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
		// Const access materializes a reusable analysis cache; mutable access invalidates indexed authority.
		const std::vector <Surface_mesh_3_double>& getMeshes() const;
		std::vector <Surface_mesh_3_double>& getMeshes();
		bool meshesMaterialized() const;
		std::shared_ptr<const IndexedMeshGeometry> indexedGeometry() const;
		inline const std::vector <poca::core::Vec3mf>& getCentroids() const { return m_centroids; }
		inline const std::vector <poca::core::BoundingBox>& getBBoxMeshes() const { return m_bboxMeshes; }
		inline bool useVertexNormals() const { return m_useVertexNormals; }
		inline void setUseVertexNormals(const bool _val) { m_useVertexNormals = _val; }

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
		void ensureMeshesMaterialized() const; // Caller holds m_meshMutex.
		mutable std::mutex m_meshMutex;
		mutable std::vector < Surface_mesh_3_double> m_meshes;
		mutable bool m_meshesMaterialized{ true };
		std::shared_ptr<const IndexedMeshGeometry> m_indexedGeometry;
		std::vector<poca::core::Vec3mf> m_indexedVertexNormals, m_indexedFaceNormals;
		std::vector <poca::core::Vec3mf> m_centroids;
		std::vector <poca::core::BoundingBox> m_bboxMeshes;

		poca::core::MyArrayVec3mf m_edgesSkeleton, m_linksSkeleton;

		//For now duplicate information about the points for compatibility with ObjectListInterface and existing plugins
		std::vector <float> m_xs, m_ys, m_zs;

		bool m_repair{ true };
		bool m_applyRemeshing{ false };
		double m_targetLength{ 1. };
		int32_t m_iterations{ 1 };
		bool m_useVertexNormals{ true };
	};
}

#endif
