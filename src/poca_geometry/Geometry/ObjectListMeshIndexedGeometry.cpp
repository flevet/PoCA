/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "ObjectListMesh.hpp"
#include <chrono>
#include <CGAL/boost/graph/iterator.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <boost/property_map/property_map.hpp>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace poca::geometry {
	void ObjectListMesh::IndexedMeshGeometry::restoreValidation(MeshValidationLevel level, uint32_t version)
	{
		m_validation = version == MeshValidationVersion &&
			(level == MeshValidationLevel::IndexedValidated || level == MeshValidationLevel::CgalValidated) ? level : MeshValidationLevel::Unknown;
	}

	void ObjectListMesh::IndexedMeshGeometry::validateStorage() const
	{
		const uint64_t limit = (std::numeric_limits<uint32_t>::max)();
		if (!nbObjects() || nbObjects() > limit || m_vertices.size() > limit || m_faces.size() > limit/3 ||
			m_faceOffsets.size() != m_vertexOffsets.size() || m_vertexOffsets.front() || m_faceOffsets.front() ||
			m_vertexOffsets.back() != m_vertices.size() || m_faceOffsets.back() != m_faces.size())
			throw std::invalid_argument("Invalid indexed mesh sizes/offsets");
		for (const auto& point : m_vertices)
			for (double value : point)
				if (!std::isfinite(value) || std::abs(value) > (std::numeric_limits<float>::max)())
					throw std::invalid_argument("Indexed mesh coordinate cannot be rendered by PoCA");
		for (size_t object = 0; object < nbObjects(); ++object) {
			const auto first = m_vertexOffsets[object], end = m_vertexOffsets[object+1];
			if (first >= end || end > m_vertices.size() || m_faceOffsets[object] > m_faceOffsets[object+1] ||
				m_faceOffsets[object+1] > m_faces.size())
				throw std::invalid_argument("Invalid indexed mesh object interval");
			for (uint64_t f = m_faceOffsets[object]; f < m_faceOffsets[object+1]; ++f) {
				const auto& face = m_faces[f];
				for (size_t k = 0; k < 3; ++k)
					if (face[k] < first || face[k] >= end || face[k] == face[(k+1)%3])
						throw std::invalid_argument("Indexed triangle outside its object interval or repeated vertex");
			}
		}
	}

	void ObjectListMesh::IndexedMeshGeometry::validate() const
	{
		m_validation = MeshValidationLevel::Unknown;
		validateStorage();
		for (size_t object = 0; object < nbObjects(); ++object) {
			const auto first = m_vertexOffsets[object], end = m_vertexOffsets[object+1];
			// Reject duplicate directed edges and disconnected vertex fans without Surface_mesh.
			std::map<std::pair<uint64_t,uint64_t>,size_t> directed;
			std::map<uint64_t,std::map<uint64_t,uint64_t>> fans;
			for (uint64_t f = m_faceOffsets[object]; f < m_faceOffsets[object+1]; ++f) {
				const auto& face = m_faces[f];
				for (size_t k = 0; k < 3; ++k) {
					const auto v = face[k], next = face[(k+1)%3], previous = face[(k+2)%3];
					if (v < first || v >= end || v == next ||
						!directed.emplace(std::make_pair(v,next),static_cast<size_t>(f)).second ||
						!fans[v].emplace(previous,next).second)
						throw std::invalid_argument("Indexed triangle has invalid indices or nonmanifold/orientation conflicts");
				}
			}
			for (const auto& fan : fans) {
				std::set<uint64_t> incoming;
				for (const auto& link : fan.second)
					if (!incoming.insert(link.second).second) throw std::invalid_argument("Nonmanifold indexed vertex fan");
				uint64_t start = fan.second.begin()->first;
				size_t boundaries = 0;
				for (const auto& link : fan.second)
					if (!incoming.count(link.first)) { start = link.first; ++boundaries; }
				if (boundaries > 1) throw std::invalid_argument("Disconnected indexed vertex fan");
				uint64_t cursor = start;
				size_t visited = 0;
				do {
					const auto link = fan.second.find(cursor);
					if (link == fan.second.end()) break;
					cursor = link->second;
					++visited;
				} while (cursor != start && visited <= fan.second.size());
				if (visited != fan.second.size()) throw std::invalid_argument("Disconnected indexed vertex fan");
			}
		}
		m_validation = MeshValidationLevel::IndexedValidated;
	}

	void ObjectListMesh::IndexedMeshGeometry::validateCgal() const
	{
		validate();
		const auto meshes = materialize();
		for (const auto& mesh : meshes)
			if (!MeshRepair::isStrictlyValid(MeshRepair::inspect(mesh)))
				throw std::invalid_argument("Indexed geometry failed the complete CGAL mesh-repair contract");
		m_validation = MeshValidationLevel::CgalValidated;
	}

	ObjectListMesh::IndexedMeshGeometry ObjectListMesh::IndexedMeshGeometry::fromMeshes(
		const std::vector<Surface_mesh_3_double>& _meshes, MeshValidationLevel _validation)
	{
		return fromMeshes(_meshes,_validation,nullptr);
	}

	ObjectListMesh::IndexedMeshGeometry ObjectListMesh::IndexedMeshGeometry::fromMeshes(
		const std::vector<Surface_mesh_3_double>& _meshes, MeshValidationLevel _validation, double* _validationSeconds)
	{
		IndexedMeshGeometry result;
		result.m_vertexOffsets.push_back(0); result.m_faceOffsets.push_back(0);
		for (const auto& mesh : _meshes) {
			std::map<vertex_descriptor,uint64_t> indices;
			for (const auto vertex : mesh.vertices()) {
				indices.emplace(vertex,result.m_vertices.size());
				const auto& point = mesh.point(vertex);
				result.m_vertices.push_back({CGAL::to_double(point.x()),CGAL::to_double(point.y()),CGAL::to_double(point.z())});
			}
			for (const auto f : mesh.faces()) {
				std::array<uint64_t,3> face;
				size_t corner = 0;
				for (const auto vertex : CGAL::vertices_around_face(mesh.halfedge(f),mesh)) {
					if (corner == 3) throw std::invalid_argument("Nontriangular mesh cannot be persisted");
					face[corner++] = indices.at(vertex);
				}
				if (corner != 3) throw std::invalid_argument("Nontriangular mesh cannot be persisted");
				result.m_faces.push_back(face);
			}
			result.m_vertexOffsets.push_back(result.m_vertices.size());
			result.m_faceOffsets.push_back(result.m_faces.size());
		}
		result.restoreValidation(_validation,MeshValidationVersion);
		const auto started = _validationSeconds ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
		if (result.validationLevel() == MeshValidationLevel::Unknown) result.validate();
		else result.validateStorage();
		if (_validationSeconds) *_validationSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
		return result;
	}

	std::vector<Surface_mesh_3_double> ObjectListMesh::IndexedMeshGeometry::materialize() const
	{
		std::vector<Surface_mesh_3_double> result;
		result.reserve(nbObjects());
		for (size_t object = 0; object < nbObjects(); ++object) {
			Surface_mesh_3_double mesh;
			std::vector<vertex_descriptor> local;
			local.reserve(static_cast<size_t>(m_vertexOffsets[object+1]-m_vertexOffsets[object]));
			for (uint64_t v = m_vertexOffsets[object]; v < m_vertexOffsets[object+1]; ++v) {
				const auto& point = m_vertices[v];
				local.push_back(mesh.add_vertex(Point_3_double(point[0],point[1],point[2])));
			}
			for (uint64_t f = m_faceOffsets[object]; f < m_faceOffsets[object+1]; ++f) {
				const auto& triangle = m_faces[f];
				const auto first = local.at(static_cast<size_t>(triangle[0]-m_vertexOffsets[object]));
				const auto fd = mesh.add_face(first,local.at(static_cast<size_t>(triangle[1]-m_vertexOffsets[object])),
					local.at(static_cast<size_t>(triangle[2]-m_vertexOffsets[object])));
				if (fd == Surface_mesh_3_double::null_face())
					throw std::runtime_error("Indexed triangle cannot be represented as Surface_mesh");
				for (size_t rotation = 0; rotation < 3; ++rotation) {
					const auto around = CGAL::vertices_around_face(mesh.halfedge(fd),mesh);
					if (*around.begin() == first) break;
					mesh.set_halfedge(fd,mesh.next(mesh.halfedge(fd)));
				}
			}
			auto fn = mesh.add_property_map<face_descriptor,Kernel::Vector_3>("f:norm").first;
			auto vn = mesh.add_property_map<vertex_descriptor,Kernel::Vector_3>("v:norm").first;
			CGAL::Polygon_mesh_processing::compute_face_normals(mesh,fn);
			CGAL::Polygon_mesh_processing::compute_vertex_normals(mesh,vn);
			result.push_back(std::move(mesh));
		}
		return result; // Publish only after every conversion succeeds.
	}

	void ObjectListMesh::IndexedMeshGeometry::generateNormals(std::vector<poca::core::Vec3mf>& _vertex,
		std::vector<poca::core::Vec3mf>& _face) const
	{
		using Vector = Kernel::Vector_3;
		Kernel traits;
		std::map<face_descriptor,Vector> normals;
		std::vector<std::vector<face_descriptor>> incident(m_vertices.size());
		_face.clear(); _face.reserve(m_faces.size());
		for (size_t f = 0; f < m_faces.size(); ++f) {
			const auto& face = m_faces[f];
			const auto& a = m_vertices[face[0]], &b = m_vertices[face[1]], &c = m_vertices[face[2]];
			const Point_3_double p(a[0],a[1],a[2]), q(b[0],b[1],b[2]), r(c[0],c[1],c[2]);
			auto normal = CGAL::cross_product(r-q,p-q)/2.;
			CGAL::Polygon_mesh_processing::internal::normalize(normal,traits);
			const face_descriptor fd(static_cast<uint32_t>(f));
			normals.emplace(fd,normal);
			for (const auto vertex : face) incident[vertex].push_back(fd);
			_face.emplace_back(normal.x(),normal.y(),normal.z());
		}
		boost::associative_property_map<std::map<face_descriptor,Vector>> map(normals);
		_vertex.clear(); _vertex.reserve(m_vertices.size());
		for (size_t v = 0; v < m_vertices.size(); ++v) {
			auto& around = incident[v];
			Vector normal(CGAL::NULL_VECTOR);
			if (around.size() == 1) normal = normals.at(around.front());
			else if (around.size() > 1) {
				// Reuse CGAL's existing most-visible-normal solver on indexed incidence.
				// Surface_mesh is only a descriptor type here; no mesh instance is constructed.
				normal = CGAL::Polygon_mesh_processing::internal::compute_most_visible_normal_2_points<Surface_mesh_3_double>(around,map,traits);
				if (normal == CGAL::NULL_VECTOR && around.size() > 2)
					normal = CGAL::Polygon_mesh_processing::internal::compute_most_visible_normal_3_points<Surface_mesh_3_double>(around,map,traits);
			}
			if (normal == CGAL::NULL_VECTOR && !around.empty()) {
				Vector weighted(CGAL::NULL_VECTOR), unweighted(CGAL::NULL_VECTOR);
				bool degenerate = false;
				const auto& p = m_vertices[v];
				for (const auto f : around) {
					const auto& face = m_faces[static_cast<size_t>(f.idx())];
					size_t k = 0;
					while (face[k] != v) ++k;
					const auto& a = m_vertices[face[(k+1)%3]], &b = m_vertices[face[(k+2)%3]];
					const Vector u(a[0]-p[0],a[1]-p[1],a[2]-p[2]), w(b[0]-p[0],b[1]-p[1],b[2]-p[2]);
					const double den = std::sqrt(u.squared_length()*w.squared_length());
					if (den == 0.) degenerate = true;
					else weighted = weighted + CGAL::cross_product(u,w)/den;
					unweighted = unweighted + normals.at(f);
				}
				normal = degenerate ? unweighted : weighted;
			}
			CGAL::Polygon_mesh_processing::internal::normalize(normal,traits);
			_vertex.emplace_back(normal.x(),normal.y(),normal.z());
		}
	}

	size_t ObjectListMesh::IndexedMeshGeometry::memorySize() const
	{
		return m_vertices.capacity()*sizeof(std::array<double,3>) + m_faces.capacity()*sizeof(std::array<uint64_t,3>) +
			(m_vertexOffsets.capacity()+m_faceOffsets.capacity())*sizeof(uint64_t);
	}
}