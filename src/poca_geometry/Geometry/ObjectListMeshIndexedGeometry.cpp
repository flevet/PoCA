/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "ObjectListMesh.hpp"
#include <CGAL/boost/graph/iterator.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <boost/property_map/property_map.hpp>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace poca::geometry {
	void ObjectListMesh::IndexedMeshGeometry::validate() const
	{
		const uint64_t limit = (std::numeric_limits<uint32_t>::max)();
		if (!nbObjects() || nbObjects() > limit || vertices.size() > limit || faces.size() > limit/3 ||
			faceOffsets.size() != vertexOffsets.size() || vertexOffsets.front() || faceOffsets.front() ||
			vertexOffsets.back() != vertices.size() || faceOffsets.back() != faces.size())
			throw std::invalid_argument("Invalid indexed mesh sizes/offsets");
		for (const auto& point : vertices)
			for (double value : point)
				if (!std::isfinite(value) || std::abs(value) > (std::numeric_limits<float>::max)())
					throw std::invalid_argument("Indexed mesh coordinate cannot be rendered by PoCA");
		for (size_t object = 0; object < nbObjects(); ++object) {
			const auto first = vertexOffsets[object], end = vertexOffsets[object+1];
			if (first >= end || end > vertices.size() || faceOffsets[object] > faceOffsets[object+1] ||
				faceOffsets[object+1] > faces.size())
				throw std::invalid_argument("Invalid indexed mesh object interval");
			// Reject duplicate directed edges and disconnected vertex fans without Surface_mesh.
			std::map<std::pair<uint64_t,uint64_t>,size_t> directed;
			std::map<uint64_t,std::map<uint64_t,uint64_t>> fans;
			for (uint64_t f = faceOffsets[object]; f < faceOffsets[object+1]; ++f) {
				const auto& face = faces[f];
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
	}

	ObjectListMesh::IndexedMeshGeometry ObjectListMesh::IndexedMeshGeometry::fromMeshes(
		const std::vector<Surface_mesh_3_double>& _meshes)
	{
		IndexedMeshGeometry result;
		result.vertexOffsets.push_back(0); result.faceOffsets.push_back(0);
		for (const auto& mesh : _meshes) {
			std::map<vertex_descriptor,uint64_t> indices;
			for (const auto vertex : mesh.vertices()) {
				indices.emplace(vertex,result.vertices.size());
				const auto& point = mesh.point(vertex);
				result.vertices.push_back({CGAL::to_double(point.x()),CGAL::to_double(point.y()),CGAL::to_double(point.z())});
			}
			for (const auto f : mesh.faces()) {
				std::array<uint64_t,3> face;
				size_t corner = 0;
				for (const auto vertex : CGAL::vertices_around_face(mesh.halfedge(f),mesh)) {
					if (corner == 3) throw std::invalid_argument("Nontriangular mesh cannot be persisted");
					face[corner++] = indices.at(vertex);
				}
				if (corner != 3) throw std::invalid_argument("Nontriangular mesh cannot be persisted");
				result.faces.push_back(face);
			}
			result.vertexOffsets.push_back(result.vertices.size());
			result.faceOffsets.push_back(result.faces.size());
		}
		result.validate();
		return result;
	}

	std::vector<Surface_mesh_3_double> ObjectListMesh::IndexedMeshGeometry::materialize() const
	{
		std::vector<Surface_mesh_3_double> result;
		result.reserve(nbObjects());
		for (size_t object = 0; object < nbObjects(); ++object) {
			Surface_mesh_3_double mesh;
			std::vector<vertex_descriptor> local;
			local.reserve(static_cast<size_t>(vertexOffsets[object+1]-vertexOffsets[object]));
			for (uint64_t v = vertexOffsets[object]; v < vertexOffsets[object+1]; ++v) {
				const auto& point = vertices[v];
				local.push_back(mesh.add_vertex(Point_3_double(point[0],point[1],point[2])));
			}
			for (uint64_t f = faceOffsets[object]; f < faceOffsets[object+1]; ++f) {
				const auto& triangle = faces[f];
				const auto first = local.at(static_cast<size_t>(triangle[0]-vertexOffsets[object]));
				const auto fd = mesh.add_face(first,local.at(static_cast<size_t>(triangle[1]-vertexOffsets[object])),
					local.at(static_cast<size_t>(triangle[2]-vertexOffsets[object])));
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
		std::vector<std::vector<face_descriptor>> incident(vertices.size());
		_face.clear(); _face.reserve(faces.size());
		for (size_t f = 0; f < faces.size(); ++f) {
			const auto& face = faces[f];
			const auto& a = vertices[face[0]], &b = vertices[face[1]], &c = vertices[face[2]];
			const Point_3_double p(a[0],a[1],a[2]), q(b[0],b[1],b[2]), r(c[0],c[1],c[2]);
			auto normal = CGAL::cross_product(r-q,p-q)/2.;
			CGAL::Polygon_mesh_processing::internal::normalize(normal,traits);
			const face_descriptor fd(static_cast<uint32_t>(f));
			normals.emplace(fd,normal);
			for (const auto vertex : face) incident[vertex].push_back(fd);
			_face.emplace_back(normal.x(),normal.y(),normal.z());
		}
		boost::associative_property_map<std::map<face_descriptor,Vector>> map(normals);
		_vertex.clear(); _vertex.reserve(vertices.size());
		for (size_t v = 0; v < vertices.size(); ++v) {
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
				const auto& p = vertices[v];
				for (const auto f : around) {
					const auto& face = faces[static_cast<size_t>(f.idx())];
					size_t k = 0;
					while (face[k] != v) ++k;
					const auto& a = vertices[face[(k+1)%3]], &b = vertices[face[(k+2)%3]];
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
		return vertices.capacity()*sizeof(std::array<double,3>) + faces.capacity()*sizeof(std::array<uint64_t,3>) +
			(vertexOffsets.capacity()+faceOffsets.capacity())*sizeof(uint64_t);
	}
}