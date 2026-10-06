/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "ObjectListMesh.hpp"
#include <General/Misc.h>
#include <CGAL/boost/graph/iterator.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace poca::geometry {
	ObjectListMesh::ObjectListMesh(PersistedIndexedMeshes, std::vector<Surface_mesh_3_double>&& _meshes,
		std::map<std::string, std::unique_ptr<poca::core::MyData>> _features, bool _triangleZ,
		std::vector<std::array<poca::core::Vec3mf,3>> _axes)
		:ObjectListInterface("ObjectListMesh"), m_meshes(std::move(_meshes)), m_repair(false), m_applyRemeshing(false)
	{
		if (m_meshes.empty() || _features.empty()) throw std::invalid_argument("Persisted mesh requires objects and features");
		const uint64_t limit = (std::numeric_limits<uint32_t>::max)();
		uint64_t vertices = 0, corners = 0;
		if (m_meshes.size() > limit) throw std::invalid_argument("Too many persisted mesh objects");
		for (const auto& mesh : m_meshes) {
			if (mesh.number_of_vertices() > limit-vertices || mesh.number_of_faces() > (limit-corners)/3)
				throw std::invalid_argument("Persisted mesh exceeds resident PoCA rendering index limits");
			vertices += mesh.number_of_vertices();
			corners += uint64_t(mesh.number_of_faces()) * 3;
		}
		for (const auto& feature : _features)
			if (!feature.second || feature.second->nbElements() != m_meshes.size())
				throw std::invalid_argument("Persisted per-object feature count mismatch: " + feature.first);
		std::vector<poca::core::Vec3mf> triangles;
		std::vector<uint32_t> firstTriangles{0}, firstLocs{0}, locs;
		triangles.reserve(static_cast<size_t>(corners));
		m_xs.reserve(static_cast<size_t>(vertices));
		m_ys.reserve(static_cast<size_t>(vertices));
		m_zs.reserve(static_cast<size_t>(vertices));
		m_bbox = poca::core::BoundingBox::initBBox();
		if (!_axes.empty() && _axes.size() != m_meshes.size())
			throw std::invalid_argument("Persisted mesh rendering axis count mismatch");
		m_axis = std::move(_axes);
		for (auto& mesh : m_meshes) {
			if (!mesh.number_of_vertices() || !CGAL::is_triangle_mesh(mesh))
				throw std::invalid_argument("Persisted mesh must contain vertices and triangular faces");
			auto bbox = poca::core::BoundingBox::initBBox();
			poca::core::Vec3md sum(0.,0.,0.);
			for (const auto v : mesh.vertices()) {
				const auto& p = mesh.point(v);
				const double xyz[3]{CGAL::to_double(p.x()),CGAL::to_double(p.y()),CGAL::to_double(p.z())};
				for (double value : xyz)
					if (!std::isfinite(value) || std::abs(value) > (std::numeric_limits<float>::max)())
						throw std::invalid_argument("Mesh coordinate cannot be rendered by PoCA");
				m_xs.push_back(static_cast<float>(xyz[0])); m_ys.push_back(static_cast<float>(xyz[1])); m_zs.push_back(static_cast<float>(xyz[2]));
				locs.push_back(static_cast<uint32_t>(locs.size()));
				bbox.addPointBBox(xyz[0],xyz[1],xyz[2]);
				m_bbox.addPointBBox(xyz[0],xyz[1],xyz[2]);
				sum += poca::core::Vec3md(xyz[0],xyz[1],xyz[2]);
			}
			const auto center = sum / static_cast<double>(mesh.number_of_vertices());
			m_centroids.emplace_back(center.x(),center.y(),center.z());
			m_bboxMeshes.push_back(bbox);
			firstLocs.push_back(static_cast<uint32_t>(locs.size()));
			for (const auto face : mesh.faces())
				for (const auto vertex : CGAL::vertices_around_face(mesh.halfedge(face),mesh)) {
					const auto& p = mesh.point(vertex);
					triangles.emplace_back(p.x(),p.y(),p.z());
				}
			firstTriangles.push_back(static_cast<uint32_t>(triangles.size()));
			auto fn = mesh.add_property_map<face_descriptor,Kernel::Vector_3>("f:norm").first;
			auto vn = mesh.add_property_map<vertex_descriptor,Kernel::Vector_3>("v:norm").first;
			CGAL::Polygon_mesh_processing::compute_face_normals(mesh,fn);
			CGAL::Polygon_mesh_processing::compute_vertex_normals(mesh,vn);
		}
		m_locs.initialize(locs,firstLocs);
		m_outlineLocs = m_locs;
		m_triangles.initialize(triangles,firstTriangles);
		if (_triangleZ) {
			if (_features.count("z")) throw std::invalid_argument("Duplicate persisted object/triangle z feature");
			std::vector<float> zs;
			zs.reserve(triangles.size());
			for (const auto& point : triangles) zs.push_back(point.z());
			if (zs.empty()) throw std::invalid_argument("Derived triangle z requires mesh faces");
			const auto bounds = std::minmax_element(zs.begin(),zs.end());
			auto histogram = std::make_unique<poca::core::Histogram<float>>();
			histogram->initializeStorageBacked(zs.size(),{},true,*bounds.first,*bounds.second);
			histogram->setHistogram(zs,false); // Use the constant-safe storage histogram path.
			histogram->setInteraction(false);
			auto data = std::make_unique<poca::core::MyData>(histogram.get(),true,true);
			histogram.release();
			_features.emplace("z",std::move(data));
		}
		std::map<std::string,poca::core::MyData*> features;
		for (auto& feature : _features) features.emplace(feature.first,feature.second.get());
		m_data.swap(features);
		for (auto& feature : _features) feature.second.release();
		m_selection.assign(m_meshes.size(),true);
		m_nbSelection = static_cast<unsigned int>(m_meshes.size());
		m_currentHistogram = m_data.count("volume") ? "volume" : m_data.begin()->first;
		m_centroid = m_bbox.centroid();
		// Selection/filtering is requested later; no persisted lazy feature is touched here.
	}

	const unsigned int ObjectListMesh::memorySize() const
	{
		size_t bytes = poca::core::BasicComponent::memorySize();
		bytes += (m_xs.capacity()+m_ys.capacity()+m_zs.capacity())*sizeof(float);
		bytes += m_triangles.getData().capacity()*sizeof(poca::core::Vec3mf);
		// CGAL owns additional allocator/property-map storage; report the resident arrays
		// we can measure, without counting unloaded feature lengths as allocated RAM.
		bytes += m_centroids.capacity()*sizeof(poca::core::Vec3mf);
		bytes += m_bboxMeshes.capacity()*sizeof(poca::core::BoundingBox);
		bytes += m_axis.capacity()*sizeof(std::array<poca::core::Vec3mf,3>);
		return static_cast<unsigned int>((std::min)(bytes,size_t((std::numeric_limits<unsigned int>::max)())));
	}
}
