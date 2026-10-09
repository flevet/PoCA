/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "ObjectListMesh.hpp"
#include <General/Misc.h>
#include <algorithm>
#include <limits>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace poca::geometry {
	ObjectListMesh::ObjectListMesh(PersistedIndexedMeshes, IndexedMeshGeometry&& _geometry,
		std::map<std::string, std::unique_ptr<poca::core::MyData>> _features, bool _triangleZ,
		std::vector<std::array<poca::core::Vec3mf,3>> _axes,
		std::optional<poca::core::PersistedHistogramState> _triangleState, PersistedConstructionTiming* _timing, PersistedNormals _normals)
		:ObjectListInterface("ObjectListMesh"), m_meshesMaterialized(false), m_repair(false), m_applyRemeshing(false)
	{
		using Clock = std::chrono::steady_clock;
		auto start = _timing ? Clock::now() : Clock::time_point{};
		const auto checkpoint = [&](PersistedConstructionTiming::Phase phase) {
			if (!_timing) return;
			const auto now = Clock::now();
			_timing->seconds[phase] += std::chrono::duration<double>(now-start).count();
			start = now;
		};
		_geometry.validateStorage();
		checkpoint(PersistedConstructionTiming::Safety);
		if (_geometry.validationLevel() == MeshValidationLevel::Unknown) {
			_geometry.validate();
			if (_timing) _timing->validationExecuted = true;
			checkpoint(PersistedConstructionTiming::Validation);
		}
		else if (_timing) start = Clock::now(); // A skipped topology pass contributes no validation work.
		const auto objects = _geometry.nbObjects();
		for (const auto& feature : _features)
			if (!feature.second || feature.second->nbElements() != objects)
				throw std::invalid_argument("Persisted per-object feature count mismatch: " + feature.first);
		std::vector<poca::core::Vec3mf> triangles;
		std::vector<uint32_t> firstTriangles{0}, firstLocs{0}, locs;
		triangles.reserve(_geometry.faces().size()*3);
		m_xs.reserve(_geometry.vertices().size()); m_ys.reserve(_geometry.vertices().size()); m_zs.reserve(_geometry.vertices().size());
		m_bbox = poca::core::BoundingBox::initBBox();
		if (!_axes.empty() && _axes.size() != objects)
			throw std::invalid_argument("Persisted mesh rendering axis count mismatch");
		m_axis = std::move(_axes);
		checkpoint(PersistedConstructionTiming::Auxiliary);
		for (size_t object = 0; object < objects; ++object) {
			auto bbox = poca::core::BoundingBox::initBBox();
			poca::core::Vec3md sum(0.,0.,0.);
			for (uint64_t v = _geometry.vertexOffsets()[object]; v < _geometry.vertexOffsets()[object+1]; ++v) {
				const auto& xyz = _geometry.vertices()[v];
				m_xs.push_back(static_cast<float>(xyz[0])); m_ys.push_back(static_cast<float>(xyz[1])); m_zs.push_back(static_cast<float>(xyz[2]));
				locs.push_back(static_cast<uint32_t>(locs.size()));
				bbox.addPointBBox(xyz[0],xyz[1],xyz[2]); m_bbox.addPointBBox(xyz[0],xyz[1],xyz[2]);
				sum += poca::core::Vec3md(xyz[0],xyz[1],xyz[2]);
			}
			const auto center = sum / static_cast<double>(_geometry.vertexOffsets()[object+1]-_geometry.vertexOffsets()[object]);
			m_centroids.emplace_back(center.x(),center.y(),center.z());
			m_bboxMeshes.push_back(bbox);
			firstLocs.push_back(static_cast<uint32_t>(locs.size()));
			checkpoint(PersistedConstructionTiming::Auxiliary);
			for (uint64_t f = _geometry.faceOffsets()[object]; f < _geometry.faceOffsets()[object+1]; ++f)
				for (const auto vertex : _geometry.faces()[f]) {
					const auto& xyz = _geometry.vertices()[vertex];
					triangles.emplace_back(xyz[0],xyz[1],xyz[2]);
				}
			firstTriangles.push_back(static_cast<uint32_t>(triangles.size()));
			checkpoint(PersistedConstructionTiming::Triangles);
		}
		m_locs.initialize(locs,firstLocs); m_outlineLocs = m_locs;
		checkpoint(PersistedConstructionTiming::Auxiliary);
		m_triangles.initialize(triangles,firstTriangles);
		checkpoint(PersistedConstructionTiming::Triangles);
		if (!_normals.vertex.empty() || !_normals.face.empty()) {
			if (_normals.vertex.size() != _geometry.vertices().size() || _normals.face.size() != _geometry.faces().size())
				throw std::invalid_argument("Persisted mesh normal count mismatch");
			for (const auto* normals : {&_normals.vertex,&_normals.face})
				for (const auto& normal : *normals)
					for (size_t d = 0; d < 3; ++d)
						if (!std::isfinite(normal[d])) throw std::invalid_argument("Non-finite persisted mesh normal");
			m_indexedVertexNormals = std::move(_normals.vertex); m_indexedFaceNormals = std::move(_normals.face);
			if (_timing) _timing->normalsRestored = true;
		}
		else _geometry.generateNormals(m_indexedVertexNormals,m_indexedFaceNormals);
		checkpoint(PersistedConstructionTiming::Normals);
		if (_features.empty()) {
			// Older payloads with no quantitative features still need one object-level display feature.
			std::vector<float> ids(objects);
			for (size_t i = 0; i < objects; ++i) ids[i] = static_cast<float>(i+1);
			auto hist = std::make_unique<poca::core::Histogram<float>>();
			hist->initializeStorageBacked(objects,{},true,ids.front(),ids.back(),true);
			hist->setHistogram(ids,false);
			auto data = std::make_unique<poca::core::MyData>(hist.get(),true,true);
			hist.release();
			_features.emplace("id",std::move(data));
		}
		checkpoint(PersistedConstructionTiming::Adoption);
		if (_triangleZ) {
			if (_features.count("z")) throw std::invalid_argument("Duplicate persisted object/triangle z feature");
			std::vector<float> zs;
			zs.reserve(triangles.size());
			for (const auto& point : triangles) zs.push_back(point.z());
			if (zs.empty()) throw std::invalid_argument("Derived triangle z requires mesh faces");
			const auto bounds = std::minmax_element(zs.begin(),zs.end());
			auto histogram = std::make_unique<poca::core::Histogram<float>>();
			if (_triangleState) {
				histogram->restorePersistedState(*_triangleState);
				histogram->adoptPersistedValues(std::move(zs));
			}
			else {
				histogram->initializeStorageBacked(zs.size(),{},true,*bounds.first,*bounds.second);
				histogram->setHistogram(zs,false);
			}
			histogram->setInteraction(false);
			auto data = std::make_unique<poca::core::MyData>(histogram.get(),true,true);
			histogram.release();
			_features.emplace("z",std::move(data));
		}
		checkpoint(PersistedConstructionTiming::TriangleZ);
		const auto current = _features.count("volume") ? "volume" : _features.begin()->first;
		adoptPersistedFeatures(std::move(_features),objects,current);
		m_centroid = m_bbox.centroid();
		m_indexedGeometry = std::make_shared<const IndexedMeshGeometry>(std::move(_geometry));
		checkpoint(PersistedConstructionTiming::Adoption);
	}

	ObjectListMesh::ObjectListMesh(const ObjectListMesh& _other) : ObjectListInterface(_other)
	{
		std::lock_guard<std::mutex> lock(_other.m_meshMutex);
		m_meshes = _other.m_meshes; m_meshesMaterialized = _other.m_meshesMaterialized;
		m_indexedGeometry = _other.m_indexedGeometry; // Immutable resident ownership survives the source.
		m_indexedVertexNormals = _other.m_indexedVertexNormals; m_indexedFaceNormals = _other.m_indexedFaceNormals;
		m_mutableMeshesExposed = _other.m_mutableMeshesExposed;
		m_cgalValidatedObjects = _other.m_mutableMeshesExposed ? std::vector<bool>{} : _other.m_cgalValidatedObjects;
		// A mutable source may already contain stale property maps/inspection results at copy time.
		m_centroids = _other.m_centroids; m_bboxMeshes = _other.m_bboxMeshes;
		m_edgesSkeleton = _other.m_edgesSkeleton; m_linksSkeleton = _other.m_linksSkeleton;
		m_xs = _other.m_xs; m_ys = _other.m_ys; m_zs = _other.m_zs;
		m_repair = _other.m_repair; m_applyRemeshing = _other.m_applyRemeshing;
		m_targetLength = _other.m_targetLength; m_iterations = _other.m_iterations;
		m_useVertexNormals = _other.m_useVertexNormals;
	}

	void ObjectListMesh::ensureMeshesMaterialized() const
	{
		if (m_meshesMaterialized) return;
		if (!m_indexedGeometry) throw std::logic_error("Missing indexed mesh authority");
		auto meshes = m_indexedGeometry->materialize();
		m_meshes.swap(meshes);
		m_meshesMaterialized = true;
	}

	const std::vector<Surface_mesh_3_double>& ObjectListMesh::getMeshes() const
	{
		std::lock_guard<std::mutex> lock(m_meshMutex);
		ensureMeshesMaterialized();
		return m_meshes;
	}

	std::vector<Surface_mesh_3_double>& ObjectListMesh::getMeshes()
	{
		std::lock_guard<std::mutex> lock(m_meshMutex);
		ensureMeshesMaterialized();
		// A retained mutable reference can change later: never reuse its old backing for export.
		// As with the existing API, callers serialize mutation against display/copy/export.
		m_indexedGeometry.reset();
		m_indexedVertexNormals.clear(); m_indexedFaceNormals.clear();
		m_cgalValidatedObjects.clear(); m_mutableMeshesExposed = true;
		return m_meshes;
	}

	bool ObjectListMesh::meshesMaterialized() const
	{
		std::lock_guard<std::mutex> lock(m_meshMutex);
		return m_meshesMaterialized;
	}

	std::shared_ptr<const ObjectListMesh::IndexedMeshGeometry> ObjectListMesh::indexedGeometry() const
	{
		std::lock_guard<std::mutex> lock(m_meshMutex);
		if (m_indexedGeometry) return m_indexedGeometry;
		const bool certified = !m_mutableMeshesExposed && m_cgalValidatedObjects.size() == m_meshes.size() &&
			std::all_of(m_cgalValidatedObjects.begin(),m_cgalValidatedObjects.end(),[](bool value) { return value; });
		auto geometry = IndexedMeshGeometry::fromMeshes(m_meshes,certified ? MeshValidationLevel::CgalValidated : MeshValidationLevel::Unknown);
		return std::make_shared<const IndexedMeshGeometry>(std::move(geometry));
	}

	ObjectListMesh::MeshValidationLevel ObjectListMesh::meshValidationLevel() const
	{
		std::lock_guard<std::mutex> lock(m_meshMutex);
		if (m_mutableMeshesExposed) return MeshValidationLevel::Unknown;
		if (m_indexedGeometry) return m_indexedGeometry->validationLevel();
		return !m_meshes.empty() && m_cgalValidatedObjects.size() == m_meshes.size() &&
			std::all_of(m_cgalValidatedObjects.begin(),m_cgalValidatedObjects.end(),[](bool value) { return value; }) ?
			MeshValidationLevel::CgalValidated : MeshValidationLevel::Unknown;
	}

	MeshInspection ObjectListMesh::inspectMesh(size_t object) const
	{
		std::lock_guard<std::mutex> lock(m_meshMutex);
		ensureMeshesMaterialized();
		if (m_cgalValidatedObjects.size() != m_meshes.size()) m_cgalValidatedObjects.assign(m_meshes.size(),false);
		m_cgalValidatedObjects.at(object) = false;
		if (m_indexedGeometry && m_indexedGeometry->m_validation == MeshValidationLevel::CgalValidated)
			m_indexedGeometry->m_validation = MeshValidationLevel::IndexedValidated;
		const auto inspection = MeshRepair::inspect(m_meshes.at(object));
		m_cgalValidatedObjects[object] = MeshRepair::isStrictlyValid(inspection);
		if (!m_mutableMeshesExposed && m_indexedGeometry &&
			std::all_of(m_cgalValidatedObjects.begin(),m_cgalValidatedObjects.end(),[](bool value) { return value; }))
			m_indexedGeometry->m_validation = MeshValidationLevel::CgalValidated;
		return inspection;
	}

	void ObjectListMesh::normalsForGeometry(const std::shared_ptr<const IndexedMeshGeometry>& geometry, PersistedNormals& normals) const
	{
		std::lock_guard<std::mutex> lock(m_meshMutex);
		if (geometry && geometry == m_indexedGeometry) {
			normals.vertex = m_indexedVertexNormals; normals.face = m_indexedFaceNormals;
			return;
		}
		if (!geometry) throw std::invalid_argument("Missing geometry for normal export");
		if (!m_mutableMeshesExposed) {
			normals.vertex.clear(); normals.face.clear();
			bool complete = true;
			for (const auto& mesh : m_meshes) {
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(6, 0, 0)
				const auto vertex = mesh.property_map<vertex_descriptor,Kernel::Vector_3>("v:norm");
				const auto face = mesh.property_map<face_descriptor,Kernel::Vector_3>("f:norm");
				if (!vertex || !face) { complete = false; break; }
				const auto vn = vertex.value(); const auto fn = face.value();
#else
				const auto vertex = mesh.property_map<vertex_descriptor,Kernel::Vector_3>("v:norm");
				const auto face = mesh.property_map<face_descriptor,Kernel::Vector_3>("f:norm");
				if (!vertex.second || !face.second) { complete = false; break; }
				const auto vn = vertex.first; const auto fn = face.first;
#endif
				for (const auto v : mesh.vertices()) { const auto n = vn[v]; normals.vertex.emplace_back(n.x(),n.y(),n.z()); }
				for (const auto f : mesh.faces()) { const auto n = fn[f]; normals.face.emplace_back(n.x(),n.y(),n.z()); }
			}
			if (complete && normals.vertex.size() == geometry->vertices().size() && normals.face.size() == geometry->faces().size()) return;
		}
		// Mutable CGAL references can outlive an export: old normal properties are not authoritative.
		geometry->generateNormals(normals.vertex,normals.face);
	}

	const unsigned int ObjectListMesh::memorySize() const
	{
		size_t bytes = poca::core::BasicComponent::memorySize();
		bytes += (m_xs.capacity()+m_ys.capacity()+m_zs.capacity())*sizeof(float);
		bytes += m_triangles.getData().capacity()*sizeof(poca::core::Vec3mf);
		bytes += (m_indexedVertexNormals.capacity()+m_indexedFaceNormals.capacity())*sizeof(poca::core::Vec3mf);
		if (m_indexedGeometry) bytes += m_indexedGeometry->memorySize();
		// CGAL owns additional allocator/property-map storage; feature lengths are not resident RAM.
		bytes += m_centroids.capacity()*sizeof(poca::core::Vec3mf);
		bytes += m_bboxMeshes.capacity()*sizeof(poca::core::BoundingBox);
		bytes += m_axis.capacity()*sizeof(std::array<poca::core::Vec3mf,3>);
		return static_cast<unsigned int>((std::min)(bytes,size_t((std::numeric_limits<unsigned int>::max)())));
	}
}