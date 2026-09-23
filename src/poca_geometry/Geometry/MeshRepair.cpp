/*
* Software:  PoCA: Point Cloud Analyst
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*/

#include "MeshRepair.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <new>
#include <set>
#include <sstream>
#include <stdexcept>

#include <CGAL/Kernel_traits.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/boost/graph/Euler_operations.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/tags.h>

namespace {
	namespace PMP = CGAL::Polygon_mesh_processing;
	using Mesh = Surface_mesh_3_double;
	using Face = boost::graph_traits<Mesh>::face_descriptor;
	using Vertex = boost::graph_traits<Mesh>::vertex_descriptor;
	using Halfedge = boost::graph_traits<Mesh>::halfedge_descriptor;
	using Vector = Kernel::Vector_3;

	std::vector<Vertex> faceVertices(const Mesh& mesh, const Face face)
	{
		std::vector<Vertex> vertices;
		const auto halfedge = mesh.halfedge(face);
		if (halfedge == Mesh::null_halfedge()) return vertices;
		CGAL::Vertex_around_face_iterator<Mesh> begin, end;
		for (boost::tie(begin, end) = vertices_around_face(halfedge, mesh); begin != end; ++begin)
			vertices.push_back(*begin);
		return vertices;
	}

	template <typename Point>
	bool finitePoint(const Point& point)
	{
		return std::isfinite(CGAL::to_double(point.x()))
			&& std::isfinite(CGAL::to_double(point.y()))
			&& std::isfinite(CGAL::to_double(point.z()));
	}

	template <typename VectorType>
	bool finiteVector(const VectorType& vector)
	{
		return std::isfinite(CGAL::to_double(vector.x()))
			&& std::isfinite(CGAL::to_double(vector.y()))
			&& std::isfinite(CGAL::to_double(vector.z()));
	}

	void addFaceIssue(std::map<std::size_t, std::uint32_t>& faceIssues, const Face face, const std::uint32_t issue)
	{
		faceIssues[static_cast<std::size_t>(face.idx())] |= issue;
	}

	void appendInspectionError(poca::geometry::MeshInspection& inspection, const std::string& check, const std::exception& exception)
	{
		inspection.inspectionErrors.push_back(check + ": " + exception.what());
	}

	bool basicStageValid(const poca::geometry::MeshInspection& inspection)
	{
		return !inspection.empty
			&& inspection.finiteCoordinates
			&& inspection.validPolygonMesh
			&& inspection.triangleMesh
			&& inspection.inspectionErrors.empty();
	}

	std::set<Face> facePatch(const Mesh& mesh, const std::set<std::size_t>& faceIndices)
	{
		std::set<Face> patch;
		for (const auto face : mesh.faces())
			if (faceIndices.count(static_cast<std::size_t>(face.idx())) != 0) patch.insert(face);
		return patch;
	}

	void expandFacePatch(const Mesh& mesh, std::set<Face>& patch)
	{
		auto expanded = patch;
		for (const auto patchFace : patch) {
			for (const auto halfedge : halfedges_around_face(mesh.halfedge(patchFace), mesh)) {
				const auto neighbor = face(opposite(halfedge, mesh), mesh);
				if (neighbor != Mesh::null_face()) expanded.insert(neighbor);
			}
		}
		patch.swap(expanded);
	}

	void removeFacePatch(Mesh& mesh, const std::set<std::size_t>& faceIndices)
	{
		const auto patch = facePatch(mesh, faceIndices);
		if (patch.size() != faceIndices.size())
			throw std::runtime_error("could not resolve every source patch face on the fresh mesh copy");
		for (const auto patchFace : patch)
			CGAL::Euler::remove_face(mesh.halfedge(patchFace), mesh);
		PMP::remove_isolated_vertices(mesh);
		mesh.collect_garbage();
	}

	bool structurallyValidBeforeOrientation(
		const poca::geometry::MeshInspection& inspection,
		std::vector<std::string>& failures,
		const bool requireNoSelfIntersections = true)
	{
		failures = inspection.inspectionErrors;
		if (inspection.empty) failures.emplace_back("mesh is empty");
		if (!inspection.finiteCoordinates) failures.emplace_back("coordinates are not finite");
		if (!inspection.validPolygonMesh) failures.emplace_back("CGAL::is_valid_polygon_mesh = false");
		if (!inspection.triangleMesh) failures.emplace_back("CGAL::is_triangle_mesh = false");
		if (inspection.degenerateTriangleCount != 0) failures.emplace_back(std::to_string(inspection.degenerateTriangleCount) + " degenerate triangle(s)");
		if (!inspection.closed) failures.emplace_back("mesh is open");
		if (inspection.borderEdgeCount != 0) failures.emplace_back(std::to_string(inspection.borderEdgeCount) + " border edge(s)");
		if (inspection.connectedComponentCount != 1) failures.emplace_back(std::to_string(inspection.connectedComponentCount) + " connected component(s), expected one");
		if (requireNoSelfIntersections && !inspection.selfIntersectionPairs.empty()) failures.emplace_back(std::to_string(inspection.selfIntersectionPairs.size()) + " true self-intersection pair(s)");
		if (inspection.nonManifoldVertexCount != 0) failures.emplace_back(std::to_string(inspection.nonManifoldVertexCount) + " non-manifold vertex/vertices");
		if (!inspection.normalsChecked || !inspection.finiteFaceNormals || !inspection.finiteVertexNormals) failures.emplace_back("face/vertex normals are not all finite");
		return failures.empty();
	}

	bool repairSelfIntersectionClusterLocally(
		const Mesh& source,
		const std::set<std::size_t>& offendingFaceIndices,
		const std::size_t clusterIndex,
		const std::size_t clusterPairCount,
		const std::size_t sourcePairCount,
		Mesh& repaired,
		poca::geometry::MeshRepairCounts& counts,
		std::vector<std::string>& steps,
		std::vector<std::string>& failures)
	{
		auto patch = facePatch(source, offendingFaceIndices);
		if (patch.size() != offendingFaceIndices.size()) {
			failures.emplace_back("cluster " + std::to_string(clusterIndex) + ": could not resolve every self-intersecting face on the current-stage mesh");
			return false;
		}

		for (std::size_t ring = 0; ring <= 3; ++ring) {
			if (ring != 0) expandFacePatch(source, patch);
			std::set<std::size_t> patchIndices;
			for (const auto patchFace : patch) patchIndices.insert(static_cast<std::size_t>(patchFace.idx()));

			try {
				Mesh candidate = source;
				removeFacePatch(candidate, patchIndices);
				if (!CGAL::is_valid_polygon_mesh(candidate)) {
					steps.push_back("Cluster " + std::to_string(clusterIndex) + " ring " + std::to_string(ring) + " rejected: face removal produced invalid polygon topology.");
					continue;
				}

				std::vector<Halfedge> boundaryLoops;
				PMP::extract_boundary_cycles(candidate, std::back_inserter(boundaryLoops));
				if (boundaryLoops.size() != 1) {
					steps.push_back("Cluster " + std::to_string(clusterIndex) + " ring " + std::to_string(ring) + " rejected: local patch produced " + std::to_string(boundaryLoops.size()) + " boundary loops instead of one.");
					continue;
				}

				std::vector<Face> createdFaces;
				PMP::triangulate_hole(candidate, boundaryLoops.front(),
					CGAL::parameters::face_output_iterator(std::back_inserter(createdFaces)));
				if (createdFaces.empty()) {
					steps.push_back("Cluster " + std::to_string(clusterIndex) + " ring " + std::to_string(ring) + " rejected: triangulate_hole created no faces.");
					continue;
				}

				const auto inspection = poca::geometry::MeshRepair::inspect(candidate);
				std::vector<std::string> candidateFailures;
				if (!structurallyValidBeforeOrientation(inspection, candidateFailures, false)) {
					std::ostringstream reason;
					reason << "Cluster " << clusterIndex << " ring " << ring << " rejected after validation";
					if (!candidateFailures.empty()) reason << ": " << candidateFailures.front();
					steps.push_back(reason.str());
					continue;
				}
				const std::size_t maximumRemainingPairCount = sourcePairCount - clusterPairCount;
				if (inspection.selfIntersectionPairs.size() > maximumRemainingPairCount) {
					steps.push_back("Cluster " + std::to_string(clusterIndex) + " ring " + std::to_string(ring)
						+ " rejected: it did not eliminate the selected cluster's pair count without introducing replacements.");
					continue;
				}

				repaired = std::move(candidate);
				++counts.repairedSelfIntersectionClusters;
				counts.repairedSelfIntersectionPairs += sourcePairCount - inspection.selfIntersectionPairs.size();
				counts.removedSelfIntersectionPatchFaces += patchIndices.size();
				counts.createdSelfIntersectionPatchFaces += createdFaces.size();
				counts.selectedSelfIntersectionRing = ring;
				steps.push_back("Cluster " + std::to_string(clusterIndex) + " repaired with local ring " + std::to_string(ring)
					+ ": removed " + std::to_string(patchIndices.size()) + " faces and created "
					+ std::to_string(createdFaces.size()) + " hole-fill faces.");
				return true;
			}
			catch (const std::bad_alloc&) {
				throw;
			}
			catch (const std::exception& exception) {
				steps.push_back("Cluster " + std::to_string(clusterIndex) + " ring " + std::to_string(ring) + " rejected by exception: " + exception.what());
			}
		}

		failures.emplace_back("cluster " + std::to_string(clusterIndex) + ": no local self-intersection patch in rings 0..3 passed validation");
		return false;
	}
}

namespace poca::geometry {
	std::vector<std::vector<std::size_t>> MeshRepair::clusterIntersectionPairs(
		const std::vector<std::pair<std::size_t, std::size_t>>& pairs)
	{
		std::map<std::size_t, std::set<std::size_t>> adjacency;
		for (const auto& pair : pairs) {
			adjacency[pair.first].insert(pair.second);
			adjacency[pair.second].insert(pair.first);
		}

		std::set<std::size_t> remaining;
		for (const auto& entry : adjacency) remaining.insert(entry.first);
		std::vector<std::vector<std::size_t>> clusters;
		while (!remaining.empty()) {
			std::vector<std::size_t> pending{ *remaining.begin() };
			remaining.erase(remaining.begin());
			std::vector<std::size_t> cluster;
			while (!pending.empty()) {
				const auto faceIndex = pending.back();
				pending.pop_back();
				cluster.push_back(faceIndex);
				for (const auto neighbor : adjacency.at(faceIndex)) {
					const auto found = remaining.find(neighbor);
					if (found == remaining.end()) continue;
					pending.push_back(neighbor);
					remaining.erase(found);
				}
			}
			std::sort(cluster.begin(), cluster.end());
			clusters.push_back(std::move(cluster));
		}
		std::sort(clusters.begin(), clusters.end(), [](const auto& left, const auto& right) {
			return left.front() < right.front();
			});
		return clusters;
	}

	MeshInspection MeshRepair::inspect(const Surface_mesh_3_double& source)
	{
		MeshInspection result;
		std::map<std::size_t, std::uint32_t> faceIssues;
		Mesh mesh = source;

		result.vertexCount = mesh.number_of_vertices();
		result.edgeCount = mesh.number_of_edges();
		result.faceCount = mesh.number_of_faces();
		result.empty = mesh.is_empty();
		if (result.empty) result.issueMask |= MeshIssueEmpty;

		result.finiteCoordinates = true;
		std::set<std::size_t> nonFiniteVertices;
		for (const auto vertex : mesh.vertices()) {
			if (!finitePoint(mesh.point(vertex))) {
				result.finiteCoordinates = false;
				nonFiniteVertices.insert(static_cast<std::size_t>(vertex.idx()));
			}
		}
		if (!result.finiteCoordinates) result.issueMask |= MeshIssueNonFinite;

		result.validPolygonMesh = !result.empty && CGAL::is_valid_polygon_mesh(mesh);
		if (!result.validPolygonMesh) result.issueMask |= MeshIssueInvalidPolygonMesh;

		result.triangleMesh = result.validPolygonMesh && CGAL::is_triangle_mesh(mesh);
		if (result.validPolygonMesh) {
			for (const auto face : mesh.faces()) {
				const auto vertices = faceVertices(mesh, face);
				bool nonFiniteFace = false;
				for (const auto vertex : vertices) {
					if (nonFiniteVertices.count(static_cast<std::size_t>(vertex.idx())) != 0) {
						nonFiniteFace = true;
						break;
					}
				}
				if (nonFiniteFace) addFaceIssue(faceIssues, face, MeshIssueNonFinite);
				if (vertices.size() != 3) {
					++result.nonTriangleFaceCount;
					addFaceIssue(faceIssues, face, MeshIssueNonTriangle);
					continue;
				}
				const auto& a = mesh.point(vertices[0]);
				const auto& b = mesh.point(vertices[1]);
				const auto& c = mesh.point(vertices[2]);
				if (finitePoint(a) && finitePoint(b) && finitePoint(c) && CGAL::collinear(a, b, c)) {
					++result.degenerateTriangleCount;
					addFaceIssue(faceIssues, face, MeshIssueDegenerate);
				}
			}
		}
		if (result.validPolygonMesh && !result.triangleMesh) result.issueMask |= MeshIssueNonTriangle;
		if (result.degenerateTriangleCount != 0) result.issueMask |= MeshIssueDegenerate;

		if (result.validPolygonMesh) {
			std::set<std::size_t> borderFaces;
			for (const auto edge : mesh.edges()) {
				if (!is_border(edge, mesh)) continue;
				++result.borderEdgeCount;
				auto halfedge = mesh.halfedge(edge);
				if (is_border(halfedge, mesh)) halfedge = opposite(halfedge, mesh);
				const auto borderFace = face(halfedge, mesh);
				if (borderFace != Mesh::null_face()) {
					borderFaces.insert(static_cast<std::size_t>(borderFace.idx()));
					addFaceIssue(faceIssues, borderFace, MeshIssueBoundary);
				}
			}
			result.borderFaceCount = borderFaces.size();
			result.closed = CGAL::is_closed(mesh);
			if (!result.closed || result.borderEdgeCount != 0) result.issueMask |= MeshIssueBoundary;

			try {
				auto componentMap = mesh.add_property_map<Face, std::size_t>("f:poca_mesh_repair_cc", 0).first;
				result.connectedComponentCount = PMP::connected_components(mesh, componentMap);
				result.connectedComponentFaceCounts.assign(result.connectedComponentCount, 0);
				for (const auto face : mesh.faces())
					if (componentMap[face] < result.connectedComponentFaceCounts.size()) ++result.connectedComponentFaceCounts[componentMap[face]];
				if (result.connectedComponentCount > 1) {
					result.issueMask |= MeshIssueDisconnected;
					const auto largest = static_cast<std::size_t>(std::distance(result.connectedComponentFaceCounts.begin(),
						std::max_element(result.connectedComponentFaceCounts.begin(), result.connectedComponentFaceCounts.end())));
					for (const auto face : mesh.faces()) {
						if (componentMap[face] == largest) continue;
						++result.facesOutsideLargestComponent;
						addFaceIssue(faceIssues, face, MeshIssueDisconnected);
					}
				}
				mesh.remove_property_map(componentMap);
			}
			catch (const std::bad_alloc&) {
				throw;
			}
			catch (const std::exception& exception) {
				appendInspectionError(result, "connected-component check", exception);
			}

			try {
				std::vector<Halfedge> nonManifoldHalfedges;
				PMP::non_manifold_vertices(mesh, std::back_inserter(nonManifoldHalfedges));
				std::set<std::size_t> nonManifoldVertices;
				for (const auto halfedge : nonManifoldHalfedges) {
					nonManifoldVertices.insert(static_cast<std::size_t>(target(halfedge, mesh).idx()));
					const auto firstFace = face(halfedge, mesh);
					const auto secondFace = face(opposite(halfedge, mesh), mesh);
					if (firstFace != Mesh::null_face()) addFaceIssue(faceIssues, firstFace, MeshIssueNonManifold);
					if (secondFace != Mesh::null_face()) addFaceIssue(faceIssues, secondFace, MeshIssueNonManifold);
				}
				result.nonManifoldVertexCount = nonManifoldVertices.size();
				if (result.nonManifoldVertexCount != 0) result.issueMask |= MeshIssueNonManifold;
			}
			catch (const std::bad_alloc&) {
				throw;
			}
			catch (const std::exception& exception) {
				appendInspectionError(result, "non-manifold vertex check", exception);
			}
		}

		if (result.validPolygonMesh && result.triangleMesh && result.finiteCoordinates) {
			try {
				std::vector<std::pair<Face, Face>> rawPairs;
				PMP::self_intersections<CGAL::Sequential_tag>(faces(mesh), mesh, std::back_inserter(rawPairs));
				for (const auto& pair : rawPairs) {
					// CGAL uses (f,f) for a degenerate face; it is not a true crossing.
					if (pair.first == pair.second) {
						addFaceIssue(faceIssues, pair.first, MeshIssueDegenerate);
						continue;
					}
					result.selfIntersectionPairs.emplace_back(
						static_cast<std::size_t>(pair.first.idx()), static_cast<std::size_t>(pair.second.idx()));
					addFaceIssue(faceIssues, pair.first, MeshIssueSelfIntersection);
					addFaceIssue(faceIssues, pair.second, MeshIssueSelfIntersection);
				}
				if (!result.selfIntersectionPairs.empty()) result.issueMask |= MeshIssueSelfIntersection;
			}
			catch (const std::bad_alloc&) {
				throw;
			}
			catch (const std::exception& exception) {
				appendInspectionError(result, "self-intersection check", exception);
			}

			try {
				result.normalsChecked = true;
				auto faceNormals = mesh.add_property_map<Face, Vector>("f:poca_mesh_repair_normal", CGAL::NULL_VECTOR).first;
				auto vertexNormals = mesh.add_property_map<Vertex, Vector>("v:poca_mesh_repair_normal", CGAL::NULL_VECTOR).first;
				PMP::compute_face_normals(mesh, faceNormals);
				PMP::compute_vertex_normals(mesh, vertexNormals);
				std::set<std::size_t> badVertices;
				for (const auto vertex : mesh.vertices()) {
					if (!finiteVector(vertexNormals[vertex])) {
						badVertices.insert(static_cast<std::size_t>(vertex.idx()));
						++result.nonFiniteVertexNormalCount;
					}
				}
				for (const auto face : mesh.faces()) {
					const bool badFaceNormal = !finiteVector(faceNormals[face]);
					if (badFaceNormal) ++result.nonFiniteFaceNormalCount;
					bool affected = badFaceNormal;
					if (!affected) {
						for (const auto vertex : faceVertices(mesh, face)) {
							if (badVertices.count(static_cast<std::size_t>(vertex.idx())) != 0) {
								affected = true;
								break;
							}
						}
					}
					if (affected) {
						++result.facesWithNonFiniteNormals;
						addFaceIssue(faceIssues, face, MeshIssueNonFiniteNormal);
					}
				}
				result.finiteFaceNormals = result.nonFiniteFaceNormalCount == 0;
				result.finiteVertexNormals = result.nonFiniteVertexNormalCount == 0;
				if (!result.finiteFaceNormals || !result.finiteVertexNormals) result.issueMask |= MeshIssueNonFiniteNormal;
				mesh.remove_property_map(faceNormals);
				mesh.remove_property_map(vertexNormals);
			}
			catch (const std::bad_alloc&) {
				throw;
			}
			catch (const std::exception& exception) {
				appendInspectionError(result, "normal check", exception);
			}
		}

		const bool volumePrerequisites = result.validPolygonMesh
			&& result.triangleMesh
			&& result.finiteCoordinates
			&& result.inspectionErrors.empty()
			&& result.closed
			&& result.borderEdgeCount == 0
			&& result.connectedComponentCount == 1
			&& result.degenerateTriangleCount == 0
			&& result.selfIntersectionPairs.empty()
			&& result.nonManifoldVertexCount == 0;
		if (volumePrerequisites) {
			try {
				result.orientationChecked = true;
				result.outwardOriented = PMP::is_outward_oriented(mesh);
				if (!result.outwardOriented) result.issueMask |= MeshIssueOrientation;
				result.boundsVolumeChecked = true;
				result.boundsVolume = PMP::does_bound_a_volume(mesh);
				result.volumeChecked = true;
				result.volume = CGAL::to_double(PMP::volume(mesh));
				result.finiteVolume = std::isfinite(result.volume);
				if (!result.boundsVolume || !result.finiteVolume || result.volume <= 0.0) result.issueMask |= MeshIssueVolume;
			}
			catch (const std::bad_alloc&) {
				throw;
			}
			catch (const std::exception& exception) {
				appendInspectionError(result, "orientation/volume check", exception);
			}
		}

		for (const auto& issue : faceIssues) result.problemFaces.push_back({ issue.first, issue.second });
		return result;
	}

	std::vector<std::string> MeshRepair::strictValidationFailures(const MeshInspection& inspection)
	{
		std::vector<std::string> failures = inspection.inspectionErrors;
		if (inspection.empty) failures.emplace_back("mesh is empty");
		if (!inspection.finiteCoordinates) failures.emplace_back("coordinates are not finite");
		if (!inspection.validPolygonMesh) failures.emplace_back("CGAL::is_valid_polygon_mesh = false");
		if (!inspection.triangleMesh) failures.emplace_back("CGAL::is_triangle_mesh = false");
		if (inspection.degenerateTriangleCount != 0) failures.emplace_back(std::to_string(inspection.degenerateTriangleCount) + " degenerate triangle(s)");
		if (!inspection.closed) failures.emplace_back("mesh is open");
		if (inspection.borderEdgeCount != 0) failures.emplace_back(std::to_string(inspection.borderEdgeCount) + " border edge(s)");
		if (inspection.connectedComponentCount != 1) failures.emplace_back(std::to_string(inspection.connectedComponentCount) + " connected component(s), expected one");
		if (!inspection.selfIntersectionPairs.empty()) failures.emplace_back(std::to_string(inspection.selfIntersectionPairs.size()) + " true self-intersection pair(s)");
		if (inspection.nonManifoldVertexCount != 0) failures.emplace_back(std::to_string(inspection.nonManifoldVertexCount) + " non-manifold vertex/vertices");
		if (!inspection.normalsChecked || !inspection.finiteFaceNormals || !inspection.finiteVertexNormals) failures.emplace_back("face/vertex normals are not all finite");
		if (!inspection.orientationChecked || !inspection.outwardOriented) failures.emplace_back("mesh orientation is not outward");
		if (!inspection.boundsVolumeChecked || !inspection.boundsVolume) failures.emplace_back("mesh does not bound a volume");
		if (!inspection.volumeChecked || !inspection.finiteVolume || inspection.volume <= 0.0) failures.emplace_back("mesh volume is not finite and positive");
		return failures;
	}

	bool MeshRepair::isStrictlyValid(const MeshInspection& inspection)
	{
		return strictValidationFailures(inspection).empty();
	}

	MeshRepairResult MeshRepair::repair(const Surface_mesh_3_double& source)
	{
		MeshRepairResult result;
		try {
			result.before = inspect(source);
			result.after = result.before;
			if (isStrictlyValid(result.before)) {
				result.status = MeshRepairStatus::Clean;
				result.steps.emplace_back("No repair required.");
				return result;
			}
			if (result.before.empty) {
				result.failures.emplace_back("empty meshes are not safely repairable");
				return result;
			}
			if (!result.before.finiteCoordinates) {
				result.failures.emplace_back("non-finite coordinates are not safely repairable");
				return result;
			}
			if (!result.before.validPolygonMesh) {
				result.failures.emplace_back("invalid polygon topology has no conservative generic repair");
				return result;
			}
			if (!result.before.inspectionErrors.empty()) {
				result.failures.emplace_back("source inspection did not complete successfully");
				return result;
			}

			Mesh candidate = source;
			MeshInspection current = result.before;
			std::uint32_t candidateRepairMask = MeshRepairNone;

			if (!current.triangleMesh) {
				result.attemptedRepairMask |= MeshRepairTriangulatedFaces;
				const std::size_t facesBefore = candidate.number_of_faces();
				PMP::triangulate_faces(candidate);
				current = inspect(candidate);
				if (!basicStageValid(current)) {
					result.failures.emplace_back("triangulate_faces did not produce a finite valid triangle mesh");
					result.after = current;
					return result;
				}
				candidateRepairMask |= MeshRepairTriangulatedFaces;
				result.counts.triangulatedFaces = result.before.nonTriangleFaceCount;
				result.steps.push_back("Triangulated " + std::to_string(result.before.nonTriangleFaceCount)
					+ " non-triangular face(s); face count changed from " + std::to_string(facesBefore)
					+ " to " + std::to_string(candidate.number_of_faces()) + ".");
			}

			if (current.degenerateTriangleCount != 0) {
				result.attemptedRepairMask |= MeshRepairDegenerateFaces;
				const std::size_t degeneratesBefore = current.degenerateTriangleCount;
				PMP::remove_degenerate_faces(candidate);
				PMP::remove_isolated_vertices(candidate);
				candidate.collect_garbage();
				current = inspect(candidate);
				if (!basicStageValid(current) || current.degenerateTriangleCount != 0) {
					result.failures.emplace_back("remove_degenerate_faces did not conservatively eliminate the degenerate-face defect");
					result.after = current;
					return result;
				}
				candidateRepairMask |= MeshRepairDegenerateFaces;
				result.counts.removedDegenerateFaces = degeneratesBefore - current.degenerateTriangleCount;
				result.steps.push_back("Removed " + std::to_string(result.counts.removedDegenerateFaces) + " degenerate face(s) and isolated vertices.");
			}

			if (current.connectedComponentCount > 1) {
				result.attemptedRepairMask |= MeshRepairDisconnectedComponents;
				const std::size_t componentsBefore = current.connectedComponentCount;
				const std::size_t facesBefore = candidate.number_of_faces();
				PMP::keep_largest_connected_components(candidate, 1);
				PMP::remove_isolated_vertices(candidate);
				candidate.collect_garbage();
				current = inspect(candidate);
				if (!basicStageValid(current) || current.connectedComponentCount != 1) {
					result.failures.emplace_back("keeping the largest connected component did not produce one valid component");
					result.after = current;
					return result;
				}
				candidateRepairMask |= MeshRepairDisconnectedComponents;
				result.counts.removedComponents = componentsBefore - 1;
				result.counts.removedComponentFaces = facesBefore - candidate.number_of_faces();
				result.steps.push_back("Kept the largest face-connected component; removed "
					+ std::to_string(result.counts.removedComponents) + " component(s) and "
					+ std::to_string(result.counts.removedComponentFaces) + " face(s).");
			}

			if (!current.closed || current.borderEdgeCount != 0) {
				if (current.nonManifoldVertexCount != 0) {
					result.failures.emplace_back("border repair is not attempted on a non-manifold mesh because CGAL border stitching requires manifold input");
					result.after = current;
					return result;
				}
				result.attemptedRepairMask |= MeshRepairBoundary;
				const std::size_t bordersBefore = current.borderEdgeCount;
				result.counts.stitchedBorderPairs = PMP::stitch_borders(candidate);
				current = inspect(candidate);
				if (!basicStageValid(current)) {
					result.failures.emplace_back("stitch_borders produced invalid geometry");
					result.after = current;
					return result;
				}

				if (current.borderEdgeCount != 0) {
					std::vector<Halfedge> boundaryLoops;
					PMP::extract_boundary_cycles(candidate, std::back_inserter(boundaryLoops));
					result.counts.boundaryLoops = boundaryLoops.size();
					for (const auto boundary : boundaryLoops) {
						std::vector<Face> createdFaces;
						PMP::triangulate_hole(candidate, boundary,
							CGAL::parameters::face_output_iterator(std::back_inserter(createdFaces)));
						if (createdFaces.empty()) {
							result.failures.emplace_back("triangulate_hole could not fill every remaining boundary loop");
							result.after = inspect(candidate);
							return result;
						}
						++result.counts.filledHoles;
						result.counts.createdHoleFaces += createdFaces.size();
					}
					current = inspect(candidate);
				}
				if (!basicStageValid(current) || !current.closed || current.borderEdgeCount != 0) {
					result.failures.emplace_back("border stitching and local hole triangulation did not close the mesh safely");
					result.after = current;
					return result;
				}
				candidateRepairMask |= MeshRepairBoundary;
				result.steps.push_back("Repaired " + std::to_string(bordersBefore) + " initial border edge(s): stitched "
					+ std::to_string(result.counts.stitchedBorderPairs) + " border pair(s), filled "
					+ std::to_string(result.counts.filledHoles) + " hole(s), and created "
					+ std::to_string(result.counts.createdHoleFaces) + " face(s).");
			}

			if (current.nonManifoldVertexCount != 0) {
				result.attemptedRepairMask |= MeshRepairNonManifoldTopology;
				const std::size_t nonManifoldBefore = current.nonManifoldVertexCount;
				result.counts.duplicatedNonManifoldVertices = PMP::duplicate_non_manifold_vertices(candidate);
				current = inspect(candidate);
				if (!basicStageValid(current)
					|| current.nonManifoldVertexCount != 0
					|| current.connectedComponentCount != 1
					|| !current.closed
					|| current.borderEdgeCount != 0
					|| !current.selfIntersectionPairs.empty()) {
					result.failures.emplace_back("duplicate_non_manifold_vertices introduced or retained unsafe topology");
					result.after = current;
					return result;
				}
				candidateRepairMask |= MeshRepairNonManifoldTopology;
				result.steps.push_back("Resolved " + std::to_string(nonManifoldBefore)
					+ " non-manifold vertex/vertices by duplicating "
					+ std::to_string(result.counts.duplicatedNonManifoldVertices) + " vertex occurrence group(s).");
			}

			// Topology-changing stages invalidate old face descriptors. The current
			// inspection above is therefore the only source of self-intersection faces.
			if (!current.selfIntersectionPairs.empty()) {
				result.attemptedRepairMask |= MeshRepairSelfIntersection;
				std::size_t clusterIndex = 0;
				while (!current.selfIntersectionPairs.empty()) {
					const auto clusters = clusterIntersectionPairs(current.selfIntersectionPairs);
					if (clusters.empty()) {
						result.failures.emplace_back("self-intersection pairs were present but no face cluster could be formed");
						result.after = current;
						return result;
					}

					std::vector<std::size_t> clusterPairCounts;
					clusterPairCounts.reserve(clusters.size());
					std::ostringstream clusterSummary;
					clusterSummary << "Self-intersection clusters detected: " << clusters.size() << " [faces/pairs: ";
					for (std::size_t currentCluster = 0; currentCluster < clusters.size(); ++currentCluster) {
						const auto& cluster = clusters[currentCluster];
						const std::size_t pairCount = static_cast<std::size_t>(std::count_if(
							current.selfIntersectionPairs.begin(), current.selfIntersectionPairs.end(),
							[&cluster](const auto& pair) {
								return std::binary_search(cluster.begin(), cluster.end(), pair.first)
									&& std::binary_search(cluster.begin(), cluster.end(), pair.second);
							}));
						clusterPairCounts.push_back(pairCount);
						if (currentCluster != 0) clusterSummary << ", ";
						clusterSummary << cluster.size() << "/" << pairCount;
					}
					clusterSummary << "].";
					result.steps.push_back(clusterSummary.str());

					const std::set<std::size_t> clusterFaces(clusters.front().begin(), clusters.front().end());
					const std::size_t clusterPairCount = clusterPairCounts.front();
					result.steps.push_back("Cluster " + std::to_string(clusterIndex) + ": "
						+ std::to_string(clusterFaces.size()) + " face(s), "
						+ std::to_string(clusterPairCount) + " pair(s); trying local rings 0..3.");

					Mesh locallyRepaired;
					if (!repairSelfIntersectionClusterLocally(candidate, clusterFaces, clusterIndex,
						clusterPairCount, current.selfIntersectionPairs.size(), locallyRepaired,
						result.counts, result.steps, result.failures)) {
						result.after = current;
						return result;
					}
					candidate = std::move(locallyRepaired);
					current = inspect(candidate);
					const auto remainingClusters = clusterIntersectionPairs(current.selfIntersectionPairs);
					result.steps.push_back("Recomputed self-intersections after cluster " + std::to_string(clusterIndex)
						+ ": " + std::to_string(current.selfIntersectionPairs.size()) + " pair(s) in "
						+ std::to_string(remainingClusters.size()) + " cluster(s) remain.");
					++clusterIndex;
				}
				candidateRepairMask |= MeshRepairSelfIntersection;
			}

			std::vector<std::string> structuralFailures;
			if (!structurallyValidBeforeOrientation(current, structuralFailures)) {
				result.failures.insert(result.failures.end(), structuralFailures.begin(), structuralFailures.end());
				result.after = current;
				return result;
			}

			if (!current.outwardOriented || !current.boundsVolume || !current.finiteVolume || current.volume <= 0.0) {
				result.attemptedRepairMask |= MeshRepairOrientation;
				PMP::orient_to_bound_a_volume(candidate);
				candidateRepairMask |= MeshRepairOrientation;
				result.steps.emplace_back("Oriented the repaired topology to bound a positive outward volume.");
			}

			result.after = inspect(candidate);
			const auto finalFailures = strictValidationFailures(result.after);
			if (!finalFailures.empty()) {
				result.failures.insert(result.failures.end(), finalFailures.begin(), finalFailures.end());
				return result;
			}
			if (candidateRepairMask == MeshRepairNone) {
				result.failures.emplace_back("the source was not clean but no conservative repair operation was applicable");
				return result;
			}

			result.repairedMesh = std::move(candidate);
			result.repairMask = candidateRepairMask;
			result.status = MeshRepairStatus::Repaired;
			return result;
		}
		catch (const std::bad_alloc&) {
			throw;
		}
		catch (const std::exception& exception) {
			result.failures.push_back(std::string("repair exception: ") + exception.what());
			return result;
		}
		catch (...) {
			result.failures.emplace_back("unknown repair exception");
			return result;
		}
	}
}
