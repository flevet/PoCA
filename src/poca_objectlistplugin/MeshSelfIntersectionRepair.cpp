/*
* Software:  PoCA: Point Cloud Analyst
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*/

#include "MeshSelfIntersectionRepair.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <map>
#include <new>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <CGAL/Cartesian_converter.h>
#include <CGAL/Kernel_traits.h>
#include <CGAL/Polygon_mesh_processing/autorefinement.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/tags.h>

namespace {
	namespace PMP = CGAL::Polygon_mesh_processing;

	template <typename MeshType>
	using FaceDescriptor = typename boost::graph_traits<MeshType>::face_descriptor;

	template <typename MeshType>
	using VertexDescriptor = typename boost::graph_traits<MeshType>::vertex_descriptor;

	template <typename MeshType>
	std::vector<VertexDescriptor<MeshType>> faceVertices(const MeshType& mesh, const FaceDescriptor<MeshType> face)
	{
		std::vector<VertexDescriptor<MeshType>> vertices;
		const auto halfedge = mesh.halfedge(face);
		if (halfedge == MeshType::null_halfedge()) return vertices;
		CGAL::Vertex_around_face_iterator<MeshType> begin, end;
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

	template <typename Vector>
	bool finiteVector(const Vector& vector)
	{
		return std::isfinite(CGAL::to_double(vector.x()))
			&& std::isfinite(CGAL::to_double(vector.y()))
			&& std::isfinite(CGAL::to_double(vector.z()));
	}

	struct MeshInspection {
		size_t vertices = 0;
		size_t edges = 0;
		size_t faces = 0;
		bool empty = true;
		bool valid = false;
		bool finite = false;
		bool triangle = false;
		bool closed = false;
		size_t borderEdges = 0;
		size_t connectedComponents = 0;
		size_t degenerateTriangles = 0;
		size_t nonManifoldVertices = 0;
		std::vector<std::pair<size_t, size_t>> selfIntersections;
		bool normalsChecked = false;
		size_t invalidNormalFaces = 0;
	};

	template <typename MeshType>
	MeshInspection inspectMesh(MeshType& mesh, const bool checkNormals)
	{
		using Face = FaceDescriptor<MeshType>;
		using Vertex = VertexDescriptor<MeshType>;
		using MeshKernel = typename CGAL::Kernel_traits<typename MeshType::Point>::Kernel;
		using Vector = typename MeshKernel::Vector_3;

		MeshInspection result;
		result.vertices = mesh.number_of_vertices();
		result.edges = mesh.number_of_edges();
		result.faces = mesh.number_of_faces();
		result.empty = mesh.is_empty();
		result.valid = !result.empty && CGAL::is_valid_polygon_mesh(mesh);

		result.finite = true;
		for (Vertex vertex : mesh.vertices()) {
			if (!finitePoint(mesh.point(vertex))) {
				result.finite = false;
				break;
			}
		}

		result.triangle = result.valid && CGAL::is_triangle_mesh(mesh);
		if (result.valid) {
			for (Face face : mesh.faces()) {
				const auto vertices = faceVertices(mesh, face);
				if (vertices.size() != 3) continue;
				const auto& a = mesh.point(vertices[0]);
				const auto& b = mesh.point(vertices[1]);
				const auto& c = mesh.point(vertices[2]);
				if (finitePoint(a) && finitePoint(b) && finitePoint(c) && CGAL::collinear(a, b, c))
					++result.degenerateTriangles;
			}
			for (const auto edge : mesh.edges())
				if (is_border(edge, mesh)) ++result.borderEdges;
			result.closed = CGAL::is_closed(mesh);

			auto componentMap = mesh.template add_property_map<Face, std::size_t>("f:poca_repair_cc", 0).first;
			result.connectedComponents = PMP::connected_components(mesh, componentMap);
			mesh.remove_property_map(componentMap);

			std::vector<typename boost::graph_traits<MeshType>::halfedge_descriptor> nonManifoldHalfedges;
			PMP::non_manifold_vertices(mesh, std::back_inserter(nonManifoldHalfedges));
			std::set<size_t> nonManifoldVertexIndices;
			for (const auto halfedge : nonManifoldHalfedges)
				nonManifoldVertexIndices.insert(static_cast<size_t>(target(halfedge, mesh).idx()));
			result.nonManifoldVertices = nonManifoldVertexIndices.size();
		}

		if (result.valid && result.triangle && result.finite) {
			std::vector<std::pair<Face, Face>> rawPairs;
			PMP::self_intersections<CGAL::Sequential_tag>(faces(mesh), mesh, std::back_inserter(rawPairs));
			for (const auto& pair : rawPairs) {
				// CGAL uses (f,f) for a degenerate face. It is not a true crossing.
				if (pair.first == pair.second) continue;
				result.selfIntersections.emplace_back(
					static_cast<size_t>(pair.first.idx()), static_cast<size_t>(pair.second.idx()));
			}
		}

		if (checkNormals && result.valid && result.triangle && result.finite) {
			result.normalsChecked = true;
			auto faceNormals = mesh.template add_property_map<Face, Vector>("f:poca_repair_normal", CGAL::NULL_VECTOR).first;
			auto vertexNormals = mesh.template add_property_map<Vertex, Vector>("v:poca_repair_normal", CGAL::NULL_VECTOR).first;
			PMP::compute_face_normals(mesh, faceNormals);
			PMP::compute_vertex_normals(mesh, vertexNormals);
			std::set<size_t> badVertices;
			for (Vertex vertex : mesh.vertices())
				if (!finiteVector(vertexNormals[vertex])) badVertices.insert(static_cast<size_t>(vertex.idx()));
			for (Face face : mesh.faces()) {
				bool invalid = !finiteVector(faceNormals[face]);
				if (!invalid) {
					for (Vertex vertex : faceVertices(mesh, face)) {
						if (badVertices.count(static_cast<size_t>(vertex.idx())) != 0) {
							invalid = true;
							break;
						}
					}
				}
				if (invalid) ++result.invalidNormalFaces;
			}
			mesh.remove_property_map(faceNormals);
			mesh.remove_property_map(vertexNormals);
		}

		return result;
	}

	void reportInspection(std::ostringstream& report, const MeshInspection& inspection, const bool includeNormals)
	{
		report << "vertices = " << inspection.vertices << "\n";
		report << "edges = " << inspection.edges << "\n";
		report << "faces = " << inspection.faces << "\n";
		report << "finite coordinates: " << (inspection.finite ? "PASS" : "FAIL") << "\n";
		report << "valid polygon mesh: " << (inspection.valid ? "PASS" : "FAIL") << "\n";
		report << "triangle mesh: " << (inspection.triangle ? "PASS" : "FAIL") << "\n";
		report << "degenerate triangles: " << inspection.degenerateTriangles << "\n";
		report << "closed: " << (inspection.closed ? "PASS" : "FAIL") << "\n";
		report << "border edges: " << inspection.borderEdges << "\n";
		report << "connected components: " << inspection.connectedComponents << "\n";
		report << "non-manifold vertices: " << inspection.nonManifoldVertices << "\n";
		report << "self intersections: " << inspection.selfIntersections.size() << "\n";
		if (includeNormals) {
			report << "finite face/vertex normals: ";
			if (!inspection.normalsChecked) report << "SKIPPED";
			else report << (inspection.invalidNormalFaces == 0 ? "PASS" : "FAIL")
				<< " (affected faces=" << inspection.invalidNormalFaces << ")";
			report << "\n";
		}
	}

	std::vector<std::string> strictFailures(const MeshInspection& inspection, const bool requireNormals, const std::string& conversionName)
	{
		std::vector<std::string> failures;
		if (inspection.empty) failures.emplace_back("mesh is empty");
		if (!inspection.finite) failures.emplace_back("coordinates are not finite/representable");
		if (!inspection.valid) failures.emplace_back("CGAL::is_valid_polygon_mesh = false");
		if (!inspection.triangle) failures.emplace_back("CGAL::is_triangle_mesh = false");
		if (!inspection.closed) failures.emplace_back("mesh is open");
		if (inspection.borderEdges != 0) failures.emplace_back(std::to_string(inspection.borderEdges) + " border edge(s)");
		if (inspection.connectedComponents != 1) failures.emplace_back(std::to_string(inspection.connectedComponents) + " connected component(s), expected 1");
		if (inspection.degenerateTriangles != 0) failures.emplace_back(std::to_string(inspection.degenerateTriangles) + " degenerate triangle(s)");
		if (!inspection.selfIntersections.empty()) {
			const std::string prefix = conversionName.empty() ? std::string() : conversionName + " ";
			failures.emplace_back(prefix + "has " + std::to_string(inspection.selfIntersections.size()) + " true self-intersection pair(s)");
		}
		if (inspection.nonManifoldVertices != 0) failures.emplace_back(std::to_string(inspection.nonManifoldVertices) + " non-manifold vertex/vertices");
		if (requireNormals && (!inspection.normalsChecked || inspection.invalidNormalFaces != 0))
			failures.emplace_back("face/vertex normals are not all finite");
		return failures;
	}

	template <typename SourceMesh, typename TargetMesh, typename Converter>
	bool convertTriangleMesh(const SourceMesh& source, TargetMesh& target, const Converter& converter, std::string& reason)
	{
		using SourceVertex = VertexDescriptor<SourceMesh>;
		using TargetVertex = VertexDescriptor<TargetMesh>;
		std::map<SourceVertex, TargetVertex> vertices;
		for (SourceVertex sourceVertex : source.vertices()) {
			const TargetVertex targetVertex = target.add_vertex(converter(source.point(sourceVertex)));
			vertices.emplace(sourceVertex, targetVertex);
		}

		for (const auto sourceFace : source.faces()) {
			const auto sourceVertices = faceVertices(source, sourceFace);
			if (sourceVertices.size() != 3) {
				reason = "encountered a non-triangle face during conversion";
				return false;
			}
			const auto face = target.add_face(vertices.at(sourceVertices[0]), vertices.at(sourceVertices[1]), vertices.at(sourceVertices[2]));
			if (face == TargetMesh::null_face()) {
				reason = "CGAL rejected a face while rebuilding the converted Surface_mesh";
				return false;
			}
		}
		return true;
	}

	void reportPairsAndFaces(std::ostringstream& report, const MeshInspection& inspection)
	{
		if (inspection.selfIntersections.empty()) return;
		report << "first pairs:\n";
		for (size_t index = 0; index < std::min<size_t>(10, inspection.selfIntersections.size()); ++index)
			report << "  (" << inspection.selfIntersections[index].first << ", " << inspection.selfIntersections[index].second << ")\n";
		std::set<size_t> faces;
		for (const auto& pair : inspection.selfIntersections) {
			faces.insert(pair.first);
			faces.insert(pair.second);
		}
		report << "Self-intersection source faces:\n";
		for (const size_t face : faces) report << "  " << face << "\n";
	}

	void reportFailure(std::ostringstream& report, const std::vector<std::string>& failures)
	{
		report << "RESULT: REJECTED\n";
		report << "Reason:\n";
		for (const std::string& failure : failures) report << "  " << failure << "\n";
	}
}

namespace poca::objectlist {
	MeshSelfIntersectionRepairResult repairSelfIntersections(const std::vector<Surface_mesh_3_double>& sources)
	{
		MeshSelfIntersectionRepairResult result;
		std::ostringstream report;
		report << std::setprecision(17);
		report << "Mesh repair - self intersections\n";
		report << "CGAL exact-kernel autorefinement is applied only to copies. Source meshes are never modified.\n\n";

		size_t meshesWithSelfIntersections = 0;
		size_t repairAttempted = 0;
		size_t repairedSuccessfully = 0;
		size_t rejectedAfterValidation = 0;
		size_t repairExceptions = 0;

		for (size_t meshIndex = 0; meshIndex < sources.size(); ++meshIndex) {
			report << "--------------------------------------------------\n";
			report << "Mesh " << meshIndex << "\n\n";
			try {
				Surface_mesh_3_double diagnosticCopy = sources[meshIndex];
				const MeshInspection before = inspectMesh(diagnosticCopy, true);
				report << "BEFORE\n";
				reportInspection(report, before, true);
				reportPairsAndFaces(report, before);
				if (before.valid && before.triangle && before.closed && before.finite && before.selfIntersections.empty()) {
					try {
						const bool outward = PMP::is_outward_oriented(diagnosticCopy);
						PMP::orient_to_bound_a_volume(diagnosticCopy);
						const double volume = CGAL::to_double(PMP::volume(diagnosticCopy));
						report << "outward oriented: " << (outward ? "yes" : "no (repairable on a copy)") << "\n";
						report << "orient_to_bound_a_volume + volume: "
							<< ((std::isfinite(volume) && volume > 0.0) ? "PASS" : "FAIL") << ", volume=" << volume << "\n";
					}
					catch (const std::exception& exception) {
						report << "orientation/volume: EXCEPTION: " << exception.what() << "\n";
					}
				}
				else report << "orientation/volume: SKIPPED (requires finite, valid, closed, non-self-intersecting triangle mesh)\n";
				report << "\n";

				if (!before.selfIntersections.empty()) ++meshesWithSelfIntersections;
				std::vector<std::string> prerequisites;
				if (before.empty) prerequisites.emplace_back("empty mesh");
				if (!before.finite) prerequisites.emplace_back("non-finite vertex coordinates");
				if (!before.valid) prerequisites.emplace_back("invalid polygon topology");
				if (!before.triangle) prerequisites.emplace_back("not a triangle mesh");
				if (!before.closed) prerequisites.emplace_back("open mesh");
				if (before.selfIntersections.empty()) prerequisites.emplace_back("no true self-intersection pairs");
				if (!prerequisites.empty()) {
					report << "REPAIR\nrepair not attempted:\n";
					for (const std::string& prerequisite : prerequisites) report << "  " << prerequisite << "\n";
					report << "RESULT: NOT ATTEMPTED\n\n";
					continue;
				}

				++repairAttempted;
				report << "REPAIR\n";
				Surface_mesh_3_exact exactMesh;
				std::string conversionFailure;
				const CGAL::Cartesian_converter<Kernel, K_exact> toExact;
				if (!convertTriangleMesh(sources[meshIndex], exactMesh, toExact, conversionFailure)) {
					report << "exact-kernel conversion: FAIL\n";
					++rejectedAfterValidation;
					reportFailure(report, { "exact-kernel conversion failed: " + conversionFailure });
					report << "\n";
					continue;
				}
				report << "exact-kernel conversion: PASS\n";

				PMP::autorefine(exactMesh);
				report << "autorefinement: PASS\n";
				report << "vertices after refinement = " << exactMesh.number_of_vertices() << "\n";
				report << "faces after refinement = " << exactMesh.number_of_faces() << "\n\n";

				MeshInspection exactInspection = inspectMesh(exactMesh, true);
				report << "EXACT POST-REPAIR VALIDATION\n";
				reportInspection(report, exactInspection, true);
				std::vector<std::string> failures = strictFailures(exactInspection, true, std::string());
				if (!failures.empty()) {
					report << "orientation modified: no (validation failed before orientation)\n";
					report << "manifold/volume checks: FAIL\n";
					++rejectedAfterValidation;
					reportFailure(report, failures);
					report << "\n";
					continue;
				}

				const bool initiallyOutward = PMP::is_outward_oriented(exactMesh);
				if (!initiallyOutward) PMP::orient_to_bound_a_volume(exactMesh);
				const bool orientationModified = !initiallyOutward;
				const bool exactOutward = PMP::is_outward_oriented(exactMesh);
				const bool exactBoundsVolume = PMP::does_bound_a_volume(exactMesh);
				const double exactVolume = CGAL::to_double(PMP::volume(exactMesh));
				report << "orientation modified: " << (orientationModified ? "yes" : "no") << "\n";
				report << "orientation: " << (exactOutward ? "PASS" : "FAIL") << "\n";
				report << "manifold/volume checks: " << (exactBoundsVolume ? "PASS" : "FAIL") << "\n";
				report << "volume = " << exactVolume << "\n\n";
				if (!exactOutward) failures.emplace_back("exact repaired mesh could not be established as outward-oriented");
				if (!exactBoundsVolume) failures.emplace_back("exact repaired mesh does not bound a valid volume");
				if (!std::isfinite(exactVolume) || exactVolume <= 0.0) failures.emplace_back("exact repaired mesh volume is not finite and positive");
				if (!failures.empty()) {
					++rejectedAfterValidation;
					reportFailure(report, failures);
					report << "\n";
					continue;
				}

				Surface_mesh_3_double doubleMesh;
				const CGAL::Cartesian_converter<K_exact, Kernel> toDouble;
				if (!convertTriangleMesh(exactMesh, doubleMesh, toDouble, conversionFailure)) {
					++rejectedAfterValidation;
					report << "DOUBLE CONVERSION VALIDATION\nconversion: FAIL\n";
					reportFailure(report, { "exact-to-double conversion failed: " + conversionFailure });
					report << "\n";
					continue;
				}

				MeshInspection doubleInspection = inspectMesh(doubleMesh, true);
				report << "DOUBLE CONVERSION VALIDATION\n";
				reportInspection(report, doubleInspection, true);
				failures = strictFailures(doubleInspection, true, "exact-to-double conversion");
				bool doubleOutward = false;
				bool doubleBoundsVolume = false;
				double doubleVolume = 0.0;
				if (failures.empty()) {
					doubleOutward = PMP::is_outward_oriented(doubleMesh);
					doubleBoundsVolume = PMP::does_bound_a_volume(doubleMesh);
					doubleVolume = CGAL::to_double(PMP::volume(doubleMesh));
					if (!doubleOutward) failures.emplace_back("double mesh is not outward-oriented");
					if (!doubleBoundsVolume) failures.emplace_back("double mesh does not bound a valid volume");
					if (!std::isfinite(doubleVolume) || doubleVolume <= 0.0) failures.emplace_back("double mesh volume is not finite and positive");
				}
				report << "orientation: " << (doubleOutward ? "PASS" : "FAIL") << "\n";
				report << "bounds a volume: " << (doubleBoundsVolume ? "PASS" : "FAIL") << "\n";
				report << "volume = " << doubleVolume << "\n";
				if (!failures.empty()) {
					++rejectedAfterValidation;
					reportFailure(report, failures);
					report << "\n";
					continue;
				}

				result.meshes.push_back(std::move(doubleMesh));
				result.sourceMeshIndices.push_back(static_cast<float>(meshIndex));
				++repairedSuccessfully;
				report << "OrganoGraph organoid-shape preflight: PASS\n";
				report << "RESULT: SUCCESS\n\n";
			}
			catch (const std::bad_alloc&) {
				throw;
			}
			catch (const std::exception& exception) {
				++repairExceptions;
				report << "RESULT: REJECTED\n";
				report << "Reason:\n  repair exception for mesh " << meshIndex << ": " << exception.what() << "\n\n";
			}
		}

		report << "==================================================\n";
		report << "Mesh repair summary\n\n";
		report << "source meshes: " << sources.size() << "\n";
		report << "meshes with self-intersections: " << meshesWithSelfIntersections << "\n";
		report << "repair attempted: " << repairAttempted << "\n";
		report << "repaired successfully: " << repairedSuccessfully << "\n";
		report << "rejected after validation: " << rejectedAfterValidation << "\n";
		report << "repair exceptions: " << repairExceptions << "\n";
		result.report = report.str();
		return result;
	}
}
