/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListBasicCommands.cpp
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

#include <fstream>
#include <iomanip>
#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <sstream>
#include <cmath>
#include <glm/gtc/quaternion.hpp>
#include <math.h>
#include <tinysplinecxx.h>
#include <CGAL/subdivision_method_3.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/boost/graph/helpers.h>

#include <QtCore/QString>
#include <QtCore/QFileInfo>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QLabel>

#include <General/Engine.hpp>
#include <Geometry/DetectionSet.hpp>
#include <Objects/MyObject.hpp>
#include <General/MyData.hpp>
#include <General/Histogram.hpp>
#include <Interfaces/PaletteInterface.hpp>
#include <General/Misc.h>
#include <Interfaces/CameraInterface.hpp>
#include <Geometry/ObjectListMesh.hpp>
#include <Geometry/ObjectListPolygon.hpp>
#include <General/PluginList.hpp>
#include <Geometry/ObjectLists.hpp>
#include <Geometry/BasicComputation.hpp>
#include <Geometry/CGAL_helpers.hpp>
#include <Geometry/GeometryCommandContext.hpp>
#include <Geometry/MeshRepair.hpp>
#include <Objects/ObjectCommandContext.hpp>

#include "ObjectListBasicCommands.hpp"
#include "ObjectListPlugin.hpp"


namespace {
	using Mesh = Surface_mesh_3_double;
	using Face = boost::graph_traits<Mesh>::face_descriptor;
	using Vertex = boost::graph_traits<Mesh>::vertex_descriptor;

	struct DiagnosticTriangle {
		std::array<poca::core::Vec3mf, 3> triangle;
		float sourceMesh = 0.f;
		float sourceFace = 0.f;
		float issueMask = 0.f;
	};

	bool finitePoint(const Point_3_double& point)
	{
		return std::isfinite(CGAL::to_double(point.x()))
			&& std::isfinite(CGAL::to_double(point.y()))
			&& std::isfinite(CGAL::to_double(point.z()));
	}

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

	void appendDiagnosticFace(const Mesh& mesh, const Face face, const size_t meshIndex, const unsigned int issueMask, std::vector<DiagnosticTriangle>& out)
	{
		const auto vertices = faceVertices(mesh, face);
		if (vertices.size() < 3) return;
		const Point_3_double p0 = mesh.point(vertices[0]);
		if (!finitePoint(p0)) return;
		for (size_t index = 1; index + 1 < vertices.size(); ++index) {
			const Point_3_double p1 = mesh.point(vertices[index]);
			const Point_3_double p2 = mesh.point(vertices[index + 1]);
			if (!finitePoint(p1) || !finitePoint(p2)) continue;
			DiagnosticTriangle diagnostic;
			diagnostic.triangle = {
				poca::core::Vec3mf(static_cast<float>(CGAL::to_double(p0.x())), static_cast<float>(CGAL::to_double(p0.y())), static_cast<float>(CGAL::to_double(p0.z()))),
				poca::core::Vec3mf(static_cast<float>(CGAL::to_double(p1.x())), static_cast<float>(CGAL::to_double(p1.y())), static_cast<float>(CGAL::to_double(p1.z()))),
				poca::core::Vec3mf(static_cast<float>(CGAL::to_double(p2.x())), static_cast<float>(CGAL::to_double(p2.y())), static_cast<float>(CGAL::to_double(p2.z())))
			};
			diagnostic.sourceMesh = static_cast<float>(meshIndex);
			diagnostic.sourceFace = static_cast<float>(face.idx());
			diagnostic.issueMask = static_cast<float>(issueMask);
			out.push_back(diagnostic);
		}
	}

	void showMeshReport(const std::string& report, const QString& title, const QString& description)
	{
		QDialog dialog;
		dialog.setWindowTitle(title);
		dialog.resize(900, 700);
		auto* layout = new QVBoxLayout(&dialog);
		auto* label = new QLabel(description, &dialog);
		layout->addWidget(label);
		auto* text = new QPlainTextEdit(QString::fromStdString(report), &dialog);
		text->setReadOnly(true);
		text->setLineWrapMode(QPlainTextEdit::NoWrap);
		layout->addWidget(text, 1);
		auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, &dialog);
		QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
		QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
		layout->addWidget(buttons);
		dialog.exec();
	}

	void reportInspection(std::ostringstream& report, const poca::geometry::MeshInspection& inspection)
	{
		report << "vertices=" << inspection.vertexCount << ", edges=" << inspection.edgeCount << ", faces=" << inspection.faceCount << "\n";
		report << "empty: " << (inspection.empty ? "FAIL" : "PASS") << "\n";
		report << "finite coordinates: " << (inspection.finiteCoordinates ? "PASS" : "FAIL") << "\n";
		report << "valid polygon mesh: " << (inspection.validPolygonMesh ? "PASS" : "FAIL") << "\n";
		report << "triangle mesh: " << (inspection.triangleMesh ? "PASS" : "FAIL")
			<< ", non-triangular faces=" << inspection.nonTriangleFaceCount << "\n";
		report << "degenerate triangles: " << inspection.degenerateTriangleCount << "\n";
		report << "closed: " << (inspection.closed ? "PASS" : "FAIL")
			<< ", border edges=" << inspection.borderEdgeCount << ", border faces=" << inspection.borderFaceCount << "\n";
		report << "connected components: " << inspection.connectedComponentCount;
		if (!inspection.connectedComponentFaceCounts.empty()) {
			report << " [faces:";
			for (size_t index = 0; index < inspection.connectedComponentFaceCounts.size(); ++index)
				report << (index ? "," : " ") << inspection.connectedComponentFaceCounts[index];
			report << "]";
		}
		report << "\n";
		report << "non-manifold vertices: " << inspection.nonManifoldVertexCount << "\n";
		report << "self intersections: " << inspection.selfIntersectionPairs.size() << "\n";
		if (!inspection.selfIntersectionPairs.empty()) {
			report << "  first face pairs: ";
			for (size_t index = 0; index < std::min<size_t>(10, inspection.selfIntersectionPairs.size()); ++index)
				report << (index ? ", " : "") << "(" << inspection.selfIntersectionPairs[index].first << "," << inspection.selfIntersectionPairs[index].second << ")";
			report << "\n";
		}
		report << "finite CGAL face normals: " << (inspection.normalsChecked ? (inspection.finiteFaceNormals ? "PASS" : "FAIL") : "SKIPPED")
			<< ", non-finite=" << inspection.nonFiniteFaceNormalCount << "\n";
		report << "finite CGAL vertex normals: " << (inspection.normalsChecked ? (inspection.finiteVertexNormals ? "PASS" : "FAIL") : "SKIPPED")
			<< ", non-finite=" << inspection.nonFiniteVertexNormalCount << "\n";
		report << "outward oriented: " << (inspection.orientationChecked ? (inspection.outwardOriented ? "PASS" : "FAIL") : "SKIPPED") << "\n";
		report << "bounds a volume: " << (inspection.boundsVolumeChecked ? (inspection.boundsVolume ? "PASS" : "FAIL") : "SKIPPED") << "\n";
		report << "volume: ";
		if (inspection.volumeChecked) report << inspection.volume << (inspection.finiteVolume ? "" : " (non-finite)");
		else report << "SKIPPED";
		report << "\n";
		for (const auto& error : inspection.inspectionErrors) report << "inspection exception: " << error << "\n";
	}

	void collectDiagnosticMessages(const poca::geometry::MeshInspection& inspection, std::vector<std::string>& errors, std::vector<std::string>& warnings)
	{
		for (const auto& exception : inspection.inspectionErrors) errors.push_back("Inspection failed: " + exception);
		if (inspection.empty) errors.emplace_back("Mesh is empty.");
		if (!inspection.finiteCoordinates) errors.emplace_back("Mesh has non-finite coordinates.");
		if (!inspection.validPolygonMesh) errors.emplace_back("CGAL::is_valid_polygon_mesh = false.");
		if (inspection.validPolygonMesh && !inspection.triangleMesh) errors.emplace_back("CGAL::is_triangle_mesh = false.");
		if (inspection.validPolygonMesh && !inspection.closed)
			errors.push_back("Mesh is open (" + std::to_string(inspection.borderEdgeCount) + " border edge(s)).");
		if (inspection.degenerateTriangleCount != 0)
			warnings.push_back(std::to_string(inspection.degenerateTriangleCount) + " degenerate/collinear triangle(s).");
		if (inspection.connectedComponentCount > 1)
			warnings.push_back(std::to_string(inspection.connectedComponentCount) + " connected components in one mesh.");
		if (inspection.nonManifoldVertexCount != 0)
			warnings.push_back(std::to_string(inspection.nonManifoldVertexCount) + " non-manifold vertex/vertices.");
		if (!inspection.selfIntersectionPairs.empty())
			errors.push_back(std::to_string(inspection.selfIntersectionPairs.size()) + " self-intersecting face pair(s).");
		if (inspection.normalsChecked && (!inspection.finiteFaceNormals || !inspection.finiteVertexNormals))
			warnings.push_back(std::to_string(inspection.facesWithNonFiniteNormals) + " face(s) have non-finite face/vertex normals.");
		if (inspection.orientationChecked && !inspection.outwardOriented)
			warnings.emplace_back("Closed mesh is inward oriented; Repair meshes can orient a copy.");
		if (inspection.boundsVolumeChecked && !inspection.boundsVolume)
			errors.emplace_back("Mesh does not bound a volume.");
		if (inspection.volumeChecked && (!inspection.finiteVolume || (inspection.outwardOriented && inspection.volume <= 0.0)))
			errors.emplace_back("Outward mesh volume is not finite and positive.");
	}

	void validateObjectListMeshes(poca::geometry::ObjectListMesh* objectList, poca::core::MyObjectInterface* owner, const poca::core::CommandInfo& command)
	{
		if (!objectList) {
			QMessageBox::warning(nullptr, "Object mesh quality", "The current ObjectList is not an ObjectListMesh.");
			return;
		}

		std::ostringstream report;
		report << std::setprecision(17);
		const auto& meshes = objectList->getMeshes();
		report << "Object: " << (owner ? owner->getName() : std::string("<unknown>")) << "\n";
		if (owner) report << "Folder: " << owner->getDir() << "\n";
		report << "Meshes: " << meshes.size() << "\n\n";
		report << "All geometric checks below come from poca::geometry::MeshRepair::inspect().\n";
		report << "The source ObjectListMesh is never modified.\n\n";

		std::vector<DiagnosticTriangle> diagnosticTriangles;
		size_t meshesWithErrors = 0, meshesWithWarnings = 0, cleanMeshes = 0;
		size_t totalSelfPairs = 0, totalDegenerate = 0, totalBoundaryFaces = 0, totalDisconnectedFaces = 0;

		for (size_t meshIndex = 0; meshIndex < meshes.size(); ++meshIndex) {
			report << "--------------------------------------------------\n";
			report << "Mesh " << meshIndex << "\n";
			try {
				const auto inspection = poca::geometry::MeshRepair::inspect(meshes[meshIndex]);
				reportInspection(report, inspection);
				std::vector<std::string> errors, warnings;
				collectDiagnosticMessages(inspection, errors, warnings);

				std::map<size_t, unsigned int> faceIssues;
				for (const auto& faceIssue : inspection.problemFaces) faceIssues[faceIssue.faceIndex] |= faceIssue.issueMask;
				for (const auto face : meshes[meshIndex].faces()) {
					const auto found = faceIssues.find(static_cast<size_t>(face.idx()));
					if (found != faceIssues.end() && found->second != 0)
						appendDiagnosticFace(meshes[meshIndex], face, meshIndex, found->second, diagnosticTriangles);
				}

				totalSelfPairs += inspection.selfIntersectionPairs.size();
				totalDegenerate += inspection.degenerateTriangleCount;
				totalBoundaryFaces += inspection.borderFaceCount;
				totalDisconnectedFaces += inspection.facesOutsideLargestComponent;
				if (!errors.empty()) {
					++meshesWithErrors;
					report << "RESULT: ERROR\n";
					for (const auto& error : errors) report << "  ERROR: " << error << "\n";
				}
				else if (!warnings.empty()) {
					++meshesWithWarnings;
					report << "RESULT: WARNING\n";
				}
				else {
					++cleanMeshes;
					report << "RESULT: PASS\n";
				}
				for (const auto& warning : warnings) report << "  WARNING: " << warning << "\n";
				report << "diagnostic problem faces exported from this mesh: " << faceIssues.size() << "\n\n";
			}
			catch (const std::bad_alloc&) {
				throw;
			}
			catch (const std::exception& exception) {
				++meshesWithErrors;
				report << "RESULT: ERROR\n  ERROR: inspection exception: " << exception.what() << "\n\n";
			}
			catch (...) {
				++meshesWithErrors;
				report << "RESULT: ERROR\n  ERROR: unknown inspection exception\n\n";
			}
		}

		report << "==================================================\n";
		report << "SUMMARY\n";
		report << "clean meshes: " << cleanMeshes << "\n";
		report << "meshes with warnings only: " << meshesWithWarnings << "\n";
		report << "meshes with errors: " << meshesWithErrors << "\n";
		report << "self-intersection pairs: " << totalSelfPairs << "\n";
		report << "degenerate triangles: " << totalDegenerate << "\n";
		report << "boundary faces: " << totalBoundaryFaces << "\n";
		report << "faces outside the largest connected component: " << totalDisconnectedFaces << "\n";
		report << "diagnostic triangle objects: " << diagnosticTriangles.size() << "\n";
		report << "issueMask bits: 1=selfIntersection, 2=degenerate, 4=boundary, 8=nonFinite, 16=nonTriangle, 32=disconnected, 64=nonFiniteNormal, 128=nonManifold, 256=invalidPolygonMesh, 512=orientation, 1024=volume, 2048=empty\n";

		if (!diagnosticTriangles.empty() && owner) {
			std::vector<std::array<poca::core::Vec3mf, 3>> triangles;
			std::vector<float> sourceMesh, sourceFace, issueMask, selfIntersection, degenerate, boundary, nonFinite, nonTriangle, disconnected, nonFiniteNormal, nonManifold;
			triangles.reserve(diagnosticTriangles.size());
			for (const auto& diagnostic : diagnosticTriangles) {
				triangles.push_back(diagnostic.triangle);
				sourceMesh.push_back(diagnostic.sourceMesh);
				sourceFace.push_back(diagnostic.sourceFace);
				issueMask.push_back(diagnostic.issueMask);
				const unsigned int mask = static_cast<unsigned int>(diagnostic.issueMask);
				selfIntersection.push_back(mask & poca::geometry::MeshIssueSelfIntersection ? 1.f : 0.f);
				degenerate.push_back(mask & poca::geometry::MeshIssueDegenerate ? 1.f : 0.f);
				boundary.push_back(mask & poca::geometry::MeshIssueBoundary ? 1.f : 0.f);
				nonFinite.push_back(mask & poca::geometry::MeshIssueNonFinite ? 1.f : 0.f);
				nonTriangle.push_back(mask & poca::geometry::MeshIssueNonTriangle ? 1.f : 0.f);
				disconnected.push_back(mask & poca::geometry::MeshIssueDisconnected ? 1.f : 0.f);
				nonFiniteNormal.push_back(mask & poca::geometry::MeshIssueNonFiniteNormal ? 1.f : 0.f);
				nonManifold.push_back(mask & poca::geometry::MeshIssueNonManifold ? 1.f : 0.f);
			}
			auto* debug = new poca::geometry::ObjectListMesh(triangles);
			debug->addFeature("sourceMeshIndex", poca::core::generateDataWithLogNoInteraction(sourceMesh));
			debug->addFeature("sourceFaceIndex", poca::core::generateDataWithLogNoInteraction(sourceFace));
			debug->addFeature("issueMask", poca::core::generateDataWithLogNoInteraction(issueMask));
			debug->addFeature("selfIntersection", poca::core::generateDataWithLogNoInteraction(selfIntersection));
			debug->addFeature("degenerate", poca::core::generateDataWithLogNoInteraction(degenerate));
			debug->addFeature("boundary", poca::core::generateDataWithLogNoInteraction(boundary));
			debug->addFeature("nonFinite", poca::core::generateDataWithLogNoInteraction(nonFinite));
			debug->addFeature("nonTriangle", poca::core::generateDataWithLogNoInteraction(nonTriangle));
			debug->addFeature("disconnected", poca::core::generateDataWithLogNoInteraction(disconnected));
			debug->addFeature("nonFiniteNormal", poca::core::generateDataWithLogNoInteraction(nonFiniteNormal));
			debug->addFeature("nonManifold", poca::core::generateDataWithLogNoInteraction(nonManifold));
			debug->setCurrentHistogramType("issueMask");
			ObjectListPlugin::m_plugins->addCommands(debug);
			auto* lists = dynamic_cast<poca::geometry::ObjectLists*>(owner->getBasicComponent("ObjectLists"));
			if (lists) {
				lists->addObjectList(debug, command, "ObjectListPlugin", "Mesh diagnostics - problem triangles");
				report << "Created ObjectList: Mesh diagnostics - problem triangles\n";
				report << "Each object is one offending source face (fan-triangulated only for display if the source face was non-triangular).\n";
				report << "Features: sourceMeshIndex, sourceFaceIndex, issueMask, selfIntersection, degenerate, boundary, nonFinite, nonTriangle, disconnected, nonFiniteNormal, nonManifold.\n";
			}
			else delete debug;
		}
		else report << "No problem triangles were identified for export.\n";

		const std::string text = report.str();
		std::cout << text << std::endl;
		showMeshReport(text, "Object mesh quality report", "CGAL mesh diagnostics for the current ObjectListMesh");
	}

	const char* repairStatusName(const poca::geometry::MeshRepairStatus status)
	{
		switch (status) {
		case poca::geometry::MeshRepairStatus::Clean: return "CLEAN - NO REPAIR REQUIRED";
		case poca::geometry::MeshRepairStatus::Repaired: return "REPAIRED";
		default: return "REPAIR REJECTED";
		}
	}

	void reportRepairDetails(std::ostringstream& report, const poca::geometry::MeshRepairResult& result)
	{
		report << "BEFORE\n";
		reportInspection(report, result.before);
		report << "\nDETECTED ISSUES\n";
		std::vector<std::string> detectedErrors, detectedWarnings;
		collectDiagnosticMessages(result.before, detectedErrors, detectedWarnings);
		if (detectedErrors.empty() && detectedWarnings.empty()) report << "  none\n";
		else {
			for (const auto& issue : detectedErrors) report << "  ERROR: " << issue << "\n";
			for (const auto& issue : detectedWarnings) report << "  WARNING: " << issue << "\n";
		}
		report << "\nREPAIR STEPS\n";
		if (result.steps.empty()) report << "  none\n";
		else for (const auto& step : result.steps) report << "  " << step << "\n";
		if (!result.failures.empty()) {
			report << "\nREPAIR FAILURES\n";
			for (const auto& failure : result.failures) report << "  " << failure << "\n";
		}
		report << "\nAFTER\n";
		reportInspection(report, result.after);
		report << "\nRESULT\n  " << repairStatusName(result.status) << "\n";
		report << "repairMask=" << result.repairMask << "\n\n";
	}

	void preserveCompatibleObjectFeatures(const poca::geometry::ObjectListMesh* source, poca::geometry::ObjectListMesh* destination)
	{
		for (const auto& feature : source->getNameData()) {
			if (feature == "sourceMeshIndex" || feature == "repairStatus" || feature == "repairMask" || destination->hasData(feature)) continue;
			auto* histogram = dynamic_cast<poca::core::Histogram<float>*>(source->getOriginalHistogram(feature));
			if (!histogram) continue;
			const auto& values = histogram->getValues();
			if (values.size() != source->nbObjects()) continue;
			destination->addFeature(feature, poca::core::generateDataWithLogNoInteraction(values));
		}
	}

	void repairObjectListMeshes(poca::geometry::ObjectListMesh* objectList, poca::core::MyObjectInterface* owner, const poca::core::CommandInfo& command)
	{
		if (!objectList) {
			QMessageBox::warning(nullptr, "Object mesh repair", "The current ObjectList is not an ObjectListMesh.");
			return;
		}

		const auto& sources = objectList->getMeshes();
		std::vector<Mesh> outputMeshes;
		std::vector<float> sourceMeshIndices, repairStatuses, repairMasks;
		outputMeshes.reserve(sources.size());
		sourceMeshIndices.reserve(sources.size());
		repairStatuses.reserve(sources.size());
		repairMasks.reserve(sources.size());

		std::ostringstream report;
		report << std::setprecision(17);
		report << "Mesh repair\n";
		report << "Every source mesh is inspected. All repairs operate on copies; rejected meshes retain their original geometry.\n\n";
		size_t clean = 0, requiringRepair = 0, repairedCount = 0, rejected = 0;
		size_t detectedSelfIntersections = 0, detectedDegenerates = 0, detectedBoundaries = 0;
		size_t detectedComponents = 0, detectedTriangulation = 0, detectedOrientation = 0, detectedNonManifold = 0;
		size_t repairedSelfIntersections = 0, repairedDegenerates = 0, repairedBoundaries = 0;
		size_t repairedComponents = 0, repairedTriangulation = 0, repairedOrientation = 0, repairedNonManifold = 0;

		for (size_t meshIndex = 0; meshIndex < sources.size(); ++meshIndex) {
			report << "--------------------------------------------------\n";
			report << "Mesh " << meshIndex << "\n\n";
			poca::geometry::MeshRepairResult result;
			try {
				result = poca::geometry::MeshRepair::repair(sources[meshIndex]);
			}
			catch (const std::bad_alloc&) {
				throw;
			}
			catch (const std::exception& exception) {
				result.status = poca::geometry::MeshRepairStatus::Rejected;
				result.failures.push_back(std::string("per-mesh repair exception: ") + exception.what());
			}
			catch (...) {
				result.status = poca::geometry::MeshRepairStatus::Rejected;
				result.failures.emplace_back("unknown per-mesh repair exception");
			}

			if (result.status == poca::geometry::MeshRepairStatus::Clean) {
				++clean;
				outputMeshes.push_back(sources[meshIndex]);
				report << "Mesh " << meshIndex << ": CLEAN\n\n";
			}
			else if (result.status == poca::geometry::MeshRepairStatus::Repaired) {
				++requiringRepair;
				++repairedCount;
				outputMeshes.push_back(std::move(result.repairedMesh));
				reportRepairDetails(report, result);
			}
			else {
				++requiringRepair;
				++rejected;
				outputMeshes.push_back(sources[meshIndex]);
				reportRepairDetails(report, result);
			}

			sourceMeshIndices.push_back(static_cast<float>(meshIndex));
			repairStatuses.push_back(result.status == poca::geometry::MeshRepairStatus::Clean ? 0.f
				: result.status == poca::geometry::MeshRepairStatus::Repaired ? 1.f : 2.f);
			if (result.before.issueMask & poca::geometry::MeshIssueSelfIntersection) ++detectedSelfIntersections;
			if (result.before.issueMask & poca::geometry::MeshIssueDegenerate) ++detectedDegenerates;
			if (result.before.issueMask & poca::geometry::MeshIssueBoundary) ++detectedBoundaries;
			if (result.before.issueMask & poca::geometry::MeshIssueDisconnected) ++detectedComponents;
			if (result.before.issueMask & poca::geometry::MeshIssueNonTriangle) ++detectedTriangulation;
			if (result.before.issueMask & poca::geometry::MeshIssueOrientation) ++detectedOrientation;
			if (result.before.issueMask & poca::geometry::MeshIssueNonManifold) ++detectedNonManifold;
			const std::uint32_t acceptedRepairMask = result.status == poca::geometry::MeshRepairStatus::Repaired ? result.repairMask : poca::geometry::MeshRepairNone;
			repairMasks.push_back(static_cast<float>(acceptedRepairMask));
			if (acceptedRepairMask & poca::geometry::MeshRepairSelfIntersection) ++repairedSelfIntersections;
			if (acceptedRepairMask & poca::geometry::MeshRepairDegenerateFaces) ++repairedDegenerates;
			if (acceptedRepairMask & poca::geometry::MeshRepairBoundary) ++repairedBoundaries;
			if (acceptedRepairMask & poca::geometry::MeshRepairDisconnectedComponents) ++repairedComponents;
			if (acceptedRepairMask & poca::geometry::MeshRepairTriangulatedFaces) ++repairedTriangulation;
			if (acceptedRepairMask & poca::geometry::MeshRepairOrientation) ++repairedOrientation;
			if (acceptedRepairMask & poca::geometry::MeshRepairNonManifoldTopology) ++repairedNonManifold;
		}

		report << "==================================================\n";
		report << "Mesh repair summary\n\n";
		report << "total meshes: " << sources.size() << "\n";
		report << "clean meshes: " << clean << "\n";
		report << "meshes requiring repair: " << requiringRepair << "\n";
		report << "successfully repaired: " << repairedCount << "\n";
		report << "repair rejected: " << rejected << "\n";
		report << "detected - self-intersection: " << detectedSelfIntersections << "\n";
		report << "detected - degenerates: " << detectedDegenerates << "\n";
		report << "detected - disconnected components: " << detectedComponents << "\n";
		report << "detected - borders/holes: " << detectedBoundaries << "\n";
		report << "detected - triangulation required: " << detectedTriangulation << "\n";
		report << "detected - orientation: " << detectedOrientation << "\n";
		report << "detected - non-manifold: " << detectedNonManifold << "\n";
		report << "repairs applied - self-intersection: " << repairedSelfIntersections << "\n";
		report << "repairs applied - degenerates: " << repairedDegenerates << "\n";
		report << "repairs applied - disconnected components: " << repairedComponents << "\n";
		report << "repairs applied - borders/holes: " << repairedBoundaries << "\n";
		report << "repairs applied - triangulation: " << repairedTriangulation << "\n";
		report << "repairs applied - orientation: " << repairedOrientation << "\n";
		report << "repairs applied - non-manifold: " << repairedNonManifold << "\n";
		report << "repairMask bits: 1=selfIntersection, 2=degenerateFaces, 4=boundaryOrHole, 8=disconnectedComponents, 16=triangulatedFaces, 32=orientation, 64=nonManifoldTopology\n";

		if (outputMeshes.size() == sources.size() && owner) {
			auto* lists = dynamic_cast<poca::geometry::ObjectLists*>(owner->getBasicComponent("ObjectLists"));
			if (lists) {
				try {
					std::unique_ptr<poca::geometry::ObjectListMesh> repaired(new poca::geometry::ObjectListMesh(outputMeshes, false, 0.f, 0, false));
					if (repaired->nbObjects() != sources.size()) {
						report << "\nNo repaired ObjectListMesh created: ObjectListMesh construction did not preserve the exact source mesh count.\n";
					}
					else {
						repaired->setUseVertexNormals(objectList->useVertexNormals());
						repaired->setSelection(objectList->getSelection());
						preserveCompatibleObjectFeatures(objectList, repaired.get());
						repaired->addFeature("sourceMeshIndex", poca::core::generateDataWithLogNoInteraction(sourceMeshIndices));
						repaired->addFeature("repairStatus", poca::core::generateDataWithLogNoInteraction(repairStatuses));
						repaired->addFeature("repairMask", poca::core::generateDataWithLogNoInteraction(repairMasks));
						repaired->setCurrentHistogramType("repairStatus");
						ObjectListPlugin::m_plugins->addCommands(repaired.get());
						lists->addObjectList(repaired.get(), command, "ObjectListPlugin", "Mesh repaired");
						repaired.release();
						report << "\nCreated ObjectListMesh:\n  Mesh repaired\n";
						report << "Output mesh count/order exactly matches the source. Features: sourceMeshIndex, repairStatus (0 clean, 1 repaired, 2 rejected), repairMask.\n";
					}
				}
				catch (const std::bad_alloc&) {
					throw;
				}
				catch (const std::exception& exception) {
					report << "\nNo repaired ObjectListMesh created: output construction failed: " << exception.what() << "\n";
				}
			}
			else report << "\nNo repaired ObjectListMesh created: the owning ObjectLists component is unavailable.\n";
		}
		else report << "\nNo repaired ObjectListMesh created: output/source count mismatch or owning object unavailable.\n";

		const std::string text = report.str();
		std::cout << text << std::endl;
		showMeshReport(text, "Object mesh repair report", "Conservative copy-only CGAL repair for the current ObjectListMesh");
	}
}

ObjectListBasicCommands::ObjectListBasicCommands(poca::geometry::ObjectListInterface* _objs) :poca::core::Command("ObjectListBasicCommands")
{
	m_objects = _objs;
}

ObjectListBasicCommands::ObjectListBasicCommands(const ObjectListBasicCommands& _o) : poca::core::Command(_o)
{
	m_objects = _o.m_objects;
}

ObjectListBasicCommands::~ObjectListBasicCommands()
{
}

std::vector<poca::core::CommandSpec> ObjectListBasicCommands::commandSpecs() const
{
	using poca::core::CommandParameterType;
	using poca::core::CommandSpec;

	return {
		CommandSpec("saveStatsObjs", {
			{"filename", CommandParameterType::String, false, nullptr},
			{"appendToTitle", CommandParameterType::String, false, nullptr},
			{"separator", CommandParameterType::String, false, std::string(",")}
		}),
		CommandSpec("saveLocsObjs", {
			{"filename", CommandParameterType::String, false, nullptr},
			{"appendToTitle", CommandParameterType::String, false, nullptr},
			{"separator", CommandParameterType::String, false, std::string(",")}
		}),
		CommandSpec("saveOutlineLocsObjs", {
			{"filename", CommandParameterType::String, false, nullptr},
			{"appendToTitle", CommandParameterType::String, false, nullptr},
			{"separator", CommandParameterType::String, false, std::string(",")}
		}),
		CommandSpec("saveAsSVG", {
			{"filename", CommandParameterType::String, false, nullptr},
			{"appendToTitle", CommandParameterType::String, false, nullptr},
			{"separator", CommandParameterType::String, false, std::string(",")}
		}),
		CommandSpec("saveAsOBJ", {
			{"filename", CommandParameterType::String, false, nullptr},
			{"appendToTitle", CommandParameterType::String, false, nullptr},
			{"appendToDir", CommandParameterType::String, false, nullptr},
			{"appendToName", CommandParameterType::String, false, nullptr}
		}),
		CommandSpec("duplicateCentroids"),
		CommandSpec("computeSkeletons"),
		CommandSpec("testMeshes"),
		CommandSpec("repairMeshes"),
		CommandSpec("repairSelfIntersections"),
		CommandSpec("exportObjectsInROIs"),
		CommandSpec("exportLocsInObjects"),
		CommandSpec("duplicateSelectedObjects", {
			{"selection", CommandParameterType::Array, true, nullptr}
		}),
		CommandSpec("saveSelectedObjectsVectorHeat", {
			{"selection", CommandParameterType::Array, true, nullptr}
		}),
		CommandSpec("fillHolesObjects", {
			{"minArea", CommandParameterType::Number, true, nullptr}
		}),
		CommandSpec("smoothObjects", {
			{"factorResampling", CommandParameterType::Number, true, nullptr},
			{"nbSmoothSteps", CommandParameterType::UnsignedInteger, true, nullptr},
			{"windowSize", CommandParameterType::UnsignedInteger, true, nullptr}
		}),
		CommandSpec("subdivide", {
			{"iterations", CommandParameterType::UnsignedInteger, true, nullptr}
		})
	};
}

void ObjectListBasicCommands::execute(poca::core::CommandInfo* _infos)
{
	poca::core::CommandExecutionContext context;
	poca::core::CommandExecutionResult result;
	execute(_infos, context, result);
}

void ObjectListBasicCommands::execute(poca::core::CommandInfo* _infos, const poca::core::CommandExecutionContext& _context, poca::core::CommandExecutionResult& _result)
{
	if (_infos->nameCommand == "saveStatsObjs") {
		std::string filename, separator(",");
		if (_infos->hasParameter("filename"))
			filename = _infos->getParameter<std::string>("filename");
		if (_infos->hasParameter("separator"))
			separator = _infos->getParameter<std::string>("separator");
		saveStatsObj(filename, separator);
	}
	else if (_infos->nameCommand == "saveLocsObjs") {
		std::string filename, separator(",");
		if (_infos->hasParameter("filename"))
			filename = _infos->getParameter<std::string>("filename");
		if (_infos->hasParameter("separator"))
			separator = _infos->getParameter<std::string>("separator");
		saveLocsObj(filename, separator);
	}
	else if (_infos->nameCommand == "saveOutlineLocsObjs") {
		std::string filename, separator(",");
		if (!_infos->hasParameter("filename")) return;
		filename = _infos->getParameter<std::string>("filename");
		if (_infos->hasParameter("separator"))
			separator = _infos->getParameter<std::string>("separator");
		saveOutlineLocsObj(filename, separator);
	}
	else if (_infos->nameCommand == "duplicateCentroids") {
		poca::core::MyObjectInterface* obj = duplicateCentroids();
		if(obj != NULL)
			_result.set<poca::core::CreatedObjectContext>({ obj });
	}
	else if (_infos->nameCommand == "testMeshes") {
		poca::geometry::ObjectListMesh* omesh = dynamic_cast<poca::geometry::ObjectListMesh*>(m_objects);
		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* owner = engine->getObject(m_objects);
		validateObjectListMeshes(omesh, owner, *_infos);
	}
	else if (_infos->nameCommand == "repairMeshes" || _infos->nameCommand == "repairSelfIntersections") {
		poca::geometry::ObjectListMesh* omesh = dynamic_cast<poca::geometry::ObjectListMesh*>(m_objects);
		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* owner = engine->getObject(m_objects);
		repairObjectListMeshes(omesh, owner, *_infos);
	}
	else if (_infos->nameCommand == "duplicateSelectedObjects") {
		std::set <int> selectedObjects = _infos->hasParameter("selection")? _infos->getParameter<std::set <int>>("selection") : std::set<int>();
		if (selectedObjects.empty()) {
			_infos->errorMessage("the selection of objects is empty.");
			return;
		}
		poca::core::MyObjectInterface* obj = duplicateSelectedObjects(selectedObjects);
		if (obj == NULL) {
			_infos->errorMessage("selected objects were not duplicated.");
			return;
		}
		_result.set<poca::core::CreatedObjectContext>({ obj });
	}
	else if (_infos->nameCommand == "saveSelectedObjectsForVectorHeat") {
		std::set <int> selectedObjects = _infos->hasParameter("selection") ? _infos->getParameter<std::set <int>>("selection") : std::set<int>();
		if (selectedObjects.empty()) {
			_infos->errorMessage("the selection of objects is empty.");
			return;
		}
		saveSelectedObjectsForVectorHeat(selectedObjects);
	}
	else if (_infos->nameCommand == "saveAsSVG") {
		QString filename = (_infos->getParameter<std::string>("filename")).c_str();
		saveAsSVG(filename);
	}
	else if (_infos->nameCommand == "saveAsOBJ") {
		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_objects);
		QString filename;
		if(_infos->hasParameter("filename"))
			filename = (_infos->getParameter<std::string>("filename")).c_str();
		else {
			/*poca::core::Engine* engine = poca::core::Engine::instance();
			poca::core::MyObjectInterface* obj = engine->getObject(m_objects);

			QString textToAdd = "_objects.obj";
			if (_infos->hasParameter("appendToTitle"))
				textToAdd = QString(_infos->getParameter<std::string>("appendToTitle").c_str()).append(".obj");

			const std::string& dir = obj->getDir(), name = obj->getName();
			QString completeName = dir.c_str();
			if (!completeName.endsWith('/'))
				completeName.append("/");
			completeName.append(name.c_str());
			std::cout << "Name: " << completeName.toStdString() << std::endl;
			QFileInfo fileInfo(completeName);
			filename = fileInfo.path() + "/" + fileInfo.completeBaseName() + textToAdd;
			std::cout << "Name: " << filename.toStdString() << std::endl;*/

			poca::core::Engine* engine = poca::core::Engine::instance();
			poca::core::MyObjectInterface* obj = engine->getObject(m_objects);
			QString dir = obj->getDir().c_str();
			QString name = obj->getName().c_str();
			if (!dir.endsWith('/'))
				dir.append("/");
			if (_infos->hasParameter("appendToDir")) {
				std::string addToDir = _infos->getParameter<std::string>("appendToDir");
				dir = dir + addToDir.c_str();
				if (!dir.endsWith('/'))
					dir.append("/");
			}
			if (_infos->hasParameter("appendToName")) {
				std::string addToFile = _infos->getParameter<std::string>("appendToName");
				int dotIndex = name.lastIndexOf('.');

				if (dotIndex != -1)
					name.insert(dotIndex, addToFile.c_str());
				else
					name.append(addToFile.c_str());
			}
			filename = dir + name;
		}
		if (m_objects->dimension() == 3)
			if (!filename.endsWith('.obj'))
				filename.append(".obj");
		if (m_objects->dimension() == 2)
			if (!filename.endsWith('.pol'))
				filename.append(".pol");
		saveAsOBJ(filename);
	}
	else if (_infos->nameCommand == "computeSkeletons") {
		poca::geometry::ObjectListMesh* omesh = dynamic_cast <poca::geometry::ObjectListMesh*>(m_objects);
		if (omesh == NULL) return;
		omesh->computeSkeletons();
	}
	else if (_infos->nameCommand == "exportFilteredObjects") {
		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_objects);

		poca::geometry::ObjectListInterface* newObjects = m_objects->exportFilteredObjects();
		if (!newObjects) return;
		ObjectListPlugin::m_plugins->addCommands(newObjects);
		poca::geometry::ObjectLists* objsList = dynamic_cast<poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
		if (objsList)
			objsList->addObjectList(newObjects, *_infos, "ObjectListPlugin");
	}
	else if (_infos->nameCommand == "exportObjectsInROIs") {
		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_objects);

		poca::geometry::ObjectListMesh* omesh = dynamic_cast <poca::geometry::ObjectListMesh*>(m_objects);
		if (!omesh) return;
		poca::core::BasicComponentInterface* newObjects = omesh->copy(obj->getROIs());
		if (!newObjects) return;
		ObjectListPlugin::m_plugins->addCommands(newObjects);
		poca::geometry::ObjectLists* objsList = dynamic_cast<poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
		if (objsList)
			objsList->addObjectList(static_cast<poca::geometry::ObjectListInterface*>(newObjects), *_infos, "ObjectListPlugin");
	}
	else if (_infos->nameCommand == "exportHolesObjects") {
		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_objects);

		poca::geometry::ObjectListPolygon* opol = static_cast <poca::geometry::ObjectListPolygon*>(m_objects);
		if (!opol) return;
		const std::vector <std::vector<Polygon_2>>& polygons = opol->getPolygons();
		std::vector <std::vector<Polygon_2>> holes;
		for (const auto& polygonWithHoles : polygons) {
			for (auto n = 1; n < polygonWithHoles.size(); n++) {
				holes.push_back(std::vector <Polygon_2>({ polygonWithHoles[n] }));
			}
		}
		poca::geometry::ObjectListPolygon* holesObject = new poca::geometry::ObjectListPolygon(holes);
		ObjectListPlugin::m_plugins->addCommands(holesObject);
		if (!obj->hasBasicComponent("ObjectLists")) {
			poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(holesObject, *_infos, "ObjectListPlugin");
			ObjectListPlugin::m_plugins->addCommands(objsList);
			obj->addBasicComponent(objsList);
		}
		else {
			std::string text = _infos->json.dump(4);
			poca::geometry::ObjectLists* objsList = dynamic_cast<poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
			if (objsList)
				objsList->addObjectList(holesObject, *_infos, "ObjectListPlugin");
			std::cout << text << std::endl;
		}
	}
	else if (_infos->nameCommand == "fillHolesObjects") {
		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_objects);
		float minArea = 0.f;
		if (_infos->hasParameter("minArea"))
			minArea = _infos->getParameter<float>("minArea");

		poca::geometry::ObjectListPolygon* opol = static_cast <poca::geometry::ObjectListPolygon*>(m_objects);
		if (!opol) return;
		const std::vector <std::vector<Polygon_2>>& polygons = opol->getPolygons();
		std::vector <std::vector<Polygon_2>> newObjects;
		for (const auto& polygonWithHoles : polygons) {
			newObjects.push_back(std::vector <Polygon_2>());
			auto& object = newObjects.back();
			object.emplace_back(polygonWithHoles.front());
			for (auto n = 1; n < polygonWithHoles.size(); n++) {
				if(fabs(polygonWithHoles[n].area()) > minArea)
					object.emplace_back(polygonWithHoles[n]);
			}
		}
		poca::geometry::ObjectListPolygon* holesObject = new poca::geometry::ObjectListPolygon(newObjects);
		ObjectListPlugin::m_plugins->addCommands(holesObject);
		if (!obj->hasBasicComponent("ObjectLists")) {
			poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(holesObject, *_infos, "ObjectListPlugin");
			ObjectListPlugin::m_plugins->addCommands(objsList);
			obj->addBasicComponent(objsList);
		}
		else {
			std::string text = _infos->json.dump(4);
			poca::geometry::ObjectLists* objsList = dynamic_cast<poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
			if (objsList)
				objsList->addObjectList(holesObject, *_infos, "ObjectListPlugin");
			std::cout << text << std::endl;
		}
	}
	else if (_infos->nameCommand == "smoothObjects") {
		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_objects);
		float factor = 2;
		uint32_t nbSmooth = 3, windowSize = 3;
		if (_infos->hasParameter("factorResampling"))
			factor = _infos->getParameter<float>("factorResampling");
		if (_infos->hasParameter("nbSmoothSteps"))
			nbSmooth = _infos->getParameter<uint32_t>("nbSmoothSteps");
		if (_infos->hasParameter("windowSize"))
			windowSize = _infos->getParameter<uint32_t>("windowSize");

		std::vector <int> nindices;
		int half = (int)floor(windowSize / 2);
		for (auto n = -half; n <= half; n++)
			nindices.emplace_back(n);

		poca::geometry::ObjectListPolygon* opol = static_cast <poca::geometry::ObjectListPolygon*>(m_objects);
		if (!opol) return;
		const std::vector <std::vector<Polygon_2>>& polygons = opol->getPolygons();
		std::vector <std::vector<Polygon_2>> newObjects;
		std::vector <std::vector<std::vector <float>>> allCurvatures;
		for (const auto& polygonWithHoles : polygons) {
			std::cout << "--------------------------------------------" << std::endl;
			std::cout << __LINE__ << " -> " << polygonWithHoles.size() << std::endl;
			newObjects.push_back(std::vector <Polygon_2>());
			allCurvatures.push_back(std::vector<std::vector <float>>());
			auto& newObject = newObjects.back();
			auto& newCurvatures = allCurvatures.back();

			for (const auto& polygon : polygonWithHoles) {
				std::cout << __LINE__ << " -> " << polygon.size() << std::endl;
				std::vector <poca::core::Vec3mf> originalOutline, smoothedOutline;
				for (const auto& p : polygon.container())
					originalOutline.emplace_back(p.x(), p.y(), 0.f);
				poca::geometry::smoothOutline(originalOutline, smoothedOutline, nbSmooth, windowSize, factor);
				if (!smoothedOutline.empty()) {
					std::vector < Point_2 > finalPoints;
					for (auto n = 0; n < smoothedOutline.size(); n++) {
						finalPoints.emplace_back(smoothedOutline[n].x(), smoothedOutline[n].y());
					}
					std::cout << __LINE__ << std::endl;
					newObject.emplace_back(finalPoints.begin(), finalPoints.begin() + finalPoints.size());
					std::cout << __LINE__ << std::endl;
				}
				else
					std::cout << "this is empty" << std::endl;
				/*std::vector <poca::core::Vec3mf> smoothedOutline, outlineTmp;
				std::cout << __LINE__ << std::endl;
				for (const auto& p : polygon.container())
					outlineTmp.emplace_back(p.x(), p.y(), 0.f);

				for(auto n = 0; n < outlineTmp.size(); n++){
					auto next = (n + 1) % outlineTmp.size();
					const auto& p = outlineTmp[n], pn = outlineTmp[next];
					float d = poca::geometry::distance(pn.x(), pn.y(), p.x(), p.y());
					if (d < TS_POINT_EPSILON)
						std::cout << "Very close" << std::endl;
				}

				smoothedOutline.resize(outlineTmp.size());
				for (auto step = 0; step < nbSmooth; step++) {
					for (auto n = 0; n < outlineTmp.size(); n++) {
						float x = 0, y = 0;
						for (auto nind : nindices) {
							auto id = n + nind;
							if (id < 0)
								id = outlineTmp.size() + id;
							else if (id >= outlineTmp.size())
								id = id % outlineTmp.size();
							x += outlineTmp[id].x() / (float)windowSize;
							y += outlineTmp[id].y() / (float)windowSize;
						}
						smoothedOutline[n].set(x, y, 0.f);
					}
					outlineTmp = smoothedOutline;
				}
				std::cout << __LINE__ << std::endl;

				for (auto n = 0; n < smoothedOutline.size(); n++) {
					auto next = (n + 1) % smoothedOutline.size();
					const auto& p = smoothedOutline[n], pn = smoothedOutline[next];
					float d = poca::geometry::distance(pn.x(), pn.y(), p.x(), p.y());
					if (d < TS_POINT_EPSILON)
						std::cout << "Very close" << std::endl;
				}

				std::vector<tinyspline::real> points;
				for (const auto& pt : smoothedOutline) {
					points.push_back(pt.x());
					points.push_back(pt.y());
				}

				std::cout << __LINE__ << " " << (points.size() / 2) << std::endl;

				if (points.size() > 10 * 2) {
					try {
						newCurvatures.push_back(std::vector <float>());
						auto& curvatures = newCurvatures.back();
						
						std::cout << __LINE__ << std::endl;
						tinyspline::BSpline spline = tinyspline::BSpline(points.size() / 2);
						std::cout << __LINE__ << std::endl;
						spline.setControlPoints(points);
						std::cout << __LINE__ << std::endl;
						std::vector<tinyspline::real> knotsAct = spline.chordLengths(points.size()).equidistantKnotSeq(( points.size() / 2) * factor);
						std::vector <poca::core::Vec3mf> vtmp;
						std::cout << __LINE__ << std::endl;
						for (auto n = 0; n < knotsAct.size(); n++) {
							try {
								auto pt = spline.eval(knotsAct[n]).resultVec2();
								bool addPoint = true;
								if (!vtmp.empty()) {
									const auto& p = vtmp.back();
									float d = poca::geometry::distance((float)pt.x(), (float)pt.y(), p.x(), p.y());
									if (d < TS_POINT_EPSILON)
										addPoint = false;
								}
								if (addPoint) {
									vtmp.push_back(poca::core::Vec3mf(pt.x(), pt.y(), 0.f));
								}
							}
							catch (const std::runtime_error& e) {
								std::cerr << "Caught runtime_error: " << e.what() << " for n = " << n << " and knotsAct " << knotsAct[n] << std::endl;
							}
						}
						std::cout << __LINE__ << std::endl;
						//smooth curve
						std::vector < Point_2 > finalPoints;
						for (auto n = 0; n < vtmp.size(); n++) {
							finalPoints.emplace_back(vtmp[n].x(), vtmp[n].y());
						}
						std::cout << __LINE__ << std::endl;
						newObject.emplace_back(finalPoints.begin(), finalPoints.begin() + finalPoints.size());
						std::cout << __LINE__ << std::endl;
					}
					catch (const std::runtime_error& e) {
						std::cerr << "Caught runtime_error: " << e.what() << std::endl;
					}
				}*/
			}
		}
		poca::geometry::ObjectListPolygon* objects = new poca::geometry::ObjectListPolygon(newObjects);
		objects->setCurvatures(allCurvatures);
		ObjectListPlugin::m_plugins->addCommands(objects);
		if (!obj->hasBasicComponent("ObjectLists")) {
			poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(objects, *_infos, "ObjectListPlugin");
			ObjectListPlugin::m_plugins->addCommands(objsList);
			obj->addBasicComponent(objsList);
		}
		else {
			std::string text = _infos->json.dump(4);
			poca::geometry::ObjectLists* objsList = dynamic_cast<poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
			if (objsList)
				objsList->addObjectList(objects, *_infos, "ObjectListPlugin");
			std::cout << text << std::endl;
		}
	}
	else if (_infos->nameCommand == "remesh") {
		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_objects);

		float targetLength = _infos->getParameter<float>("targetLength");
		uint32_t iterations = _infos->getParameter<uint32_t>("iterations");

		poca::geometry::ObjectListMesh* omesh = static_cast <poca::geometry::ObjectListMesh*>(m_objects);
		if (!omesh) return;
		std::vector <Surface_mesh_3_double> meshes = omesh->getMeshes();
		for (auto& mesh : meshes)
			CGAL::Polygon_mesh_processing::isotropic_remeshing(faces(mesh), targetLength, mesh, CGAL::parameters::number_of_iterations(iterations));
		poca::geometry::ObjectListMesh* newomesh = new poca::geometry::ObjectListMesh(meshes);
		//poca::geometry::ObjectListMesh* copymesh = new poca::geometry::ObjectListMesh(omesh->getMeshes(), true, targetLength, iterations);
		//copymesh->remesh(targetLength, iterations);
		ObjectListPlugin::m_plugins->addCommands(newomesh);
		if (!obj->hasBasicComponent("ObjectLists")) {
			poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(newomesh, *_infos, "ObjectListPlugin");
			ObjectListPlugin::m_plugins->addCommands(objsList);
			obj->addBasicComponent(objsList);
		}
		else {
			std::string text = _infos->json.dump(4);
			poca::geometry::ObjectLists* objsList = dynamic_cast<poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
			if (objsList)
				objsList->addObjectList(newomesh, *_infos, "ObjectListPlugin");
			std::cout << text << std::endl;
		}
	}
	else if (_infos->nameCommand == "subdivide") {
		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_objects);

		uint32_t iterations = _infos->getParameter<uint32_t>("iterations");

		poca::geometry::ObjectListMesh* omesh = static_cast <poca::geometry::ObjectListMesh*>(m_objects);
		if (!omesh) return;
		std::vector <Surface_mesh_3_double> meshes = omesh->getMeshes();
		for (auto& mesh : meshes) {
			//CGAL::Subdivision_method_3::Sqrt3_subdivision(mesh, CGAL::parameters::number_of_iterations(iterations));
			CGAL::Polygon_mesh_processing::orient(mesh);
			CGAL::Polygon_mesh_processing::stitch_borders(mesh);
			CGAL::Polygon_mesh_processing::remove_isolated_vertices(mesh);
			CGAL::Polygon_mesh_processing::remove_degenerate_faces(mesh);
			CGAL::Polygon_mesh_processing::triangulate_faces(mesh);

			// If still open, prefer Loop; if closed, Sqrt3:
			bool closed = true; for (auto e : edges(mesh)) if (is_border(e, mesh)) { closed = false; break; }
			if (engine->verbose())
				std::cout << "is closed " << closed << std::endl;
			if (closed)
				CGAL::Subdivision_method_3::Sqrt3_subdivision(mesh, CGAL::parameters::number_of_iterations(iterations));
			else
				CGAL::Subdivision_method_3::Loop_subdivision(mesh, CGAL::parameters::number_of_iterations(iterations));
		}
		poca::geometry::ObjectListMesh* newomesh = new poca::geometry::ObjectListMesh(meshes);
		//copymesh->subdivide(iterations);
		ObjectListPlugin::m_plugins->addCommands(newomesh);
		if (obj) {
			if (!obj->hasBasicComponent("ObjectLists")) {
				poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(newomesh, *_infos, "ObjectListPlugin");
				ObjectListPlugin::m_plugins->addCommands(objsList);
				obj->addBasicComponent(objsList);
			}
			else {
				std::string text = _infos->json.dump(4);
				poca::geometry::ObjectLists* objsList = dynamic_cast<poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
				if (objsList)
					objsList->addObjectList(newomesh, *_infos, "ObjectListPlugin");
				std::cout << text << std::endl;
			}
		}
		else
			_result.set<poca::geometry::CreatedObjectListMeshContext>({ newomesh });
	}
	else if (_infos->nameCommand == "laplacianSmooth") {
		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_objects);

		float lambda = _infos->getParameter<float>("lambda");
		uint32_t iterations = _infos->getParameter<uint32_t>("iterations");

		poca::geometry::ObjectListMesh* omesh = static_cast <poca::geometry::ObjectListMesh*>(m_objects);
		if (!omesh) return;
		std::vector <Surface_mesh_3_double> meshes = omesh->getMeshes();
		for (auto& mesh : meshes)
			poca::geometry::laplacian_smooth(mesh, iterations, lambda);
		poca::geometry::ObjectListMesh* newomesh = new poca::geometry::ObjectListMesh(meshes);
		ObjectListPlugin::m_plugins->addCommands(newomesh);
		if (!obj) {
			_result.set<poca::geometry::CreatedObjectListMeshContext>({ newomesh });
		}
		else if (!obj->hasBasicComponent("ObjectLists")) {
			poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(newomesh, *_infos, "ObjectListPlugin");
			ObjectListPlugin::m_plugins->addCommands(objsList);
			obj->addBasicComponent(objsList);
		}
		else {
			std::string text = _infos->json.dump(4);
			poca::geometry::ObjectLists* objsList = dynamic_cast<poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
			if (objsList)
				objsList->addObjectList(newomesh, *_infos, "ObjectListPlugin");
			std::cout << text << std::endl;
		}
	}
	else if (_infos->nameCommand == "exportLocsInObjects") {
		poca::geometry::ObjectListPolygon* opol = static_cast <poca::geometry::ObjectListPolygon*>(m_objects);
		if (!opol) return;
		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_objects);
		if (!obj->hasBasicComponent("DetectionSet"))
			return;
		poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(obj->getBasicComponent("DetectionSet"));
		if (!dset)
			return;

		std::vector <uint32_t> selectedLocs;
		const std::vector <float>& xs = dset->getData<float>("x");
		const std::vector <float>& ys = dset->getData<float>("y");
		for (auto n = 0; n < xs.size(); n++) {
			for (const auto& polygons : opol->getPolygons()) {
				bool inside = polygons[0].bounded_side(Point_2(xs[n], ys[n])) == CGAL::ON_BOUNDED_SIDE, inside_hole = false;
				if (inside) {
					for (auto cur = 1; cur < polygons.size(); cur++) {
						inside_hole |= polygons[cur].bounded_side(Point_2(xs[n], ys[n])) == CGAL::ON_BOUNDED_SIDE;
					}
				}
				if (inside && !inside_hole) {
					selectedLocs.push_back(n);
				}
			}
		}

		poca::geometry::DetectionSet* newdset = dset->copySelection(selectedLocs);
		engine->addComponentToObject(obj, newdset);
	}
}


poca::core::Command* ObjectListBasicCommands::copy()
{
	return new ObjectListBasicCommands(*this);
}

void ObjectListBasicCommands::saveStatsObj(const std::string& _filename, const std::string& _separator) const
{
	const std::map <std::string, poca::core::MyData*>& data = m_objects->getData();

	QString filename(_filename.c_str());

	if (_filename.empty()) {
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_objects);
		if (obj == NULL) return;

		std::string dir = obj->getDir(), name = obj->getName();
		filename = QString(dir.c_str());
		if (!filename.endsWith('/'))
			filename.append('/');
		filename.append(name.c_str());
		int index = filename.lastIndexOf(".");
		filename.insert(index, "_statsObjs");
	}

	QFileInfo info(filename);
	std::ofstream fs(info.absoluteFilePath().toStdString());

	if (!fs.is_open()) {
		std::cout << "Failed to open file " << _filename << std::endl;
		return;
	}

	std::vector<std::string> names;
	fs << "id" << _separator;
	for (auto& [name, values] : data) {
		fs << name << _separator;
		names.push_back(name);
	}
	fs << std::endl;
	for (size_t n = 0; n < m_objects->nbObjects(); n++) {
		fs << (n + 1) << _separator;
		for (const std::string& name : names)
			fs << m_objects->getMyData(name)->getData<float>()[n] << _separator;
		fs << std::endl;
	}
	fs.close();
	std::cout << "File " << filename.toStdString() << " was written" << std::endl;
}

void ObjectListBasicCommands::saveLocsObj(const std::string& _filename, const std::string& _separator) const
{
	poca::core::MyObjectInterface* obj = poca::core::Engine::instance()->getObject(m_objects);
	poca::core::MyObjectInterface* oneColorObj = obj->currentObject();
	poca::core::BasicComponentInterface* bci = oneColorObj->getBasicComponent("DetectionSet");
	if (bci == NULL) {
		std::cout << "PoCA did not succeed in save the localizations <-> objects link" << std::endl;
		return;
	}

	QString filename(_filename.c_str());

	if (_filename.empty()) {
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_objects);
		if (obj == NULL) return;

		std::string dir = obj->getDir(), name = obj->getName();
		filename = QString(dir.c_str());
		if (!filename.endsWith('/'))
			filename.append('/');
		filename.append(name.c_str());
		int index = filename.lastIndexOf(".");
		filename.insert(index, "_statsObjs");
	}

	const std::map <std::string, poca::core::MyData*>& datacomp = bci->getData();
	poca::core::stringList columnNames = bci->getNameData();
	QFileInfo info(filename);
	std::ofstream fs(info.absoluteFilePath().toStdString());
	if (!fs.is_open()) {
		std::cout << "Failed to open file " << _filename << std::endl;
		return;
	}

	size_t nbLocs = bci->nbElements();
	std::vector <uint32_t> idx(nbLocs, 0);

	const poca::core::MyArrayUInt32& objs = m_objects->getLocsObjects();
	const std::vector <uint32_t>& data = objs.getData();
	const std::vector <uint32_t>& firsts = objs.getFirstElements();

	unsigned int currentLine = 0, totalNb = data.size(), nbForUpdate = totalNb / 100.;
	if (nbForUpdate == 0) nbForUpdate = 1;
	printf("Computing id loc <-> id obj link: %.2f %%", ((double)currentLine / totalNb * 100.));

	for (size_t n = 0; n < objs.nbElements(); n++)
		for (uint32_t cur = firsts[n]; cur < firsts[n + 1]; cur++) {
			idx[data[cur]] = n + 1;
			if (currentLine++ % nbForUpdate == 0) printf("\rComputing id loc <-> id obj link: %.2f %%", ((double)currentLine / totalNb * 100.));
		}
	printf("\rComputing id loc <-> id obj link: 100 %%\n");

	currentLine = 0; totalNb = nbLocs;
	printf("Saving id loc <-> id obj link: %.2f %%", ((double)currentLine / totalNb * 100.));

	fs << "id object";
	for (auto cur : datacomp) {
		fs << _separator << cur.first;
	}
	fs << std::endl;
	for (size_t n = 0; n < idx.size(); n++) {
		fs << idx[n];
		for (auto cur : datacomp) {
			fs << std::setprecision(8) << _separator << cur.second->getData<float>()[n];
		}
		fs << std::endl;
		if (currentLine++ % nbForUpdate == 0) printf("\rSaving id loc <-> id obj link: %.2f %%", ((double)currentLine / totalNb * 100.));
	}
	printf("\rSaving id loc <-> id obj link: 100 %%\n");

	fs.close();
	std::cout << "File " << filename.toStdString() << " was written" << std::endl;
}

void ObjectListBasicCommands::saveOutlineLocsObj(const std::string& _filename, const std::string& _separator) const
{
	poca::core::MyObjectInterface* obj = poca::core::Engine::instance()->getObject(m_objects);
	poca::core::MyObjectInterface* oneColorObj = obj->currentObject();
	poca::core::BasicComponentInterface* bci = oneColorObj->getBasicComponent("DetectionSet");
	if (bci == NULL) {
		std::cout << "PoCA did not succeed in save the localizations <-> objects link" << std::endl;
		return;
	}

	const std::map <std::string, poca::core::MyData*>& datacomp = bci->getData();
	poca::core::stringList columnNames = bci->getNameData();
	QFileInfo info(_filename.c_str());
	std::ofstream fs(info.absoluteFilePath().toStdString());
	if (!fs.is_open()) {
		std::cout << "Failed to open file " << _filename << std::endl;
		return;
	}

	size_t nbLocs = bci->nbElements();
	std::vector <uint32_t> idx(nbLocs, 0);

	const poca::core::MyArrayUInt32& objs = m_objects->getLocOutlines();
	const std::vector <uint32_t>& data = objs.getData();
	const std::vector <uint32_t>& firsts = objs.getFirstElements();

	unsigned int currentLine = 0, totalNb = data.size(), nbForUpdate = totalNb / 100.;
	if (nbForUpdate == 0) nbForUpdate = 1;
	printf("Computing id loc <-> id obj link: %.2f %%", ((double)currentLine / totalNb * 100.));

	for (size_t n = 0; n < objs.nbElements(); n++)
		for (uint32_t cur = firsts[n]; cur < firsts[n + 1]; cur++) {
			idx[data[cur]] = n + 1;
			if (currentLine++ % nbForUpdate == 0) printf("\rComputing id loc <-> id obj link: %.2f %%", ((double)currentLine / totalNb * 100.));
		}
	printf("\rComputing id loc <-> id obj link: 100 %%\n");

	currentLine = 0; totalNb = nbLocs;
	printf("Saving id loc <-> id obj link: %.2f %%", ((double)currentLine / totalNb * 100.));

	fs << "id object";
	for (auto cur : datacomp) {
		fs << _separator << cur.first;
	}
	fs << std::endl;
	for (size_t n = 0; n < idx.size(); n++) {
		if (idx[n] == 0) continue;
		fs << idx[n];
		for (auto cur : datacomp) {
			fs << std::setprecision(8) << _separator << cur.second->getData<float>()[n];
		}
		fs << std::endl;
		if (currentLine++ % nbForUpdate == 0) printf("\rSaving id loc <-> id obj link: %.2f %%", ((double)currentLine / totalNb * 100.));
	}
	printf("\rSaving id loc <-> id obj link: 100 %%\n");

	fs.close();
	std::cout << "File " << _filename << " was written" << std::endl;
}

poca::core::MyObjectInterface* ObjectListBasicCommands::duplicateCentroids() const
{
	std::map <std::string, std::vector <float>> features;
	std::vector <poca::core::Vec3mf> centroids(m_objects->nbElements());
	for (size_t n = 0; n < m_objects->nbElements(); n++)
		centroids[n] = m_objects->computeBarycenterElement(n);
	std::vector <float> xs(centroids.size()), ys(centroids.size()), zs(centroids.size());
	for (size_t n = 0; n < centroids.size(); n++) {
		xs[n] = centroids[n][0];
		ys[n] = centroids[n][1];
		zs[n] = centroids[n][2];
	}
	features["x"] = xs;
	features["y"] = ys;
	features["z"] = zs;

	std::map <std::string, poca::core::MyData*> featuresObjects = m_objects->getData();
	for (const auto& feature : featuresObjects)
		if(feature.first != "x" && feature.first != "y" && feature.first != "z")
			features[feature.first] = feature.second->getData<float>();

	poca::geometry::DetectionSet* dset = new poca::geometry::DetectionSet(features);

	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::core::MyObjectInterface* obj = engine->getObject(m_objects);
	engine->addComponentToObject(obj, dset);
	/*const std::string& dir = obj->getDir(), name = obj->getName();
	QString newName(name.c_str());
	int index = newName.lastIndexOf(".");
	newName.insert(index, "_objectsCentroids");

	poca::core::MyObject* wobj = new poca::core::MyObject();
	wobj->setDir(dir.c_str());
	wobj->setName(newName.toLatin1().data());
	wobj->addBasicComponent(dset);
	wobj->setDimension(dset->dimension());

	return wobj;*/
	return NULL;
}

poca::core::MyObjectInterface* ObjectListBasicCommands::duplicateSelectedObjects(const std::set<int>& _selectedObjects) const
{
	/*poca::core::MyObjectInterface* obj = poca::core::Engine::instance()->getObject(m_objects);
	poca::core::MyObjectInterface* oneColorObj = obj->currentObject();
	poca::core::BasicComponentInterface* bc = oneColorObj->getBasicComponent("DetectionSet");
	if (bc == NULL) {
		std::cout << "PoCA did not succeed in save the localizations <-> objects link" << std::endl;
		return NULL;
	}

	std::map <std::string, poca::core::MyData*> featuresDset = bc->getData();
	const poca::core::MyArrayUInt32& objs = m_objects->getLocsObjects();
	const std::vector <uint32_t>& data = objs.getData();
	const std::vector <uint32_t>& firsts = objs.getFirstElements();

	std::vector <bool> selectedLocs(bc->nbElements(), false);
	for (auto idx : _selectedObjects)
		for (uint32_t cur = firsts[idx]; cur < firsts[idx + 1]; cur++)
			selectedLocs[data[cur]] = true;

	auto nbSelectedLocs = std::count(selectedLocs.begin(), selectedLocs.end(), true);

	std::map <std::string, std::vector <float>> features;
	std::vector <float> feature(nbSelectedLocs);
	for (std::map <std::string, poca::core::MyData*>::const_iterator it = featuresDset.begin(); it != featuresDset.end(); it++) {
		size_t cpt = 0;
		const std::vector <float>& values = it->second->getData<float>();
		for (size_t n = 0; n < values.size(); n++)
			if (selectedLocs[n])
				feature[cpt++] = values[n];
		features[it->first] = feature;
	}

	poca::core::BasicComponentInterface* voro = oneColorObj->getBasicComponent("VoronoiDiagram");
	const std::vector <float>& densities = voro != NULL && voro->hasData("density") ? voro->getMyData("density")->getData<float>() : std::vector <float>();
	if (!densities.empty()) {
		for (size_t n = 0, cpt = 0; n < densities.size(); n++)
			if (selectedLocs[n])
				feature[cpt++] = densities[n];
		features["density"] = feature;
	}

	poca::geometry::DetectionSet* dset = new poca::geometry::DetectionSet(features);

	const std::string& dir = obj->getDir(), name = obj->getName();
	QString newName(name.c_str());
	int index = newName.lastIndexOf(".");
	newName.insert(index, "_selectedObjects");

	poca::core::MyObject* wobj = new poca::core::MyObject();
	wobj->setDir(dir.c_str());
	wobj->setName(newName.toLatin1().data());
	wobj->addBasicComponent(dset);
	wobj->setDimension(m_objects->dimension());

	return wobj;*/
	poca::geometry::ObjectListInterface* oli = m_objects->exportSelectedObjects(_selectedObjects);
	if (oli == NULL) return NULL;
	poca::core::MyObjectInterface* obj = poca::core::Engine::instance()->getObject(m_objects);
	const std::string& dir = obj->getDir(), name = obj->getName();
	QString newName(name.c_str());
	int index = newName.lastIndexOf(".");
	newName.insert(index, "_selectedObjects");

	poca::core::MyObject* wobj = new poca::core::MyObject();
	wobj->setDir(dir.c_str());
	wobj->setName(newName.toLatin1().data());
	poca::core::CommandInfo com;
	poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(oli, com, "ObjectListPlugin");
	wobj->addBasicComponent(objsList);
	wobj->setDimension(m_objects->dimension());

	return wobj;
}

void ObjectListBasicCommands::saveSelectedObjectsForVectorHeat(const std::set<int>& _selectedObjects) const
{
	if (_selectedObjects.empty())
		return;

	poca::core::MyObjectInterface* obj = poca::core::Engine::instance()->getObject(m_objects);
	const std::string& dir = obj->getDir(), name = obj->getName();
	QString newName(name.c_str());
	int index = newName.lastIndexOf(".");
	newName.insert(index, "_selectedObjectVectorHeat");

	const poca::core::MyArrayUInt32& objs = m_objects->getLocOutlines();
	const std::vector <uint32_t>& data = objs.getData();
	const std::vector <uint32_t>& firsts = objs.getFirstElements();
	const std::vector <poca::core::Vec3mf>& normals = m_objects->getNormalOutlineLocs();

	std::vector <poca::core::Vec3mf> outlineLocs;
	m_objects->generateOutlineLocs(outlineLocs);

	QString filename = dir.c_str() + QString("/") + newName;
	std::ofstream fs(filename.toStdString());

	auto idObj = *_selectedObjects.begin();
	for (uint32_t cur = firsts[idObj]; cur < firsts[idObj + 1]; cur++) {
		fs << "v " << outlineLocs[cur].x() << " " << outlineLocs[cur].y() << " " << outlineLocs[cur].z() << std::endl;
		fs << "vn " << normals[cur].x() << " " << normals[cur].y() << " " << normals[cur].z() << std::endl;
	}
	fs.close();
}

void ObjectListBasicCommands::saveAsSVG(const QString& _filename) const
{
	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::opengl::CameraInterface* cam = engine->getCamera(m_objects);
	poca::core::PaletteInterface* pal = m_objects->getPalette();
	poca::core::Vec3mf direction = poca::core::Vec3mf(cam->getEye().x, cam->getEye().y, cam->getEye().z);

	glm::vec3 orientation = cam->getRotationSum() * glm::vec3(0.f, 0.f, 1.f);
	glm::vec3 pos(orientation + cam->getCenter());
	pos *= 2 * cam->getOriginalDistanceOrtho();

	poca::core::BoundingBox bbox = m_objects->boundingBox();
	glm::vec2 p1 = cam->worldToScreenCoordinates(glm::vec3(bbox[0], bbox[1], bbox[2]));
	glm::vec2 p2 = cam->worldToScreenCoordinates(glm::vec3(bbox[3], bbox[4], bbox[5]));
	poca::core::Vec2mf bottomLeft(p1[0] < p2[0] ? p1[0] : p2[0], p1[1] < p2[1] ? p1[1] : p2[1]);
	poca::core::Vec2mf upRight(p1[0] > p2[0] ? p1[0] : p2[0], p1[1] > p2[1] ? p1[1] : p2[1]);
	const float width = upRight[0] - bottomLeft[0];
	const float height = upRight[1] - bottomLeft[1];
	std::ofstream fs(_filename.toStdString());
	fs << std::setprecision(5) << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
	fs << "<svg xmlns=\"http://www.w3.org/2000/svg\"\n";
	fs << "     xmlns:xlink=\"http://www.w3.org/1999/xlink\"\n     width=\"" << width << "\" height=\"" << height << "\" viewBox=\"" << bottomLeft[0] << " " << bottomLeft[1] << " " << width << " " << height << " " "\">\n";
	fs << "<title>d:/gl2ps/type_svg_outSimple.svg</title>\n";
	fs << "<desc>\n";
	fs << "Creator: Florian Levet\n";
	fs << "</desc>\n";
	fs << "<defs>\n";
	fs << "</defs>\n";

	poca::geometry::ObjectListMesh* omesh = dynamic_cast <poca::geometry::ObjectListMesh*>(m_objects);
	if (omesh) {
		poca::opengl::CameraInterface* cam = engine->getCamera(m_objects);
		poca::core::Vec3mf direction = poca::core::Vec3mf(cam->getEye().x, cam->getEye().y, cam->getEye().z);

		std::vector <poca::core::Vec3mf> triangles;
		m_objects->generateTriangles(triangles);
		std::vector <poca::core::Vec3mf> normals;
		m_objects->generateNormals(normals);

		poca::core::Histogram<float>* histogram = dynamic_cast <poca::core::Histogram<float>*>(m_objects->getCurrentHistogram());
		const std::vector<float>& values = histogram->getValues();
		const std::vector<bool>& selection = m_objects->getSelection();
		float minH = histogram->getMin(), maxH = histogram->getMax(), interH = maxH - minH;

		std::vector <float> featureValues;
		m_objects->getFeatureInSelection(featureValues, values, selection, std::numeric_limits <float>::max());

		bool fill = false;
		if (m_objects->hasParameter("fill"))
			fill = m_objects->getParameter<bool>("fill");

		/*glm::vec3 orientation = cam->getRotationSum() * glm::vec3(0.f, 0.f, 1.f);
		glm::vec3 posTmp(orientation + cam->getCenter());
		posTmp *= 2 * cam->getOriginalDistanceOrtho();
		poca::core::Vec3mf lightPos(posTmp.x, posTmp.y, posTmp.z), lightColor(1.0f, 1.0f, 1.0f);
		poca::core::Vec3mf viewPos(lightPos);*/

		poca::core::Vec3mf lightPos = direction * 2, lightColor(1.0f, 1.0f, 1.0f);
		poca::core::Vec3mf viewPos(lightPos);

		char col[32], black[32];
		unsigned char rb = 0, gb = 0, bb = 0;
		poca::core::getColorStringUC(rb, gb, bb, black);

		for (size_t n = 0; n < triangles.size(); n += 3) {
			if (featureValues[n] == std::numeric_limits <float>::max()) continue;
			float valPal = (featureValues[n] - minH) / interH;
			poca::core::Color4uc c = pal->getColor(valPal);
			unsigned char r = c[0], g = c[1], b = c[2];
			poca::core::getColorStringUC(r, g, b, col);

			poca::core::Vec3mf normal = (normals[n] + normals[n + 1] + normals[n + 2]) / 3.f;
			normal.normalize();

			if (direction.dot(normal) < 0.f) {
				/*poca::core::Vec3mf centroid = (triangles[n] + triangles[n + 1] + triangles[n + 2]) / 3.f;

				float ambientStrength = 0.1;
				poca::core::Vec3mf ambient = ambientStrength * lightColor;
				// diffuse \n"
				poca::core::Vec3mf lightDir = (centroid - lightPos).normalize();
				float diff = fabs(normal.dot(lightDir));
				poca::core::Vec3mf diffuse = diff * lightColor;
				// specular\n"
				float specularStrength = 0.5;
				poca::core::Vec3mf viewDir = (viewPos - centroid).normalize();
				poca::core::Vec3mf reflectDir = lightDir - normal * 2.0 * normal.dot(lightDir);
				float spec = pow(fabs(viewDir.dot(reflectDir)), 32);
				poca::core::Vec3mf specular = specularStrength * spec * lightColor;
				poca::core::Vec3mf result;
				poca::core::Vec3mf newColor(c[0], c[1], c[2]);
				newColor = newColor * (ambient + diffuse + specular);
				unsigned char r = newColor[0], g = newColor[1], b = newColor[2];
				poca::core::getColorStringUC(r, g, b, col);*/

				if (fill) {
					glm::vec2 p1 = cam->worldToScreenCoordinates(glm::vec3(triangles[n].x(), triangles[n].y(), triangles[n].z()));
					glm::vec2 p2 = cam->worldToScreenCoordinates(glm::vec3(triangles[n + 1].x(), triangles[n + 1].y(), triangles[n + 1].z()));
					glm::vec2 p3 = cam->worldToScreenCoordinates(glm::vec3(triangles[n + 2].x(), triangles[n + 2].y(), triangles[n + 2].z()));

					fs << "<polygon points =\"";
					fs << p1.x << ",";
					fs << p1.y << " ";
					fs << p2.x << ",";
					fs << p2.y << " ";
					fs << p3.x << ",";
					fs << p3.y << "\" stroke=\"" << col << "\" fill=\"" << col << "\" stroke-width=\"0.1\"/>\n";
				}
				else {
					size_t idx[] = { n, n + 1, n + 2 };
					for (size_t i = 0; i < 3; i++) {
						size_t i1 = idx[i], i2 = idx[(i + 1) % 3];
						glm::vec2 p1 = cam->worldToScreenCoordinates(glm::vec3(triangles[i1].x(), triangles[i1].y(), triangles[i1].z()));
						glm::vec2 p2 = cam->worldToScreenCoordinates(glm::vec3(triangles[i2].x(), triangles[i2].y(), triangles[i2].z()));
						fs << "<line x1 =\"";
						fs << p1.x << "\" y1=\"";
						fs << p1.y << "\" x2=\"";
						fs << p2.x << "\" y2=\"";
						fs << p2.y << "\" stroke=\"" << /*black*/col << "\" stroke-width=\"1\"/>\n";
					}
				}
			}
		}
	}
	poca::geometry::ObjectListPolygon* opol = dynamic_cast <poca::geometry::ObjectListPolygon*>(m_objects);
	if (opol) {
		poca::core::Histogram<float>* histogram = dynamic_cast <poca::core::Histogram<float>*>(m_objects->getCurrentHistogram());
		const std::vector<float>& values = histogram->getValues();
		const std::vector<bool>& selection = m_objects->getSelection();
		float minH = histogram->getMin(), maxH = histogram->getMax(), interH = maxH - minH;

		size_t cur = 0;
		char col[32];
		const auto& polygons = opol->getPolygons();
		for (const auto& polygonWithHoles : polygons) {
			float valPal = (values[cur] - minH) / interH;
			poca::core::Color4uc c = pal->getColor(valPal);
			unsigned char r = c[0], g = c[1], b = c[2];
			poca::core::getColorStringUC(r, g, b, col);
			fs << "<path fill = \"" << col << "\" fill-rule = \"evenodd\" d = \" ";
			for (const auto& polygon : polygonWithHoles) {
				fs << "M";
				for (const auto& vertex : polygon)
					fs << " " << vertex.x() - bbox[0] << "," << vertex.y() - bbox[1];
				fs << std::endl;
			}
			fs << "\"/>" << std::endl;
		}
	}
	fs.close();
}

void ObjectListBasicCommands::saveAsOBJ(const QString& _filename) const
{
	poca::geometry::ObjectListMesh* omesh = dynamic_cast <poca::geometry::ObjectListMesh*>(m_objects);
	if (omesh != NULL) {
		omesh->saveAsOBJ(_filename.toStdString());
		std::cout << "File " << _filename.toStdString() << "has been saved." << std::endl;
		return;
	}
	poca::geometry::ObjectListPolygon* opol = dynamic_cast <poca::geometry::ObjectListPolygon*>(m_objects);
	if (opol != NULL) {
		opol->saveAsPol(_filename.toStdString());
		std::cout << "File " << _filename.toStdString() << "has been saved." << std::endl;
		return;
	}
	std::cout << "Saving file " << _filename.toStdString() << "failed." << std::endl;
}
