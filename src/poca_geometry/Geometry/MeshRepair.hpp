/*
* Software:  PoCA: Point Cloud Analyst
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Geometry/CGAL_includes.hpp>

namespace poca::geometry {

	// Bits 0..6 preserve the historical ObjectList mesh-diagnostic meanings.
	enum MeshIssueMask : std::uint32_t {
		MeshIssueNone = 0u,
		MeshIssueSelfIntersection = 1u << 0,
		MeshIssueDegenerate = 1u << 1,
		MeshIssueBoundary = 1u << 2,
		MeshIssueNonFinite = 1u << 3,
		MeshIssueNonTriangle = 1u << 4,
		MeshIssueDisconnected = 1u << 5,
		MeshIssueNonFiniteNormal = 1u << 6,
		MeshIssueNonManifold = 1u << 7,
		MeshIssueInvalidPolygonMesh = 1u << 8,
		MeshIssueOrientation = 1u << 9,
		MeshIssueVolume = 1u << 10,
		MeshIssueEmpty = 1u << 11
	};

	// Repair bits are intentionally independent from diagnostic issue bits.
	enum MeshRepairMask : std::uint32_t {
		MeshRepairNone = 0u,
		MeshRepairSelfIntersection = 1u << 0,
		MeshRepairDegenerateFaces = 1u << 1,
		MeshRepairBoundary = 1u << 2,
		MeshRepairDisconnectedComponents = 1u << 3,
		MeshRepairTriangulatedFaces = 1u << 4,
		MeshRepairOrientation = 1u << 5,
		MeshRepairNonManifoldTopology = 1u << 6
	};

	struct MeshFaceIssue {
		std::size_t faceIndex = 0;
		std::uint32_t issueMask = MeshIssueNone;
	};

	struct MeshInspection {
		std::size_t vertexCount = 0;
		std::size_t edgeCount = 0;
		std::size_t faceCount = 0;

		bool empty = true;
		bool finiteCoordinates = false;
		bool validPolygonMesh = false;
		bool triangleMesh = false;
		std::size_t nonTriangleFaceCount = 0;
		std::size_t degenerateTriangleCount = 0;

		bool closed = false;
		std::size_t borderEdgeCount = 0;
		std::size_t borderFaceCount = 0;

		std::size_t connectedComponentCount = 0;
		std::vector<std::size_t> connectedComponentFaceCounts;
		std::size_t facesOutsideLargestComponent = 0;

		std::vector<std::pair<std::size_t, std::size_t>> selfIntersectionPairs;
		std::size_t nonManifoldVertexCount = 0;

		bool normalsChecked = false;
		bool finiteFaceNormals = false;
		bool finiteVertexNormals = false;
		std::size_t nonFiniteFaceNormalCount = 0;
		std::size_t nonFiniteVertexNormalCount = 0;
		std::size_t facesWithNonFiniteNormals = 0;

		bool orientationChecked = false;
		bool outwardOriented = false;
		bool boundsVolumeChecked = false;
		bool boundsVolume = false;
		bool volumeChecked = false;
		bool finiteVolume = false;
		double volume = std::numeric_limits<double>::quiet_NaN();

		std::uint32_t issueMask = MeshIssueNone;
		std::vector<MeshFaceIssue> problemFaces;
		std::vector<std::string> inspectionErrors;
	};

	enum class MeshRepairStatus {
		Clean,
		Repaired,
		Rejected
	};

	struct MeshRepairCounts {
		std::size_t triangulatedFaces = 0;
		std::size_t removedDegenerateFaces = 0;
		std::size_t removedComponents = 0;
		std::size_t removedComponentFaces = 0;
		std::size_t stitchedBorderPairs = 0;
		std::size_t boundaryLoops = 0;
		std::size_t filledHoles = 0;
		std::size_t createdHoleFaces = 0;
		std::size_t duplicatedNonManifoldVertices = 0;
		std::size_t repairedSelfIntersectionClusters = 0;
		std::size_t repairedSelfIntersectionPairs = 0;
		std::size_t removedSelfIntersectionPatchFaces = 0;
		std::size_t createdSelfIntersectionPatchFaces = 0;
		std::size_t selectedSelfIntersectionRing = 0;
	};

	struct MeshRepairResult {
		MeshRepairStatus status = MeshRepairStatus::Rejected;
		MeshInspection before;
		MeshInspection after;
		Surface_mesh_3_double repairedMesh;
		std::uint32_t repairMask = MeshRepairNone;
		std::uint32_t attemptedRepairMask = MeshRepairNone;
		MeshRepairCounts counts;
		std::vector<std::string> steps;
		std::vector<std::string> failures;
	};

	class MeshRepair {
	public:
		static MeshInspection inspect(const Surface_mesh_3_double&);
		static MeshRepairResult repair(const Surface_mesh_3_double&);
		static bool isStrictlyValid(const MeshInspection&);
		static std::vector<std::string> strictValidationFailures(const MeshInspection&);

	private:
		static std::vector<std::vector<std::size_t>> clusterIntersectionPairs(
			const std::vector<std::pair<std::size_t, std::size_t>>&);
	};
}
