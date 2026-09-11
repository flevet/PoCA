#include <Windows.h>
#include <GL/glew.h>
#include "ObjectListMultiObjectDisplayCommand.hpp"
#include <Objects/MyMultipleObject.hpp>
#include <OpenGL/Camera.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

void ObjectListMultiObjectDisplayCommand::drawTransparentRanges(poca::opengl::Camera* camera, bool ssao)
{
	struct Draw {
		const ListDrawRange* list;
		const ObjectTriangleRange* triangles;
		float depth;
	};
	std::vector<Draw> draws;
	const glm::mat4 parentInverse = glm::inverse(m_object->getModelMatrix());
	const glm::mat4 parentToView = camera->getViewMatrix()*camera->getModelMatrix();
	for (const auto& range : m_listDrawRanges) {
		if (!usesTransparentMeshPass(range)) continue;
		if (range.objectTriangles.empty()) throw std::runtime_error("Transparent ObjectList has no child triangle ranges.");
		for (const auto& span : range.objectTriangles) {
			auto child = m_object->getObject(span.objectIndex);
			if (!child) throw std::runtime_error("Transparent ObjectList child is unavailable.");
			// Match the exact shader transform, including parent and child gizmo matrices.
			const auto position = parentToView*parentInverse*child->getModelMatrix()*glm::vec4(span.centroid, 1.f);
			if (!std::isfinite(position.z)) throw std::runtime_error("Non-finite transparent ObjectList view depth.");
			draws.push_back({ &range, &span, position.z });
		}
	}
	// One ordering across all list indices: a later list must not repaint a farther child over a nearer one.
	std::stable_sort(draws.begin(), draws.end(), [](const Draw& a, const Draw& b) { return a.depth < b.depth; });
	for (const auto& draw : draws) {
		const auto& source = *draw.list; const auto& span = *draw.triangles;
		ListDrawRange mesh;
		mesh.listIndex = source.listIndex; mesh.displayCommand = source.displayCommand;
		mesh.textureLutID = source.textureLutID; mesh.is3D = source.is3D;
		mesh.minOriginalFeature = source.minOriginalFeature; mesh.maxOriginalFeature = source.maxOriginalFeature;
		mesh.triangleFirst = span.first; mesh.triangleCount = span.count;
		mesh.objectTriangles.push_back(span);
		drawListRange(camera, ssao, mesh);
	}
}
