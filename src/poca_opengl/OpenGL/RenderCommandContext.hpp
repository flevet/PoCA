/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      RenderCommandContext.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef RenderCommandContext_h__
#define RenderCommandContext_h__

#include <string>
#include <vector>

#include <General/Command.hpp>
#include <General/Vec3.hpp>
#include <General/Vec6.hpp>
#include <Interfaces/MyObjectInterface.hpp>

class QOpenGLFramebufferObject;

namespace poca::opengl {
	class Camera;
}

namespace poca::opengl {
	struct ActiveCamera {
		poca::opengl::Camera* camera = nullptr;
	};

	struct DeferImageListOverlays {
		bool defer = false;
	};

	struct RenderedComponentFamilies {
		std::vector<std::string> componentNames;
	};

	// Runtime marker set by a parent-side render command when it has already
	// handled child-object rendering for a MyMultipleObject. This lets the
	// container short-circuit per-child forwarding without moving batching
	// policy into poca_core.
	struct ChildObjectRenderingHandled {
		bool handled = false;
	};

	struct PickingFramebuffer {
		QOpenGLFramebufferObject* fbo = nullptr;
	};

	struct DetectionSetNormals {
		const std::vector<poca::core::Vec3mf>* normals = nullptr;
	};

	struct PickedInfoTextResult {
		std::string info;
	};

	struct PickedInfoListResult {
		poca::core::stringList infos;
	};

	struct PickedPointsResult {
		std::vector<poca::core::Vec3mf> points;
	};

	struct PickedBoundingBoxResult {
		poca::core::BoundingBox bbox;
		bool valid = false;
	};

	struct PickedObjectIdResult {
		int id = -1;
		bool valid = false;
	};

	struct CameraMouseEventHandled {
		bool handled = false;
	};
}

#endif
