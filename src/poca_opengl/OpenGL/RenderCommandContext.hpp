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

#include <vector>

#include <General/Command.hpp>
#include <General/Vec3.hpp>

class QOpenGLFramebufferObject;

namespace poca::opengl {
	class Camera;
}

namespace poca::opengl {
	struct ActiveCamera {
		poca::opengl::Camera* camera = nullptr;
	};

	struct PickingFramebuffer {
		QOpenGLFramebufferObject* fbo = nullptr;
	};

	struct DetectionSetNormals {
		const std::vector<poca::core::Vec3mf>* normals = nullptr;
	};
}

#endif
