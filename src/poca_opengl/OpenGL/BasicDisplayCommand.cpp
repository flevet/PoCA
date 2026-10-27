/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      BasicDisplayCommand.cpp
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

#include <Windows.h>
#include <GL/glew.h>

#include <QtGui/QOpenGLContext>
#include <QtGui/QOpenGLFramebufferObject>
#include <QtGui/QImage>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <General/BasicComponent.hpp>
#include <Interfaces/BasicComponentInterface.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <OpenGL/Camera.hpp>
#include <OpenGL/RenderCommandContext.hpp>

#include "BasicDisplayCommand.hpp"

namespace poca::opengl {
	namespace {
		bool isValidBoundingBox(const poca::core::BoundingBox& _bbox)
		{
			for (int n = 0; n < 6; n++)
				if (!std::isfinite(_bbox[n]))
					return false;
			return _bbox[0] <= _bbox[3] && _bbox[1] <= _bbox[4] && _bbox[2] <= _bbox[5];
		}

		poca::core::BoundingBox expandedBoundingBox(poca::core::BoundingBox _bbox)
		{
			float maxExtent = 0.f;
			for (int axis = 0; axis < 3; axis++)
				maxExtent = std::max(maxExtent, _bbox[axis + 3] - _bbox[axis]);

			const float epsilon = std::max(maxExtent * 1e-4f, 1e-3f);
			for (int axis = 0; axis < 3; axis++) {
				const float extent = _bbox[axis + 3] - _bbox[axis];
				if (extent >= epsilon)
					continue;
				const float center = (_bbox[axis] + _bbox[axis + 3]) * 0.5f;
				_bbox[axis] = center - epsilon * 0.5f;
				_bbox[axis + 3] = center + epsilon * 0.5f;
			}
			return _bbox;
		}

		poca::core::BoundingBox transformBoundingBox(const poca::core::BoundingBox& _bbox, const glm::mat4& _model)
		{
			poca::core::BoundingBox transformed = poca::core::BoundingBox::initBBox();
			if (!isValidBoundingBox(_bbox))
				return transformed;

			const float xs[2] = { _bbox[0], _bbox[3] };
			const float ys[2] = { _bbox[1], _bbox[4] };
			const float zs[2] = { _bbox[2], _bbox[5] };
			for (int ix = 0; ix < 2; ix++) {
				for (int iy = 0; iy < 2; iy++) {
					for (int iz = 0; iz < 2; iz++) {
						const glm::vec4 corner = _model * glm::vec4(xs[ix], ys[iy], zs[iz], 1.f);
						if (!std::isfinite(corner.x) || !std::isfinite(corner.y) || !std::isfinite(corner.z))
							continue;
						transformed.addPointBBox(corner.x, corner.y, corner.z);
					}
				}
			}
			return transformed;
		}

		void mergeBoundingBox(poca::core::BoundingBox& _bbox, const poca::core::BoundingBox& _other)
		{
			for (int axis = 0; axis < 3; axis++)
				_bbox[axis] = std::min(_bbox[axis], _other[axis]);
			for (int axis = 3; axis < 6; axis++)
				_bbox[axis] = std::max(_bbox[axis], _other[axis]);
		}

		bool intersectRayBoundingBox(const glm::vec3& _origin, const glm::vec3& _direction, const poca::core::BoundingBox& _bbox, float& _distance)
		{
			float tMin = -std::numeric_limits<float>::max();
			float tMax = std::numeric_limits<float>::max();

			for (int axis = 0; axis < 3; axis++) {
				const float origin = _origin[axis];
				const float direction = _direction[axis];
				const float minValue = _bbox[axis];
				const float maxValue = _bbox[axis + 3];

				if (std::abs(direction) < 1e-8f) {
					if (origin < minValue || origin > maxValue)
						return false;
					continue;
				}

				float t1 = (minValue - origin) / direction;
				float t2 = (maxValue - origin) / direction;
				if (t1 > t2)
					std::swap(t1, t2);
				tMin = std::max(tMin, t1);
				tMax = std::min(tMax, t2);
				if (tMin > tMax)
					return false;
			}

			if (tMax < 0.f)
				return false;

			_distance = tMin >= 0.f ? tMin : tMax;
			return true;
		}
	}

	std::vector<poca::core::CommandSpec> BasicDisplayCommand::commandSpecs() const
	{
		using poca::core::CommandParameterType;
		using poca::core::CommandSpec;

		return {
			CommandSpec("updatePickingBuffer", {
				{"width", CommandParameterType::Integer, true, nullptr},
				{"height", CommandParameterType::Integer, true, nullptr}
			}),
			CommandSpec("pick", {
				{"x", CommandParameterType::Integer, true, nullptr},
				{"y", CommandParameterType::Integer, true, nullptr},
				{"saveImage", CommandParameterType::Boolean, false, false}
			}),
			CommandSpec("freeGPU"),
			CommandSpec("togglePicking"),
			CommandSpec("setIDObjectPicked")
		};
	}

	BasicDisplayCommand::BasicDisplayCommand(poca::core::BasicComponentInterface* _component, const std::string& _name) : poca::core::Command(_name), m_pickFBO(NULL), m_wImage(0), m_hImage(0), m_idSelection(-1), m_pickingEnabled(true)
	{
		m_component = _component;
		if (_name == "ObjectListDisplayCommand") m_pickProfileFamily = PickProfiler::PickFamily::ObjectList;
		else if (_name == "DetectionSetDisplayCommand") m_pickProfileFamily = PickProfiler::PickFamily::DetectionSet;
	}

	BasicDisplayCommand::BasicDisplayCommand(const BasicDisplayCommand& _o) :Command(_o), m_component(_o.m_component), m_pickFBO(NULL), m_wImage(0), m_hImage(0), m_idSelection(_o.m_idSelection), m_pickingEnabled(_o.m_pickingEnabled)
	{
		m_pickProfileFamily = _o.m_pickProfileFamily;
	}

	BasicDisplayCommand::~BasicDisplayCommand()
	{
		releasePickingGpuProfile();
		releasePickingProfile();
		if (m_pickFBO != NULL)
			delete m_pickFBO;
		m_pickFBO = NULL;
	}

	bool BasicDisplayCommand::skipDisabledPickingCommand(poca::core::CommandInfo* info)
	{
		if (poca::core::POCA_ENABLE_PICKING) return false;
		m_idSelection = -1;
		if (!info) return true;
		return info->nameCommand == "pick" || info->nameCommand == "updatePickingBuffer" ||
			info->nameCommand == "getObjectPickedID" || info->nameCommand == "getInfoObjectCurrentlyPicked" ||
			info->nameCommand == "setIDObjectPicked" ||
			(info->nameCommand == "doubleClickCamera" && !info->hasParameter("objectID"));
	}

	bool BasicDisplayCommand::skipDisabledPickingCommand(poca::core::CommandInfo* info, poca::core::CommandExecutionResult& result)
	{
		if (!skipDisabledPickingCommand(info)) return false;
		result = poca::core::CommandExecutionResult{}; // Existing no-hit contract: no selected ID/info/point/bbox/preview payload.
		return true;
	}

	void BasicDisplayCommand::execute(poca::core::CommandInfo* _infos)
	{
		if (skipDisabledPickingCommand(_infos)) return;
		if (_infos == NULL)
			return;
		if (_infos->nameCommand == "updatePickingBuffer") {
			if (!_infos->hasParameter("width") || !_infos->hasParameter("height"))
				return;
			int w = _infos->getParameter<int>("width"), h = _infos->getParameter<int>("height");
			updatePickingFBO(w, h);
		}
		else if (_infos->nameCommand == "pick") {
			PickProfiler::pickCount(m_pickProfileFamily, PickProfiler::PickMetric::ComponentsVisited);
			if (!m_pickingEnabled) return;
			if (!_infos->hasParameter("x") || !_infos->hasParameter("y"))
				return;
			int x = _infos->getParameter<int>("x"), y = _infos->getParameter<int>("y");
			bool saveImage = _infos->getParameterOr<bool>("saveImage", false);
			pick(x, y, saveImage);
		}
		else if (_infos->nameCommand == "freeGPU") {
			releasePickingGpuProfile();
			releasePickingProfile();
			if (m_pickFBO != NULL)
				delete m_pickFBO;
			m_pickFBO = NULL;
		}
		else if(_infos->nameCommand == "togglePicking"){
			bool val = _infos->getParameterOr<bool>("togglePicking", m_pickingEnabled);
			m_pickingEnabled = val;
		}
		else if (_infos->nameCommand == "setIDObjectPicked") {
			if (!_infos->hasParameter("setIDObjectPicked"))
				return;
			int id = _infos->getParameter<int>("setIDObjectPicked");
			m_idSelection = id;
		}
	}

	void BasicDisplayCommand::execute(poca::core::CommandInfo* _infos, const poca::core::CommandExecutionContext&)
	{
		execute(_infos);
	}

	void BasicDisplayCommand::execute(poca::core::CommandInfo* _infos, const poca::core::CommandExecutionContext& _context, poca::core::CommandExecutionResult& _result)
	{
		if (skipDisabledPickingCommand(_infos, _result)) return;
		execute(_infos, _context);
	}

	void BasicDisplayCommand::pick(const int _x, const int _y, const bool _saveImage)
	{
		if (!poca::core::POCA_ENABLE_PICKING) { m_idSelection = -1; return; }
		int sizeImage = m_wImage * m_hImage;
		if (m_pickFBO == NULL) return;
		const bool profile = m_pickProfileFamily != PickProfiler::PickFamily::None && PickProfiler::renderInstance().renderMetricsEnabled();
		const bool bound = m_pickFBO->bind();
		if (profile) {
			PickProfiler::pickCount(m_pickProfileFamily, PickProfiler::PickMetric::FboBindAttempts);
			PickProfiler::pickCount(m_pickProfileFamily, bound ? PickProfiler::PickMetric::FboBinds : PickProfiler::PickMetric::FboBindFailures);
		}
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		float pixel, * pixs = NULL;
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		{
			PickProfiler::RenderTimer timer(PickProfiler::pickMetric(m_pickProfileFamily, PickProfiler::PickMetric::ReadpixelsUs), profile);
			if (profile) PickProfiler::pickCount(m_pickProfileFamily, PickProfiler::PickMetric::ReadpixelsCalls);
			glReadPixels(_x, m_hImage - _y - 1, 1, 1, GL_RED, GL_FLOAT, &pixel);
		}
		if (_saveImage) {
			pixs = new float[sizeImage];
			PickProfiler::RenderTimer timer(PickProfiler::pickMetric(m_pickProfileFamily, PickProfiler::PickMetric::ReadpixelsUs), profile);
			if (profile) PickProfiler::pickCount(m_pickProfileFamily, PickProfiler::PickMetric::ReadpixelsCalls);
			glReadPixels(0, 0, m_wImage, m_hImage, GL_RED, GL_FLOAT, pixs);
		}
		glReadBuffer(GL_NONE);
		m_pickFBO->release();

		m_idSelection = (int)(pixel - 1);
		if (profile) PickProfiler::pickCount(m_pickProfileFamily, m_idSelection >= 0 ? PickProfiler::PickMetric::ReadHits : PickProfiler::PickMetric::ReadMisses);

		if (_saveImage) {
			QImage image2 = QImage(m_wImage, m_hImage, QImage::Format_Grayscale16);
			for (int j = 0; j < m_hImage; ++j) {
				quint16* dst = (quint16*)(image2.bits() + (m_hImage - j - 1) * image2.bytesPerLine());
				for (int i = 0; i < m_wImage; ++i) {
					int index = i + j * m_wImage, index2 = index, index3 = m_wImage - i - 1;
					dst[i] = (quint16)pixs[index2];
				}
			}
			QString name = QString("e:/pick_obj.png");
			image2.save(name);
			quint16* tmp = (quint16*)image2.bits();
			quint16 val = tmp[m_hImage * _y + _x];
			std::cout << "For [" << _x << ", " << _y << "], val in image = " << val << std::endl;
			delete[] pixs;
		}
	}

	poca::core::BoundingBox BasicDisplayCommand::childObjectBoundingBox(poca::core::MyObjectInterface* _parent, poca::core::MyObjectInterface* _child) const
	{
		if (_child == nullptr)
			return poca::core::BoundingBox::initBBox();

		const glm::mat4 parentInvModel = _parent != nullptr ? glm::inverse(_parent->getModelMatrix()) : glm::mat4(1.f);
		const glm::mat4 childToParentModel = parentInvModel * _child->getModelMatrix();
		poca::core::BoundingBox bbox = poca::core::BoundingBox::initBBox();
		bool hasBBox = false;

		for (poca::core::BasicComponentInterface* component : _child->getComponents()) {
			if (component == nullptr)
				continue;
			const poca::core::BoundingBox transformed = transformBoundingBox(component->boundingBox(), childToParentModel);
			if (!isValidBoundingBox(transformed))
				continue;
			mergeBoundingBox(bbox, transformed);
			hasBBox = true;
		}

		if (hasBBox)
			return bbox;
		return transformBoundingBox(_child->boundingBox(), parentInvModel);
	}

	bool BasicDisplayCommand::pickObjectBoundingBox(poca::core::CommandInfo* _infos, const poca::core::CommandExecutionContext& _context, poca::core::MyObjectInterface* _object, int& _id) const
	{
		_id = -1;
		if (!poca::core::POCA_ENABLE_PICKING) return true;
		if (!m_pickingEnabled)
			return true;
		if (_infos == nullptr || _object == nullptr || !_infos->hasParameter("x") || !_infos->hasParameter("y"))
			return false;
		if (!_context.has<poca::opengl::ActiveCamera>())
			return false;

		poca::opengl::Camera* cam = _context.get<poca::opengl::ActiveCamera>().camera;
		if (cam == nullptr || cam->getWidth() <= 0 || cam->getHeight() <= 0)
			return false;

		const int x = _infos->getParameter<int>("x");
		const int y = _infos->getParameter<int>("y");
		const float glY = static_cast<float>(cam->getHeight() - y);
		const glm::vec3 nearPoint = glm::unProject(glm::vec3(static_cast<float>(x), glY, 0.f),
			cam->getModelMatrix(), cam->getProjectionMatrix() * cam->getViewMatrix(), cam->getViewport());
		const glm::vec3 farPoint = glm::unProject(glm::vec3(static_cast<float>(x), glY, 1.f),
			cam->getModelMatrix(), cam->getProjectionMatrix() * cam->getViewMatrix(), cam->getViewport());

		glm::vec3 direction = farPoint - nearPoint;
		const float length2 = glm::dot(direction, direction);
		if (length2 < 1e-12f)
			return true;
		direction = glm::normalize(direction);

		float bestDistance = std::numeric_limits<float>::max();
		for (size_t n = 0; n < _object->nbColors(); n++) {
			poca::core::MyObjectInterface* child = _object->getObject(n);
			if (child == nullptr)
				continue;

			poca::core::BoundingBox bbox = childObjectBoundingBox(_object, child);
			if (!isValidBoundingBox(bbox))
				continue;
			bbox = expandedBoundingBox(bbox);

			float distance = 0.f;
			if (intersectRayBoundingBox(nearPoint, direction, bbox, distance) && distance < bestDistance) {
				bestDistance = distance;
				_id = static_cast<int>(n);
			}
		}
		return true;
	}

	poca::core::Command* BasicDisplayCommand::copy()
	{
		return new BasicDisplayCommand(*this);
	}

	void BasicDisplayCommand::profilePickingCamera(Camera* camera)
	{
		if (!PickProfiler::renderInstance().renderMetricsEnabled()) return;
		for (const auto& existing : m_pickProfileCameras) if (existing.data() == camera) return;
		m_pickProfileCameras.erase(std::remove_if(m_pickProfileCameras.begin(), m_pickProfileCameras.end(), [](const auto& value) { return value.isNull(); }), m_pickProfileCameras.end());
		m_pickProfileCameras.emplace_back(camera);
	}
	void BasicDisplayCommand::releasePickingGpuProfile()
	{
		for (auto& camera : m_pickProfileCameras) if (camera) camera->releasePickingGpuTimers();
		m_pickProfileCameras.clear();
	}

	void BasicDisplayCommand::releasePickingProfile()
	{
		if (!m_pickProfileLive) return;
		PickProfiler::RenderFrame resource(0, false, false, false, 0, m_wImage, m_hImage, false, "pick_resource",
			!PickProfiler::renderInstance().hasRenderRecord());
		PickProfiler::pickFboChange(m_pickProfileFamily, -1, m_pickProfileBytes, m_pickProfileKnown);
		m_pickProfileLive = m_pickProfileKnown = false; m_pickProfileBytes = 0;
	}

	void BasicDisplayCommand::updatePickingFBO(const int _w, const int _h)
	{
		if (!poca::core::POCA_ENABLE_PICKING) { m_idSelection = -1; return; }
		if (_w <= 0 || _h <= 0)
			return;
		if (m_pickFBO != NULL && m_wImage == _w && m_hImage == _h)
			return;
		const bool targeted = m_pickProfileFamily != PickProfiler::PickFamily::None;
		const bool profile = targeted && PickProfiler::renderInstance().renderMetricsEnabled();
		PickProfiler::RenderFrame resource(0, false, false, false, 0, _w, _h, false, "pick_resource", profile && !PickProfiler::renderInstance().hasRenderRecord());
		PickProfiler::RenderTimer timer(PickProfiler::pickMetric(m_pickProfileFamily, PickProfiler::PickMetric::FboOutsideDrawCpuUs), profile && !PickProfiler::renderInstance().inPickScope(m_pickProfileFamily));
		const bool reallocation = m_pickProfileLive;
		releasePickingProfile();
		m_wImage = _w;
		m_hImage = _h;
		if (m_pickFBO != NULL)
			delete m_pickFBO;
		m_pickFBO = new QOpenGLFramebufferObject(m_wImage, m_hImage, QOpenGLFramebufferObject::Depth, GL_TEXTURE_2D, GL_RED);
		glBindTexture(GL_TEXTURE_2D, m_pickFBO->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_wImage, m_hImage, 0, GL_RED, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
		if (targeted && m_pickFBO->isValid()) {
			m_pickProfileLive = true;
			if (profile && QOpenGLContext::currentContext()) {
				GLint previous = 0, depthBits = 0;
				glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previous);
				glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_pickFBO->handle());
				glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE, &depthBits);
				glBindFramebuffer(GL_DRAW_FRAMEBUFFER, previous);
				m_pickProfileKnown = depthBits > 0;
				if (m_pickProfileKnown) m_pickProfileBytes = uint64_t(_w) * uint64_t(_h) * (4 + (uint64_t(depthBits) + 7) / 8);
				PickProfiler::pickCount(m_pickProfileFamily, PickProfiler::PickMetric::FboAllocations);
				if (reallocation) PickProfiler::pickCount(m_pickProfileFamily, PickProfiler::PickMetric::FboReallocations);
			}
			PickProfiler::pickFboChange(m_pickProfileFamily, 1, m_pickProfileBytes, m_pickProfileKnown);
		}
	}
}

