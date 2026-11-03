/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include <Windows.h>
#include <GL/glew.h>
#include "Camera.hpp"
#include <Objects/MyMultipleObject.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <stdexcept>
#include <exception>

namespace poca::opengl {
	bool Camera::admitProgressiveGeometryInitialization()
	{
		const auto* multiple = dynamic_cast<MyMultipleObject*>(m_object);
		if (!multiple || multiple->isLoadComplete()) return true;
		// One cold component per frame after an image-first opportunity. Warm draws
		// retain their original order and selection. Offscreen/SSAO callers bypass this gate.
		if (interactiveRendering()) return false; // Existing camera settle update resumes refinement.
		if (m_progressiveGeometryInitializations) {
			--m_progressiveGeometryInitializations;
			return true;
		}
		requestImageStreamRedraw();
		return false;
	}

	void Camera::traceResetView() const
	{
		const auto* multiple = dynamic_cast<const MyMultipleObject*>(m_object);
		std::cout << "[PoCA][Zarr][Reset] ready=" << (multiple ? multiple->nbColors() : 1)
			<< " complete=" << (!multiple || multiple->isLoadComplete()) << " bounds=" << m_object->boundingBox()
			<< " crop=" << m_currentCrop << " cropped=" << isCropped()
			<< " clip=" << m_applyClippingPlanes << " extent=" << m_distanceOrtho << " distance=" << m_cameraDistance
			<< " translation=" << glm::to_string(m_stateCamera.m_translationModel)
			<< " projection=" << glm::to_string(m_matrixProjection)
			<< " view=" << glm::to_string(m_stateCamera.m_matrix)
			<< " parent=" << glm::to_string(m_object->getModelMatrix()) << std::endl;
		if (multiple && multiple->pendingLoadBounds())
			std::cout << "[PoCA][Zarr][Reset] pending_envelope=" << *multiple->pendingLoadBounds()
				<< " expected_children=" << multiple->expectedChildren() << std::endl;
		for (size_t i=0;i<m_clip.size();++i)
			std::cout << "[PoCA][Zarr][Reset] plane=" << i << " value=" << glm::to_string(m_clip[i]) << std::endl;
	}

	QImage Camera::imageContributionSnapshot() const
	{
		// Opt-in readback is deliberately synchronous and excluded from timing runs.
		// Preserve read-FBO and pixel-pack state; never redirect a renderer's output.
		GLint viewport[4], drawFbo, readFbo, readBuffer, drawBuffer, packBuffer;
		GLint alignment, rowLength, skipRows, skipPixels, samples, targetReadBuffer;
		glGetIntegerv(GL_VIEWPORT,viewport); glGetIntegerv(GL_SAMPLES,&samples);
		if (samples || viewport[2] <= 0 || viewport[3] <= 0) {
			std::cout << "[PoCA][Zarr][ImagePixels] unavailable: multisampled or empty target" << std::endl;
			return {};
		}
		glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&drawFbo); glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING,&readFbo);
		glGetIntegerv(GL_READ_BUFFER,&readBuffer); glGetIntegerv(GL_DRAW_BUFFER0,&drawBuffer);
		glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING,&packBuffer); glGetIntegerv(GL_PACK_ALIGNMENT,&alignment);
		glGetIntegerv(GL_PACK_ROW_LENGTH,&rowLength); glGetIntegerv(GL_PACK_SKIP_ROWS,&skipRows); glGetIntegerv(GL_PACK_SKIP_PIXELS,&skipPixels);
		QImage result(viewport[2],viewport[3],QImage::Format_RGBA8888);
		if (result.isNull()) throw std::runtime_error("Image contribution readback allocation failed");
		glBindFramebuffer(GL_READ_FRAMEBUFFER,drawFbo); glGetIntegerv(GL_READ_BUFFER,&targetReadBuffer); glReadBuffer(drawBuffer);
		glBindBuffer(GL_PIXEL_PACK_BUFFER,0); glPixelStorei(GL_PACK_ALIGNMENT,1);
		glPixelStorei(GL_PACK_ROW_LENGTH,0); glPixelStorei(GL_PACK_SKIP_ROWS,0); glPixelStorei(GL_PACK_SKIP_PIXELS,0);
		glReadPixels(viewport[0],viewport[1],viewport[2],viewport[3],GL_RGBA,GL_UNSIGNED_BYTE,result.bits());
		const auto error=glGetError();
		glPixelStorei(GL_PACK_ALIGNMENT,alignment); glPixelStorei(GL_PACK_ROW_LENGTH,rowLength);
		glPixelStorei(GL_PACK_SKIP_ROWS,skipRows); glPixelStorei(GL_PACK_SKIP_PIXELS,skipPixels); glBindBuffer(GL_PIXEL_PACK_BUFFER,packBuffer);
		glReadBuffer(targetReadBuffer); glBindFramebuffer(GL_READ_FRAMEBUFFER,readFbo); glReadBuffer(readBuffer);
		if (error!=GL_NO_ERROR) throw std::runtime_error("Image contribution readback GL error " + std::to_string(error));
		return result;
	}

	Camera::ImageContributionScope::ImageContributionScope(Camera* camera, const char* pass)
		: m_camera(camera->loadingVisibilityDiagnostics() ? camera : nullptr), m_pass(pass)
	{
		if (m_camera) m_before=m_camera->imageContributionSnapshot();
	}
	Camera::ImageContributionScope::~ImageContributionScope() noexcept(false)
	{
		// Never obscure an exception from the renderer with a second diagnostic readback.
		if (std::uncaught_exceptions() || !m_camera || m_before.isNull()) return;
		const auto after=m_camera->imageContributionSnapshot();
		if (after.isNull() || after.size()!=m_before.size()) return;
		size_t changed=0;
		for (int y=0;y<after.height();++y) {
			const auto* before=m_before.constScanLine(y); const auto* next=after.constScanLine(y);
			for (int x=0;x<after.width();++x)
				if (before[4*x]!=next[4*x] || before[4*x+1]!=next[4*x+1] || before[4*x+2]!=next[4*x+2]) ++changed;
		}
		std::cout << "[PoCA][Zarr][ImagePixels] pass=" << m_pass << " changed_rgb_pixels=" << changed
			<< " target_pixels=" << size_t(after.width())*after.height()
			<< " (image-pass contribution; inspect final framebuffer for later occlusion)" << std::endl;
	}
}