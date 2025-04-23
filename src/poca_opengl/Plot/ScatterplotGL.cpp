/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ScatterplotGL.cpp
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
#include <GL/gl.h>
#include <QtGui/QOpenGLFramebufferObject>
#include <QtGui/QMouseEvent>
#include <glm/gtx/string_cast.hpp>

#include <General/Palette.hpp>
#include <General/Misc.h>
#include <OpenGL/Shader.hpp>
#include <OpenGL/TextDisplayer.hpp>

#include "ScatterplotGL.hpp"
#include "../fontstash/fontstash.h"

namespace poca::plot {
	ScatterplotGL::ScatterplotGL(QWidget* _w, QWidget* _parent, Qt::WindowFlags _f) :QOpenGLWidget(_parent, _f), m_fbo(NULL), m_textureLutID(0), m_scatterplots{NULL, NULL}, m_leftButtonOn(false), m_radius(5.f), m_texDisplayer(NULL), m_recomputeFrame(true), m_intensity(1.f)
	{
		this->setObjectName("ScatterplotGL");
		this->setMouseTracking(true);
		QObject::connect(this, SIGNAL(actionNeededSignal(const QString&)), _w, SLOT(actionNeeded(const QString&)));
	}

	ScatterplotGL::~ScatterplotGL()
	{
		freeGPU();
	}

	void ScatterplotGL::setScatterPlot(poca::core::ScatterplotInterface* _scatter1, poca::core::ScatterplotInterface* _scatter2, const bool _log)
	{
		m_log = _log;
		if (m_scatterplots[0] != _scatter1 || m_scatterplots[1] != _scatter2) {
			m_scatterplots[0] = _scatter1;
			m_scatterplots[1] = _scatter2;
			makeCurrent();
			m_recomputeFrame = true;
			createDisplay();
		}
	}

	void ScatterplotGL::createDisplay()
	{
		for (size_t n = 0; n < 2; n++)
			m_pointBuffer[n].freeGPUMemory();

		float xmin = FLT_MAX, xmax = -FLT_MAX, ymin = FLT_MAX, ymax = -FLT_MAX;
		for (size_t n = 0; n < 2; n++) {
			std::vector <poca::core::Vec3mf> points(m_scatterplots[n]->getNbValues());
			const std::vector <poca::core::Vec2mf>& scatterPoints = m_scatterplots[n]->getPoints();
			for (size_t n = 0; n < scatterPoints.size(); n++) {
				points[n].set(scatterPoints[n][0], scatterPoints[n][1], 0.f);
				if (points[n][0] < xmin) xmin = points[n][0];
				if (points[n][0] > xmax) xmax = points[n][0];
				if (points[n][1] < ymin) ymin = points[n][1];
				if (points[n][1] > ymax) ymax = points[n][1];
			}
			m_pointBuffer[n].generateBuffer(points.size(), 512 * 512, 3, GL_FLOAT);
			m_pointBuffer[n].updateBuffer(points.data());
		}

		float w = xmax - xmin, h = ymax - ymin;
		m_dimensions.set(xmin, ymin, xmax, ymax);

		recalcProjMatrix();

		m_simplePointBuffer.generateBuffer(1, 512 * 512, 2, GL_FLOAT);
		std::vector <poca::core::Vec2mf> p = { poca::core::Vec2mf(0.f, 0.f) };
		m_simplePointBuffer.updateBuffer(p);

		m_originalDistanceOrtho = m_distanceOrtho = w > h ? w / 2 : h / 2;
	}

	void ScatterplotGL::computeFrame()
	{
		if (!m_recomputeFrame) return;

		float w = m_dimensionsProj[2] - m_dimensionsProj[0], h = m_dimensionsProj[3] - m_dimensionsProj[1];
		float smallestD = w < h ? w : h;
		//We target 6 main ticks on the smallest axis
		float step = floor(smallestD / 6.f), displacement = smallestD * 0.05f;
		if (step < 1.f) step = 1.f;

		//First, create frame text to determine the max width and height of texts
		//vertical direction for draw text is inverted -> we need to recompute the y coordinate
		//TO DO: try to change the text display to have the same vertical direction
		float x, y, maxWText = -FLT_MAX, maxHText = m_texDisplayer->lineHeight();
		if (m_dimensionsProj[0] < 0.f) {
			for (x = -step; x > m_dimensionsProj[0]; x -= step) {
				std::string text = std::to_string((int)x);
				float wT = m_texDisplayer->widthOfStr(text.c_str(), 0, 0);
				if (wT > maxWText) maxWText = wT;
			}
		}
		x = 0.f;
		for (; x < m_dimensionsProj[2]; x += step) {
			if (x < m_dimensionsProj[0]) continue;
			std::string text = std::to_string((int)x);
			float wT = m_texDisplayer->widthOfStr(text.c_str(), 0, 0);
			if (wT > maxWText) maxWText = wT;
		}
		std::cout << "Maxs = " << maxWText << ", " << maxHText << std::endl;
		maxWText += 3;
		maxHText += 3;
		m_clipSpaceScreenTexts = glm::vec2(maxWText, maxHText);
		m_clipSpaceWorldTexts = screenToWorldCoordinates(m_clipSpaceScreenTexts);
		
		//Second, we create the frame text
		//Since gl clipping seems to not work with the text display, we directly test to get rid of text on the border of the window
		//TO DO: manage to repair the clipping problem
		m_frameHorizontalText.clear();
		m_frameVerticalText.clear();
		if (m_dimensionsProj[0] < 0.f) {
			for (x = -step; x > m_dimensionsProj[0]; x -= step) {
				glm::vec2 coords = worldToScreenCoordinates(glm::vec2(x, m_dimensionsProj[3] - displacement));
				if (coords.x < maxWText) continue;
				m_frameHorizontalText.push_back(std::make_pair(coords, std::to_string((int)x)));
			}
		}
		x = 0.f;
		for (; x < m_dimensionsProj[2]; x += step) {
			if (x < m_dimensionsProj[0]) continue;
			glm::vec2 coords = worldToScreenCoordinates(glm::vec2(x, m_dimensionsProj[3] - displacement));
			if (coords.x < maxWText) continue;
			m_frameHorizontalText.push_back(std::make_pair(coords, std::to_string((int)x)));
		}
		if (m_dimensionsProj[1] < 0.f) {
			for (y = -step; y > m_dimensionsProj[1]; y -= step) {
				glm::vec2 coords = worldToScreenCoordinates(glm::vec2(m_dimensionsProj[0], m_dimensionsProj[3] - (y - m_dimensionsProj[1])));
				if (coords.y > (this->height() - maxHText)) continue;
				m_frameVerticalText.push_back(std::make_pair(coords, std::to_string((int)y)));
			}
		}
		y = 0.f;
		for (; y < m_dimensionsProj[3]; y += step) {
			if (y < m_dimensionsProj[1]) continue;
			glm::vec2 coords = worldToScreenCoordinates(glm::vec2(m_dimensionsProj[0], m_dimensionsProj[3] - (y - m_dimensionsProj[1])));
			if (coords.y > (this->height() - maxHText)) continue;
			m_frameVerticalText.push_back(std::make_pair(coords, std::to_string((int)y)));
		}

		//Now we create the actual frame (lines). We use the maw width and height of texts
		//to determine where to end the lines, to not overlap with the text
		std::vector <poca::core::Vec2mf> frame;
		if (m_dimensionsProj[0] < 0.f) {
			for (x = -step; x > m_dimensionsProj[0]; x -= step) {
				frame.push_back(poca::core::Vec2mf(x, m_dimensionsProj[1] + displacement));
				frame.push_back(poca::core::Vec2mf(x, m_dimensionsProj[3]));
			}
		}
		x = 0.f;
		for (; x < m_dimensionsProj[2]; x += step) {
			if (x < m_dimensionsProj[0]) continue;
			frame.push_back(poca::core::Vec2mf(x, m_dimensionsProj[1] + displacement));
			frame.push_back(poca::core::Vec2mf(x, m_dimensionsProj[3]));
		}
		if (m_dimensionsProj[1] < 0.f) {
			for (y = -step; y > m_dimensionsProj[1]; y -= step) {
				frame.push_back(poca::core::Vec2mf(m_dimensionsProj[0] + displacement, y));
				frame.push_back(poca::core::Vec2mf(m_dimensionsProj[2], y));
			}
		}
		y = 0.f;
		for (; y < m_dimensionsProj[3]; y += step) {
			if (y < m_dimensionsProj[1]) continue;
			frame.push_back(poca::core::Vec2mf(m_dimensionsProj[0] + displacement, y));
			frame.push_back(poca::core::Vec2mf(m_dimensionsProj[2], y));
		}
		m_frameBuffer.generateBuffer(frame.size(), 512 * 512, 2, GL_FLOAT);
		m_frameBuffer.updateBuffer(frame.data());

		m_recomputeFrame = false;
	}

	glm::vec2 ScatterplotGL::worldToScreenCoordinates(const glm::vec2& _wpos) const
	{
		glm::vec4 clipSpacePos = m_matrixProjection * glm::vec4(_wpos, 0.f, 1.f);
		glm::vec3 ndcSpacePos = glm::vec3(clipSpacePos.x, clipSpacePos.y, clipSpacePos.z) / clipSpacePos.w;
		glm::vec2 viewOffset(m_viewport.x, m_viewport.y), viewSize(m_viewport[2], m_viewport[3]);
		glm::vec2 windowSpacePos = ((glm::vec2(ndcSpacePos.x, ndcSpacePos.y) + 1.f) / 2.f) * viewSize + viewOffset;
		return windowSpacePos;
	}

	glm::vec2 ScatterplotGL::screenToWorldCoordinates(const glm::vec2& _winCoords) const
	{
		return glm::unProject(glm::vec3(_winCoords, 0), glm::mat4(1.f), m_matrixProjection, m_viewport);
	}

	void ScatterplotGL::freeGPU()
	{
		if (m_textureLutID == 0) return;
		if (m_textureLutID != 0)
			glDeleteTextures(1, &m_textureLutID);
		for(size_t n = 0; n < 2; n++)
			m_pointBuffer[n].freeGPUMemory();
		m_quadVertBuffer.freeGPUMemory();
		m_quadUvBuffer.freeGPUMemory();
		m_textureLutID = 0;
	}

	void ScatterplotGL::recalcProjMatrix()
	{
		float w = m_dimensions[2] - m_dimensions[0], h = m_dimensions[3] - m_dimensions[1];
		float factorW = 1.f, factorH = 1.f;
		float diffX = this->width(), diffY = this->height();
		float rescalingW = 0.f, rescalingH = 0.f;
		if (diffX > diffY) {
			float factor = diffX / diffY;
			rescalingW = (factor - 1.f) * h;
		}
		else {
			float factor = diffY / diffX;
			rescalingH = (factor - 1.f) * w;
		}
		m_dimensionsProj.set(m_dimensions[0] - rescalingW / 2.f,
			m_dimensions[1] - rescalingH / 2.f,
			m_dimensions[2] + rescalingW / 2.f,
			m_dimensions[3] + rescalingH / 2.f);
		m_matrixProjection = glm::ortho(m_dimensionsProj[0], m_dimensionsProj[2], m_dimensionsProj[1], m_dimensionsProj[3]);
	}

	void ScatterplotGL::resizeEvent(QResizeEvent* _event)
	{
		makeCurrent();
		QOpenGLWidget::resizeEvent(_event);

		m_viewport = glm::uvec4(0, 0, this->width(), this->height());
		if (m_fbo != NULL)
			delete m_fbo;
		int w = this->width(), h = this->height();
		m_fbo = new QOpenGLFramebufferObject(this->width(), this->height(), QOpenGLFramebufferObject::Depth, GL_TEXTURE_2D, GL_RGB);
		glBindTexture(GL_TEXTURE_2D, m_fbo->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, this->width(), this->height(), 0, GL_RGB, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);

		recalcProjMatrix();
		m_recomputeFrame = true;
		update();
	}

	void ScatterplotGL::wheelEvent(QWheelEvent* _event)
	{
		float mult = _event->angleDelta().y() < 0 ? 1.f : -1.f;
		float factorZoom = 0.1 * mult;
		float w = m_dimensionsProj[2] - m_dimensionsProj[0], h = m_dimensionsProj[3] - m_dimensionsProj[1];
		float modifW = w * factorZoom, modifH = h * factorZoom;
		m_dimensionsProj[0] -= modifW;
		m_dimensionsProj[1] -= modifH;
		m_dimensionsProj[2] += modifW;
		m_dimensionsProj[3] += modifH;
		m_matrixProjection = glm::ortho(m_dimensionsProj[0], m_dimensionsProj[2], m_dimensionsProj[1], m_dimensionsProj[3]);
		m_recomputeFrame = true;

		update();
	}

	void ScatterplotGL::mousePressEvent(QMouseEvent* _event)
	{
		switch (_event->button())
		{
		case Qt::LeftButton:
			m_leftButtonOn = true;
			setClickPoint(_event->pos().x(), _event->pos().y());

			glm::vec2 glm_p = screenToWorldCoordinates(glm::vec2(_event->pos().x(), _event->pos().y()));
			m_curPosWorldCoords.set(glm_p.x, m_dimensionsProj[3] - (glm_p.y - m_dimensionsProj[1]));
			std::vector <poca::core::Vec2mf> p = { m_curPosWorldCoords };
			m_simplePointBuffer.updateBuffer(p);
			emit(actionNeededSignal("applyThresholdonClick"));
			break;
		}
	}

	void ScatterplotGL::mouseMoveEvent(QMouseEvent* _event)
	{
		if (m_leftButtonOn) {
			setClickPoint(_event->pos().x(), _event->pos().y());
			float w = m_dimensionsProj[2] - m_dimensionsProj[0], h = m_dimensionsProj[3] - m_dimensionsProj[1];
			float px = w / (float)this->width(), py = h / (float)this->height();
			glm::vec2 diff = m_clickPoint - m_prevClickPoint;
			m_dimensionsProj[0] -= diff[0] * px;
			m_dimensionsProj[1] += diff[1] * py;
			m_dimensionsProj[2] -= diff[0] * px;
			m_dimensionsProj[3] += diff[1] * py;
			m_matrixProjection = glm::ortho(m_dimensionsProj[0], m_dimensionsProj[2], m_dimensionsProj[1], m_dimensionsProj[3]);
			m_recomputeFrame = true;
		}
		glm::vec2 glm_p = screenToWorldCoordinates(glm::vec2(_event->pos().x(), _event->pos().y()));
		m_curPosWorldCoords.set(glm_p.x, m_dimensionsProj[3] - (glm_p.y - m_dimensionsProj[1]));
		std::vector <poca::core::Vec2mf> p = { m_curPosWorldCoords };
		m_simplePointBuffer.updateBuffer(p);
		update();
	}

	void ScatterplotGL::mouseReleaseEvent(QMouseEvent* _event)
	{
		m_leftButtonOn = false;
	}

	void ScatterplotGL::setClickPoint(double x, double y)
	{
		m_prevClickPoint = m_clickPoint;
		m_clickPoint.x = x;
		m_clickPoint.y = y;
	}

	void ScatterplotGL::initializeGL()
	{
		GLenum estado = glewInit();
		if (estado != GLEW_OK)
		{
			std::cerr << "Failed to initialize Glew." << std::endl;
			exit(EXIT_FAILURE);
		}
		const unsigned char* glver = glGetString(GL_VERSION);
		std::cout << glver << std::endl;

		//glShadeModel(GL_SMOOTH);
		glClearColor(1.f, 1.f, 1.f, 1.f);
		glDisable(GL_COLOR_MATERIAL);

		glGenTextures(1, &m_textureLutID);
		poca::core::PaletteInterface* palette = poca::core::Palette::getStaticLutPtr("Heatmap");
		setPalette(palette);
		delete palette;

		std::vector <poca::core::Vec3mf> quadVertices = {
			// positions        // texture Coords
			poca::core::Vec3mf(-1.f,  1.f, 0.0f),
			poca::core::Vec3mf(-1.f, -1.f, 0.0f),
			poca::core::Vec3mf(1.f,  1.f, 0.0f),
			poca::core::Vec3mf(1.f, -1.f, 0.0f)
		};
		std::vector <poca::core::Vec2mf> quadUVs = {
			// positions        // texture Coords
			poca::core::Vec2mf(0.0f, 1.0f),
			poca::core::Vec2mf(0.0f, 0.0f),
			poca::core::Vec2mf(1.0f, 1.0f),
			poca::core::Vec2mf(1.0f, 0.0f),
		};

		m_quadVertBuffer.generateBuffer(4, 512 * 512, 3, GL_FLOAT);
		m_quadUvBuffer.generateBuffer(4, 512 * 512, 2, GL_FLOAT);

		m_quadVertBuffer.updateBuffer(quadVertices.data());
		m_quadUvBuffer.updateBuffer(quadUVs.data());

		char* vsHeatmap = "#version 330 core\n"
			"layout(location = 0) in vec3 vertexPosition_modelspace;\n"
			"void main() {\n"
			"	vec4 pos = vec4(vertexPosition_modelspace, 1);\n"
			"	gl_Position = pos;\n"
			"}";

		char* gsHeatmap = "#version 330 core\n"
			"layout (points) in;\n"
			"layout (triangle_strip, max_vertices = 4) out;\n"
			"uniform mat4 MVP;\n"
			"uniform mat4 projection;\n"
			"uniform float radius;\n"
			"uniform vec2 pixelSize;\n"
			"out vec2 TexCoords_GS;\n"
			"out vec3 center;\n"
			"const vec2 uv[4] = vec2[4](vec2(1,1), vec2(1,0), vec2(0,1),	vec2(0,0));\n"
			"void main() {\n"
			"	vec4 vc[4];\n"
			"	vc[0] = vec4(radius, radius, 0.0, 0.0);\n"
			"	vc[1] = vec4(radius, -radius, 0.0, 0.0);\n"
			"	vc[2] = vec4(-radius, radius, 0.0, 0.0);\n"
			"	vc[3] = vec4(-radius, -radius, 0.0, 0.0);\n"
			"	for(int i = 0; i < 4; i++){\n"
			"		gl_Position = (MVP * gl_in[0].gl_Position) + (vc[i] * vec4(pixelSize, 0.f, 0.f));\n"
			"		TexCoords_GS = uv[i];\n"
			"		center = gl_in[0].gl_Position.xyz;\n"
			"		EmitVertex();\n"
			"	}\n"
			"	EndPrimitive();\n"
			"}";

		char* fsHeatmap = "#version 330 core\n"
			"layout (location = 0) out float gColor;\n"
			"in vec2 TexCoords_GS;\n"
			"in vec3 center;\n"
			"uniform float weight;\n"
			"uniform highp float u_intensity;\n"
			"#define GAUSS_COEF 0.3989422804014327\n"
			"void main() {\n"
			"	vec2 texC =  ( TexCoords_GS - 0.5 ) * 2;\n"
			"	float d = dot(texC, texC);\n"
			"	if (d > 1.0f) { discard; }\n"
			"	float d2 = -0.5 * 3.0 * 3.0 * d;\n"
			"	float val = weight * u_intensity * GAUSS_COEF * exp(d2);\n"
			"	gColor = val;\n"
			"}";

		char* vsTexture = "#version 330 core\n"
			"layout (location = 0) in vec3 aPos;\n"
			"layout (location = 1) in vec2 aTexCoords;\n"
			"out vec2 TexCoords;\n"
			"void main() {\n"
			"	TexCoords = aTexCoords;\n"
			"	vec4 pos = vec4(aPos, 1.0);"
			"	gl_Position = pos;\n"
			"}";

		char* fsTexture = "#version 330 core\n"
			"in vec2 TexCoords;\n"
			"out vec4 color;\n"
			"uniform sampler2D offscreen;\n"
			"uniform sampler1D lut;\n"
			"void main() {\n"
			"	float coord = texture( offscreen, TexCoords ).x;\n"
			"	if(coord <= 0.f)\n"
			"		discard;\n"
			"	color = texture( lut, coord );\n"
			"}";


		char* vs = "#version 330 core\n"
			"layout(location = 0) in vec2 vertexPosition_modelspace;\n"
			"uniform mat4 MVP;\n"
			"uniform vec4 clipPlaneX;\n"
			"uniform vec4 clipPlaneY;\n"
			"void main() {\n"
			"	vec4 pos = vec4(vertexPosition_modelspace, 0, 1);\n"
			"	gl_Position = MVP * pos;\n"
			"	gl_ClipDistance[0] = dot(pos, clipPlaneX);\n"
			"	gl_ClipDistance[1] = dot(pos, clipPlaneY);\n"
			"}";

		char* fs = "#version 330 core\n"
			"out vec4 color;\n"
			"uniform vec4 singleColor;\n"
			"void main() {\n"
			"	color = singleColor;\n"
			"}";

		m_shaderHeatmap = new poca::opengl::Shader();
		m_shaderHeatmap->createAndLinkProgramFromStr(vsHeatmap, fsHeatmap, gsHeatmap);
		m_shaderTexture = new poca::opengl::Shader();
		m_shaderTexture->createAndLinkProgramFromStr(vsTexture, fsTexture);
		m_simpleShader = new poca::opengl::Shader();
		m_simpleShader->createAndLinkProgramFromStr(vs, fs);
}

	void ScatterplotGL::paintGL()
	{
		if (m_scatterplots[0] == NULL || m_scatterplots[1] == NULL) return;
#ifndef NDEBUG
		GLint drawFboId = 0, readFboId = 0;
		glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFboId);
		glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFboId);
		std::cout << "Scatter read buf: " << readFboId << ", draw buf: " << drawFboId << ", default buf: " << defaultFramebufferObject() << std::endl;
#endif
		glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebufferObject());

		if (m_pointBuffer[0].empty())
			createDisplay();

		if (m_texDisplayer == NULL) {
			m_texDisplayer = new poca::opengl::TextDisplayer();
			m_texDisplayer->setFontSize(20.f);
		}

		if (m_fbo == NULL) return;

		computeFrame();

		glViewport(m_viewport[0], m_viewport[1], m_viewport[2], m_viewport[3]);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CLIP_DISTANCE0);
		glDisable(GL_CLIP_DISTANCE1);

		glEnable(GL_BLEND);
		glClearColor(0.f, 0.f, 0.f, 0.f);
		glBlendFunc(GL_ONE, GL_ONE);
		bool success = m_fbo->bind();
		if (!success) std::cout << "Problem with binding" << std::endl;
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		float extrude = 1.f, opacity = 1.f, weight = 1.f;
		for (size_t n = 0; n < 2; n++)
			drawHeatmap(m_pointBuffer[n], m_intensity, weight, m_radius);
		success = m_fbo->release();
		if (!success) std::cout << "Problem with releasing" << std::endl;
		GL_CHECK_ERRORS();

		glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebufferObject());

		glClearColor(0.99f, 0.99f, 0.99f, 0.f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		drawTextureFBO(m_fbo->texture(), m_textureLutID);
		glDisable(GL_BLEND);


		const glm::mat4& proj = m_matrixProjection, & view = glm::mat4(1.f), & model = glm::mat4(1.f);// glm::translate(glm::mat4(1.f), m_translationModel);
		m_simpleShader->use();
		m_simpleShader->setMat4("MVP", proj * view * model);
		glEnableVertexAttribArray(0);
		glPointSize(8.f);
		glEnable(GL_POINT_SMOOTH);
		m_simpleShader->setVec4("singleColor", 1.f, 0.f, 0.f, 1.f);
		m_simplePointBuffer.bindBuffer(0, 0);
		glDrawArrays(m_simplePointBuffer.getMode(), 0, m_simplePointBuffer.getSizeBuffers()[0]);
		
		glEnable(GL_CLIP_DISTANCE0);
		glEnable(GL_CLIP_DISTANCE1);
		glm::vec4 clip[2];
		clip[0] = glm::vec4(1, 0, 0, -m_clipSpaceWorldTexts[0]);
		clip[1] = glm::vec4(0, 1, 0, -m_clipSpaceWorldTexts[1]);
		m_simpleShader->setVec4("clipPlaneX", clip[0]);
		m_simpleShader->setVec4("clipPlaneY", clip[1]);
		m_simpleShader->setVec4("singleColor", 0.f, 0.f, 0.f, 1.f);
		const std::vector <size_t>& sizeStridesLocs = m_frameBuffer.getSizeBuffers();
		for (size_t chunk = 0; chunk < m_frameBuffer.getNbBuffers(); chunk++) {
			m_frameBuffer.bindBuffer(chunk, 0);
			glDrawArrays(m_frameBuffer.getMode(), 0, (GLsizei)sizeStridesLocs[chunk]); // Starting from vertex 0; 3 vertices total -> 1 triangle
		}
		glDisableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		m_simpleShader->release();

		glDisable(GL_CLIP_DISTANCE0);
		glDisable(GL_CLIP_DISTANCE1);
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		float x = 10.f, y = 10.f;
		glm::mat4 projText = glm::ortho(0.f, (float)this->width(), (float)this->height(), 0.f);
		m_texDisplayer->setClipPlane(glm::vec4(-1, 0, 0, -m_clipSpaceScreenTexts[0]));
		for(size_t n = 0; n < m_frameHorizontalText.size(); n++)
			poca::core::Vec2mf dxy = m_texDisplayer->renderText(projText, m_frameHorizontalText.at(n).second.c_str(), 0, 0, 0, 255, m_frameHorizontalText.at(n).first[0], m_frameHorizontalText.at(n).first[1], 0, FONS_ALIGN_CENTER | FONS_ALIGN_TOP);
		m_texDisplayer->setClipPlane(glm::vec4(0, 1, 0, m_clipSpaceScreenTexts[1]));
		for (size_t n = 0; n < m_frameVerticalText.size(); n++)
			poca::core::Vec2mf dxy = m_texDisplayer->renderText(projText, m_frameVerticalText.at(n).second.c_str(), 0, 0, 0, 255, m_frameVerticalText.at(n).first[0], m_frameVerticalText.at(n).first[1], 0, FONS_ALIGN_LEFT | FONS_ALIGN_MIDDLE);

		float xthresh = m_log ? pow(10, m_curPosWorldCoords.x()) : m_curPosWorldCoords.x();
		float ythresh = m_log ? pow(10, m_curPosWorldCoords.y()) : m_curPosWorldCoords.y();
		std::string textThresh = "[" + std::to_string(xthresh) + ", " + std::to_string(ythresh) + "]";
		poca::core::Vec2mf dxy = m_texDisplayer->renderText(projText, textThresh.c_str(), 128, 0, 128, 255, this->width(), 0, 0, FONS_ALIGN_RIGHT | FONS_ALIGN_TOP);

		glDisable(GL_BLEND);

		GL_CHECK_ERRORS();
	}

	void ScatterplotGL::renderQuad() const
	{
		glEnableVertexAttribArray(0);
		m_quadVertBuffer.bindBuffer(0, 0);
		glEnableVertexAttribArray(1);
		m_quadUvBuffer.bindBuffer(0, 1);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, m_quadVertBuffer.getSizeBuffers()[0]); // Starting from vertex 0; 3 vertices total -> 1 triangle
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
	}

	void ScatterplotGL::drawHeatmap(const poca::opengl::PointGLBuffer <poca::core::Vec3mf>& _bufferVertex, const GLfloat _intensity, const GLfloat _weight, const GLfloat _radius)
	{
		float px = 2.f / (float)this->width(), py = 2.f / (float)this->height();
		const glm::mat4& proj = m_matrixProjection, & view = glm::mat4(1.f), & model = glm::mat4(1.f);// glm::translate(glm::mat4(1.f), m_translationModel);
		m_shaderHeatmap->use();
		m_shaderHeatmap->setMat4("MVP", proj * view * model);
		m_shaderHeatmap->setMat4("projection", proj);
		m_shaderHeatmap->setFloat("u_intensity", _intensity);
		m_shaderHeatmap->setFloat("weight", _weight);
		m_shaderHeatmap->setFloat("radius", _radius);
		m_shaderHeatmap->setVec2("pixelSize", px, py);//NDC coordinates are from -1 to 1

		const std::vector <size_t>& sizeStrides = _bufferVertex.getSizeBuffers();
		glEnableVertexAttribArray(0);
		for (unsigned int chunk = 0; chunk < _bufferVertex.getNbBuffers(); chunk++) {
			_bufferVertex.bindBuffer(chunk, 0);
			glDrawArrays(_bufferVertex.getMode(), 0, (GLsizei)sizeStrides[chunk]); // Starting from vertex 0; 3 vertices total -> 1 triangle
		}
		glDisableVertexAttribArray(0);

		m_shaderHeatmap->release();
	}

	void ScatterplotGL::drawTextureFBO(const GLuint _textureIDOffscreen, const GLuint _textureIDLUT)
	{
		m_shaderTexture->use();
		m_shaderTexture->setInt("offscreen", 0);
		m_shaderTexture->setInt("lut", 1);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, _textureIDOffscreen);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_1D, _textureIDLUT);
		renderQuad();
		m_shaderTexture->release();
	}

	void ScatterplotGL::setPalette(poca::core::PaletteInterface* _palette)
	{
		poca::core::Color4uc c = _palette->colorAt(0);
		c[3] = 0;
		_palette->setColorAt(0, c);
		unsigned int sizeLut = 512;
		unsigned int cpt = 0;
		std::vector <float> lutValues(sizeLut * 4);
		float stepLut = 1. / (float)(sizeLut - 1);
		for (float val = 0.f; val <= 1.f; val += stepLut) {
			poca::core::Color4uc c = _palette->getColor(val);
			lutValues[cpt++] = (float)c[0] / 255.f;
			lutValues[cpt++] = (float)c[1] / 255.f;
			lutValues[cpt++] = (float)c[2] / 255.f;
			lutValues[cpt++] = (float)c[3] / 255.f;
		}
		glBindTexture(GL_TEXTURE_1D, m_textureLutID);
		glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, sizeLut, 0, GL_RGBA, GL_FLOAT, lutValues.data());
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_1D, 0);
	}
}

