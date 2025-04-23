/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ScatterplotGL.hpp
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

#ifndef ScatterplotGL_hpp__
#define ScatterplotGL_hpp__

#include <QtWidgets/QOpenGLWidget>

#include <glm/glm.hpp>

#include <OpenGL/GLBuffer.hpp>
#include <Interfaces/ScatterplotInterface.hpp>
#include <Interfaces/PaletteInterface.hpp>
#include <General/Vec2.hpp>
#include <General/Vec3.hpp>
#include <General/Vec4.hpp>

class QOpenGLFramebufferObject;
namespace poca::opengl {
	class Shader;
	class TextDisplayer;
}

namespace poca::plot {
	class ScatterplotGL : public QOpenGLWidget {
		Q_OBJECT

	public:
		ScatterplotGL(QWidget*, QWidget* = 0, Qt::WindowFlags = 0);
		~ScatterplotGL();

		void setScatterPlot(poca::core::ScatterplotInterface*, poca::core::ScatterplotInterface*, const bool = false);
		void setRadiusHeatmap(const float _val) { m_radius = _val; }
		void setIntensityHeatmap(const float _val) { m_intensity = _val; }
		void setPalette(poca::core::PaletteInterface*);

		inline const float getThresholdX() const { return m_log ? pow(10, m_curPosWorldCoords.x()) : m_curPosWorldCoords.x(); }
		inline const float getThresholdY() const { return m_log ? pow(10, m_curPosWorldCoords.y()) : m_curPosWorldCoords.y(); }

	protected:
		void resizeEvent(QResizeEvent*);
		void wheelEvent(QWheelEvent*);
		void mousePressEvent(QMouseEvent*);
		void mouseMoveEvent(QMouseEvent*);
		void mouseReleaseEvent(QMouseEvent*);
		void initializeGL();
		void paintGL();

		void createDisplay();
		void freeGPU();
		void recalcProjMatrix();
		void setClickPoint(double, double);

		void renderQuad() const;
		void drawHeatmap(const poca::opengl::PointGLBuffer <poca::core::Vec3mf>&, const GLfloat, const GLfloat, const GLfloat);
		void drawTextureFBO(const GLuint, const GLuint);

		void computeFrame();
		glm::vec2 worldToScreenCoordinates(const glm::vec2&) const;
		glm::vec2 screenToWorldCoordinates(const glm::vec2&) const;

	signals:
		void actionNeededSignal(const QString&);

	protected:
		glm::mat4 m_matrixProjection;
		glm::vec3 m_translationModel;
		glm::uvec4 m_viewport;
		float m_precX, m_precY, m_distanceOrtho, m_originalDistanceOrtho, m_radius, m_intensity;
		poca::core::Vec4mf m_dimensions, m_dimensionsProj;

		bool m_leftButtonOn, m_recomputeFrame;
		glm::vec2 m_prevClickPoint, m_clickPoint, m_clipSpaceWorldTexts, m_clipSpaceScreenTexts;
		poca::core::Vec2mf m_curPosWorldCoords;

		GLuint m_textureLutID;
		poca::opengl::Shader* m_shaderHeatmap, * m_shaderTexture, * m_simpleShader;
		QOpenGLFramebufferObject* m_fbo;
		poca::opengl::PointGLBuffer <poca::core::Vec3mf> m_quadVertBuffer, m_pointBuffer[2];
		poca::opengl::FeatureGLBuffer <poca::core::Vec2mf> m_quadUvBuffer;
		poca::opengl::PointGLBuffer <poca::core::Vec2mf> m_simplePointBuffer;

		poca::opengl::LineGLBuffer <poca::core::Vec2mf> m_frameBuffer;
		poca::opengl::TextDisplayer* m_texDisplayer;
		std::vector <std::pair <glm::vec2, std::string>> m_frameHorizontalText, m_frameVerticalText;

		bool m_log;

		poca::core::ScatterplotInterface* m_scatterplots[2];
	};
}
#endif

