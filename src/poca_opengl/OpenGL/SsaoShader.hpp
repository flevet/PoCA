/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      SsaoShader.hpp
*
* Copyright: Florian Levet (2020-2022)
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

#ifndef SSAOSHADER_h__
#define SSAOSHADER_h__

#include <glm/glm.hpp>

#include <OpenGL/GLBuffer.hpp>
#include <General/Vec2.hpp>
#include <General/Vec3.hpp>

class QOpenGLFramebufferObject;

namespace poca::opengl {
	class Shader;

	class SsaoShader {
	public:
		enum class SSAODebugDisplay { SSAO_NO_DEBUG = 0, SSAO_POS = 1, SSAO_NORMAL = 2, SSAO_COLOR = 3, SSAO_SSAO_MAP = 4 };

		SsaoShader();
		~SsaoShader();

		void renderQuad() const;
		void deleteFBOAndShaders();
		void deleteFBOs();
		void deleteShaders();
		void loadShaders();
		void init(const int, const int);
		void update(const int, const int);

		const Shader& getShaderSSAO() const { return *m_shaderSSAO; }
		const Shader& getShaderSSAOGeometry() const { return *m_shaderGeometryPass; }
		const Shader& getShaderSSAOBlur() const { return *m_shaderGeometryPass; }
		const Shader& getShaderSSAOLighting() const { return *m_shaderLightingPass; }


		Shader* m_shaderGeometryPass, * m_shaderLightingPass, * m_shaderSSAO, * m_shaderSSAOBlur, * m_shaderSilhouette, * m_shaderHalo;
		QOpenGLFramebufferObject* m_fboGeometry, * m_fboNoise, * m_fboBlur, * m_fboHalo, * m_fboSilhouette, * m_fboSilhouetteBlurred, * m_fboLighting;
		GLuint m_noiseTexture;
		std::vector<glm::vec3> m_ssaoKernel, m_ssaoNoise;
		std::vector<glm::vec2> m_ssaoCircle;
		glm::vec3 m_lightPos, m_lightColor;

		PointSingleGLBuffer <poca::core::Vec3mf> m_quadVertBuffer;
		FeatureSingleGLBuffer <poca::core::Vec2mf> m_quadUvBuffer;
	};
}
#endif

