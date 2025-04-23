/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      SsaoShader.cpp
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

#include <GL/glew.h>
#include <GL/gl.h>
#include <QtGui/QOpenGLFramebufferObject>
#include <QtCore/QVector>
#include <QtCore/qmath.h>
#include <random>

#include <OpenGL/Shader.hpp>

#include "SsaoShader.hpp"

namespace poca::opengl {
	float lerp(float a, float b, float f)
	{
		return a + f * (b - a);
	}

	SsaoShader::SsaoShader() :m_lightPos(0.f, 0.f, 5.f), m_lightColor(1.f, 1.f, 1.f)
	{
		m_shaderGeometryPass = m_shaderLightingPass = m_shaderSSAO = m_shaderSSAOBlur = m_shaderSilhouette = m_shaderHalo = NULL;
		m_fboGeometry = m_fboNoise = m_fboBlur = m_fboHalo = m_fboSilhouette = m_fboSilhouetteBlurred = NULL;
		m_noiseTexture = 0;
	}

	SsaoShader::~SsaoShader()
	{
		deleteFBOAndShaders();
		if (m_noiseTexture != 0)
			glDeleteTextures(1, &m_noiseTexture);
		m_quadVertBuffer.freeGPUMemory();
		m_quadUvBuffer.freeGPUMemory();
	}

	// renderQuad() renders a 1x1 XY quad in NDC
	// -----------------------------------------
	void SsaoShader::renderQuad() const
	{
		glEnableVertexAttribArray(0);
		m_quadVertBuffer.bindBuffer(0);
		glEnableVertexAttribArray(1);
		m_quadUvBuffer.bindBuffer(0);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, m_quadVertBuffer.getNbElements()); // Starting from vertex 0; 3 vertices total -> 1 triangle
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
	}

	void SsaoShader::deleteFBOAndShaders()
	{
		deleteFBOs();
		deleteShaders();
	}

	void SsaoShader::deleteFBOs()
	{
		if (m_fboGeometry != NULL)
			delete m_fboGeometry;
		if (m_fboNoise != NULL)
			delete m_fboNoise;
		if (m_fboBlur != NULL)
			delete m_fboBlur;
		if (m_fboHalo != NULL)
			delete m_fboHalo;
		if (m_fboSilhouette != NULL)
			delete m_fboSilhouette;
		if (m_fboSilhouetteBlurred != NULL)
			delete m_fboSilhouetteBlurred;
	}

	void SsaoShader::deleteShaders()
	{
		if (m_shaderGeometryPass != NULL)
			delete m_shaderGeometryPass;
		if (m_shaderLightingPass != NULL)
			delete m_shaderLightingPass;
		if (m_shaderSSAO != NULL)
			delete m_shaderSSAO;
		if (m_shaderSSAOBlur != NULL)
			delete m_shaderSSAOBlur;
		if (m_shaderSilhouette != NULL)
			delete m_shaderSilhouette;
		if (m_shaderHalo != NULL)
			delete m_shaderHalo;
	}

	void SsaoShader::loadShaders()
	{
		deleteShaders();
		m_shaderGeometryPass = new Shader("./shaders/9.ssao_geometry.vert", "./shaders/9.ssao_geometry.frag", "./shaders/9.ssao_geometry.geom");
		m_shaderLightingPass = new Shader("./shaders/9.ssao.vert", "./shaders/9.ssao_lighting.frag");
		m_shaderSSAO = new Shader("./shaders/9.ssao.vert", "./shaders/9.ssao.frag");
		m_shaderSSAOBlur = new Shader("./shaders/9.ssao.vert", "./shaders/9.ssao_blur.frag");
		m_shaderSilhouette = new Shader("./shaders/9.ssao.vert", "./shaders/9.ssao_silhouette.frag");
		//m_shaderHalo = new Shader("./shaders/9.ssao_geometry.vert", "./shaders/9.ssao_halo.frag", "./shaders/9.ssao_halo.geom");
	}

	void SsaoShader::init(const int _w, const int _h)
	{
		loadShaders();
		// generate sample kernel
	// ----------------------
		std::uniform_real_distribution<GLfloat> randomFloats(0.0, 1.0); // generates random floats between 0.0 and 1.0
		std::default_random_engine generator;
		for (unsigned int i = 0; i < 64; ++i)
		{
			glm::vec3 sample(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, randomFloats(generator));
			glm::normalize(sample);
			sample *= randomFloats(generator);
			float scale = float(i) / 64.0;

			// scale samples s.t. they're more aligned to center of kernel
			scale = lerp(0.1f, 1.0f, scale * scale);
			sample *= scale;
			m_ssaoKernel.push_back(sample);
		}

		// generate noise texture
		// ----------------------
		for (unsigned int i = 0; i < 16; i++)
		{
			glm::vec3 noise(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, 0.0f); // rotate around z-axis (in tangent space)
			m_ssaoNoise.push_back(noise);
		}

		float twicePi = 2 * M_PI;
		//Generate unit circle vectors
		for (unsigned int i = 0; i < 64; ++i)
			m_ssaoCircle.push_back(glm::vec2(cos(i * twicePi / 64), sin(i * twicePi / 64)));

		// lighting info
		// -------------
		m_lightPos = glm::vec3(0, 0, 5.0);
		m_lightColor = glm::vec3(1., 1., 1.);

		// shader configuration
		// --------------------
		m_shaderLightingPass->use();
		m_shaderLightingPass->setInt("gPosition", 0);
		m_shaderLightingPass->setInt("gNormal", 1);
		m_shaderLightingPass->setInt("gAlbedo", 2);
		m_shaderLightingPass->setInt("ssao", 3);
		m_shaderLightingPass->setInt("silhouette", 4);
		m_shaderSSAO->use();
		m_shaderSSAO->setInt("gPosition", 0);
		m_shaderSSAO->setInt("gNormal", 1);
		m_shaderSSAO->setInt("texNoise", 2);
		m_shaderSSAOBlur->use();
		m_shaderSSAOBlur->setInt("ssaoInput", 0);
		m_shaderSilhouette->use();
		m_shaderSilhouette->setInt("ssaoInput", 0);

		m_quadVertBuffer.freeGPUMemory();
		std::vector <poca::core::Vec3mf> quadVertices = {
			// positions        // texture Coords
			poca::core::Vec3mf(-1.f,  1.f, 0.0f),
			poca::core::Vec3mf(-1.f, -1.f, 0.0f),
			poca::core::Vec3mf(1.f,  1.f, 0.0f),
			poca::core::Vec3mf(1.f, -1.f, 0.0f)
		};
		m_quadVertBuffer.generateBuffer(4, 3, GL_FLOAT);
		m_quadVertBuffer.updateBuffer(quadVertices.data());

		m_quadUvBuffer.freeGPUMemory();
		std::vector <poca::core::Vec2mf> quadUVs = {
			// positions        // texture Coords
			poca::core::Vec2mf(0.0f, 1.0f),
			poca::core::Vec2mf(0.0f, 0.0f),
			poca::core::Vec2mf(1.0f, 1.0f),
			poca::core::Vec2mf(1.0f, 0.0f),
		};
		m_quadUvBuffer.generateBuffer(4, 2, GL_FLOAT);
		m_quadUvBuffer.updateBuffer(quadUVs.data());

		update(_w, _h);
	}

	void SsaoShader::update(const int _w, const int _h)
	{
		deleteFBOs();

		m_fboGeometry = new QOpenGLFramebufferObject(_w, _h, QOpenGLFramebufferObject::Depth, GL_TEXTURE_2D, GL_RGB);
		m_fboGeometry->addColorAttachment(_w, _h, GL_RGB);
		m_fboGeometry->addColorAttachment(_w, _h, GL_RGB);
		m_fboGeometry->addColorAttachment(_w, _h, GL_RGBA);
		m_fboGeometry->addColorAttachment(_w, _h, GL_RED);
		QVector<GLuint> texIds = m_fboGeometry->textures();
		glBindTexture(GL_TEXTURE_2D, texIds[0]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, _w, _h, 0, GL_RGB, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_2D, texIds[1]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, _w, _h, 0, GL_RGB, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, texIds[2]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, _w, _h, 0, GL_RGB, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, texIds[3]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _w, _h, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, texIds[4]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _w, _h, 0, GL_RED, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		unsigned int attachments[5] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4 };
		glDrawBuffers(5, attachments);
		glBindTexture(GL_TEXTURE_2D, 0);

		m_fboNoise = new QOpenGLFramebufferObject(_w, _h, QOpenGLFramebufferObject::NoAttachment, GL_TEXTURE_2D, GL_RED);
		glBindTexture(GL_TEXTURE_2D, m_fboNoise->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, _w, _h, 0, GL_RED, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);

		m_fboBlur = new QOpenGLFramebufferObject(_w, _h, QOpenGLFramebufferObject::NoAttachment, GL_TEXTURE_2D, GL_RGB);
		glBindTexture(GL_TEXTURE_2D, m_fboBlur->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _w, _h, 0, GL_RGB, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);

		m_fboHalo = new QOpenGLFramebufferObject(_w, _h, QOpenGLFramebufferObject::Depth, GL_TEXTURE_2D, GL_RGB);
		glBindTexture(GL_TEXTURE_2D, m_fboHalo->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, _w, _h, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		m_fboSilhouette = new QOpenGLFramebufferObject(_w, _h, QOpenGLFramebufferObject::NoAttachment, GL_TEXTURE_2D, GL_RGB);
		glBindTexture(GL_TEXTURE_2D, m_fboSilhouette->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, _w, _h, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		m_fboSilhouetteBlurred = new QOpenGLFramebufferObject(_w, _h, QOpenGLFramebufferObject::NoAttachment, GL_TEXTURE_2D, GL_RGB);
		glBindTexture(GL_TEXTURE_2D, m_fboSilhouetteBlurred->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, _w, _h, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		m_fboLighting = new QOpenGLFramebufferObject(_w, _h, QOpenGLFramebufferObject::NoAttachment, GL_TEXTURE_2D, GL_RGBA);
		glBindTexture(GL_TEXTURE_2D, m_fboLighting->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, _w, _h, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glBindTexture(GL_TEXTURE_2D, m_noiseTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 4, 4, 0, GL_RGB, GL_FLOAT, &m_ssaoNoise[0]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	}
}

