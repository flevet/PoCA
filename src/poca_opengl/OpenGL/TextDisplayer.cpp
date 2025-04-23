/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      TextDisplayer.cpp
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
#include <GL/glu.h>
#include <GL/gl.h>
#include <stdio.h>
#include <string.h>

#include "TextDisplayer.hpp"
#include "GLBuffer.hpp"

#define FONTSTASH_IMPLEMENTATION
#include "../fontstash/fontstash.h"
//#define GLFONTSTASH_IMPLEMENTATION
#define GLFONTSTASH_IMPLEMENTATION_ES2
#include "../fontstash/gl3corefontstash.h"

void fontStashResetAtlas(FONScontext* stash, int width, int height) {
	fonsResetAtlas(stash, width, height);
	std::cout << "reset atlas to " << width << " x " << height << std::endl;
}

void fontStashEexpandAtlas(FONScontext* stash) {
	int w = 0, h = 0;
	GLint value;
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &value);
	const int maxTexturesize = value;

	fonsGetAtlasSize(stash, &w, &h);
	if (w < h) {
		w *= 2;
	}
	else {
		h *= 2;
	}

	if (w > maxTexturesize || h > maxTexturesize) {
		fontStashResetAtlas(stash, maxTexturesize, maxTexturesize);
	}
	else {
		fonsExpandAtlas(stash, w, h);
		std::cout << "expanded atlas to " << w << " x " << h << std::endl;
	}
}

void fontStashError(void* userPointer, int error, int value) {
	FONScontext* stash = (FONScontext*)userPointer;
	switch (error) {
	case FONS_ATLAS_FULL:
		std::cout << "Font atlas full." << std::endl;
		fontStashEexpandAtlas(stash);
		break;
	case FONS_SCRATCH_FULL:
		std::cout << "Font error: scratch full, tried to allocate " << value << std::endl;
		break;
	case FONS_STATES_OVERFLOW:
		std::cout << "Font error: states overflow." << std::endl;
		break;
	case FONS_STATES_UNDERFLOW:
		std::cout << "Font error: states underflow." << std::endl;
		break;
	default:
		std::cout << "Font error: unknown." << std::endl;
		break;
	}
}

namespace poca::opengl {

	char* vsText = "#version 330 core\n"
		"uniform mat4 modelView; \n"
		"uniform mat4 projection;\n"
		"layout(location = 0) in vec4 vertexPosition;\n"
		"layout(location = 1) in vec2 vertexTexCoord;\n"
		"layout(location = 2) in vec4 vertexColor;\n"
		"out vec2 interpolatedTexCoord;\n"
		"out vec4 interpolatedColor;\n"
		"void main() {\n"
		"	interpolatedColor = vertexColor;\n"
		"	interpolatedTexCoord = vertexTexCoord;\n"
		"	gl_Position = projection * modelView * vertexPosition;\n"
		"}";

	const char* fsText = "#version 330 core\n"
		"uniform sampler2D diffuse;\n"
		"in vec2 interpolatedTexCoord;\n"
		"in vec4 interpolatedColor;\n"
		"out vec4 color;\n"
		"void main() {\n"
		"  float alpha = texture2D(diffuse, interpolatedTexCoord).a;\n"
		"  vec4 textColor = clamp(interpolatedColor, 0.0, 1.0);\n"
		"  color = vec4(textColor.rgb * textColor.a, textColor.a) * alpha;\n" // Using premultiplied alpha.
		"}\n";

	TextDisplayer::TextDisplayer() :m_fontDataDroidSans(NULL), m_fontSize(20.f)
	{
		m_shader = new Shader();
		m_shader->createAndLinkProgramFromStr(vsText, fsText);

		m_clipPlane = glm::vec4(0, 0, 0, 0);

		//
	   // Initialize fontstash.
	   //
		m_fs = glfonsCreate(512, 512, FONS_ZERO_TOPLEFT);
		if (m_fs == NULL) {
			std::cout << "Could not create font stash." << std::endl;
			return;
		}

		fonsSetErrorCallback(m_fs, fontStashError, m_fs);

		//
		// Load font data.
		//

		const char* droidSansFilename = "fonts/droid/DroidSans.ttf";
		m_fontDataDroidSans = NULL;

		FONSsdfSettings noSdf = { 0 };
		noSdf.sdfEnabled = 0;

		m_fontSdfEffects = fonsAddFontSdf(m_fs, "DroidSansSdfEffects", droidSansFilename, noSdf);
		if (m_fontSdfEffects == FONS_INVALID) {
			std::cout << "Could not add SDF font." << std::endl;
			return;
		}
	}

	TextDisplayer::~TextDisplayer()
	{
	}

	poca::core::Vec2mf TextDisplayer::renderText(const glm::mat4& _proj, const char* _text, 
		const uint8_t _r, const uint8_t _g, const uint8_t _b, const uint8_t _a, 
		const float _x, const float _y, const float _t, const int _align)
	{
		m_shader->use();
		m_shader->setMat4("projection", _proj);
		m_shader->setMat4("modelView", glm::mat4(1.f));
		m_shader->setFloat("time", _t);
		m_shader->setVec4("clipPlane", m_clipPlane);

		float x, lineHeight;
		fonsClearState(m_fs);
		fonsSetFont(m_fs, m_fontSdfEffects);
		fonsSetSize(m_fs, m_fontSize);
		fonsSetAlign(m_fs, _align);
		fonsVertMetrics(m_fs, NULL, NULL, &lineHeight);

		fonsSetColor(m_fs, glfonsRGBA(_r, _g, _b, _a));
		x = fonsDrawText(m_fs, _x, _y, _text, NULL);

		m_shader->release();
		GL_CHECK_ERRORS();

		return poca::core::Vec2mf(x, lineHeight);
	}

	const float TextDisplayer::widthOfStr(const char* _str, const float _x, const float _y)
	{
		return fonsTextBounds(m_fs, _x, _y, _str, NULL, NULL);
	}

	const float TextDisplayer::lineHeight() const {
		float val;
		fonsVertMetrics(m_fs, NULL, NULL, &val);
		return val;
	}
}

