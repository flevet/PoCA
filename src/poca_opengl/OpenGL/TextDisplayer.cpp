/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      TextDisplayer.cpp
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
#define GLFONTSTASH_IMPLEMENTATION
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

	char* vsText = "uniform mat4 modelView;\n"
		"uniform mat4 projection;\n"
		"uniform vec4 clipPlane;\n"
		"attribute vec4 vertexPosition;\n"
		"attribute vec2 vertexTexCoord;\n"
		"attribute vec4 vertexColor;\n"
		"varying vec2 interpolatedTexCoord;\n"
		"varying vec4 interpolatedColor;\n"
		"void main() {\n"
		"	interpolatedColor = vertexColor;\n"
		"	interpolatedTexCoord = vertexTexCoord;\n"
		"	vec4 pos = projection * modelView * vertexPosition;\n"
		"	gl_Position = pos;\n"
		"	gl_ClipDistance[0] = dot(pos, clipPlane);\n"
		"}";

	char* fsText = "uniform sampler2D sdf;\n"
		"uniform float time;\n"
		"varying vec2 interpolatedTexCoord;\n"
		"varying vec4 interpolatedColor;\n"
		"const float glyphEdge = 0.5;\n"
		"//\n"
		"// Some effects. Enable/disable by commenting out the defines.\n"
		"//\n"
		"//#define SUPERSAMPLE\n"
		"//#define OUTLINE\n"
		"const float outlineEdgeWidth = 0.04;\n"
		"const vec4 outlineColor = vec4(0.0, 0.0, 0.0, 1.0);\n"
		"//#define SHADOW\n"
		"const float shadowBlur = 0.5;\n"
		"const vec4 shadowColor = vec4(0.0, 0.0, 0.0, 0.5);\n"
		"//#define GROW_ANIMATION\n"
		"//#define GLOW_ANIMATION\n"
		"float contour(float dist, float edge, float width) {\n"
		"    return clamp(smoothstep(edge - width, edge + width, dist), 0.0, 1.0);\n"
		"}\n"
		"float getSample(vec2 texCoords, float edge, float width) {\n"
		"    return contour(texture2D(sdf, texCoords).a, edge, width);\n"
		"}\n"
		"void main() {\n"
		"    vec4 tex = texture2D(sdf, interpolatedTexCoord);\n"
		"    float dist = tex.a;\n"
		"    float width = fwidth(dist);\n"
		"    vec4 textColor = clamp(interpolatedColor, 0.0, 1.0);\n"
		"    float outerEdge = glyphEdge;\n"
		"#if defined(GROW_ANIMATION)\n"
		"    outerEdge -= (sin(time * 3.3) + 0.8) * 0.1;\n"
		"#endif\n"
		"#if defined(SUPERSAMPLE)\n"
		"    vec2 uv = interpolatedTexCoord.xy;\n"
		"    float alpha = contour(dist, outerEdge, width);\n"
		"    float dscale = 0.354; // half of 1/sqrt2; you can play with this\n"
		"    vec2 duv = dscale * (dFdx(uv) + dFdy(uv));\n"
		"    vec4 box = vec4(uv - duv, uv + duv);\n"
		"    float asum = getSample(box.xy, outerEdge, width)\n"
		"        + getSample(box.zw, outerEdge, width)\n"
		"        + getSample(box.xw, outerEdge, width)\n"
		"        + getSample(box.zy, outerEdge, width);\n"
		"    // weighted average, with 4 extra points having 0.5 weight each,\n"
		"    // so 1 + 0.5*4 = 3 is the divisor\n"
		"    alpha = (alpha + 0.5 * asum) / 3.0;\n"
		"#else\n"
		"    //float alpha = clamp(smoothstep(outerEdge - width, outerEdge + width, dist), 0.0, 1.0);\n"
		"    float alpha = contour(dist, outerEdge, width);\n"
		"#endif\n"
		"    // Basic simple SDF text without effects. Normal blending.\n"
		"    gl_FragColor = vec4(textColor.rgb, textColor.a * alpha);\n"
		"#if defined(OUTLINE)\n"
		"    outerEdge = outerEdge - outlineEdgeWidth;\n"
		"#if defined(GROW_ANIMATION)\n"
		"    outerEdge -= (sin(time * 10.3) + 0.8) * 0.05;\n"
		"#endif\n"
		"    float outlineOuterAlpha = clamp(smoothstep(outerEdge - width, outerEdge + width, dist), 0.0, 1.0);\n"
		"    float outlineAlpha = outlineOuterAlpha - alpha;\n"
		"    gl_FragColor.rgb = mix(outlineColor.rgb, gl_FragColor.rgb, alpha);\n"
		"    gl_FragColor.a = max(gl_FragColor.a, outlineColor.a * outlineOuterAlpha);\n"
		"#endif\n"
		"#if defined(SHADOW)\n"
		"    float shadowAlpha = clamp(smoothstep(max(outerEdge - shadowBlur, 0.05), outerEdge + shadowBlur, dist), 0.0, 1.0);\n"
		"    vec4 shadow = shadowColor * shadowAlpha;\n"
		"    gl_FragColor = shadow * (1.0 - gl_FragColor.a) + gl_FragColor * gl_FragColor.a;\n"
		"#else\n"
		"    // Premultiplied alpha output.\n"
		"    gl_FragColor.rgb *= gl_FragColor.a;\n"
		"#endif\n"
		"#if defined(GLOW_ANIMATION)\n"
		"    float glowIntensityAnim = (sin(time * 4.3) + 2.0) * 0.25;\n"
		"    float glowArea = clamp(smoothstep(glyphEdge - 0.25, glyphEdge + 0.0, dist), 0.0, 1.0);\n"
		"    vec4 glow = vec4(vec3(glowIntensityAnim), 0.0) * glowArea;\n"
		"    gl_FragColor = glow * (1.0 - gl_FragColor.a) + gl_FragColor * gl_FragColor.a;\n"
		"#endif\n"
		"}";

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

