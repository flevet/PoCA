/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      TextDisplayer.hpp
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

#ifndef TextDisplayer_h__
#define TextDisplayer_h__

#include <stdint.h>
#include <glm/glm.hpp>

#include <General/Vec2.hpp>

#include "Shader.hpp"
#include "../fontstash/fontstash.h"

namespace poca::opengl {
	class TextDisplayer {
	public:
		TextDisplayer();
		~TextDisplayer();

		poca::core::Vec2mf renderText(const glm::mat4&, const char*, const uint8_t, const uint8_t, const uint8_t, const uint8_t, const float, const float, const float, const int = FONS_ALIGN_LEFT | FONS_ALIGN_TOP);
		const float widthOfStr(const char *, const float, const float);
		const float lineHeight() const;

		inline const float getFontSize() const { return m_fontSize; }
		inline void setFontSize(const float _val) { m_fontSize = _val; }
		inline void setClipPlane(const glm::vec4& _val) { m_clipPlane = _val; }

	private:
		Shader* m_shader;
		FONScontext* m_fs = NULL;
		uint8_t* m_fontDataDroidSans;
		int m_fontSdfEffects;
		float m_fontSize;
		glm::vec4 m_clipPlane;
	};
}
#endif

