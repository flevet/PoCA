/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PaletteInterface.hpp
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

#ifndef PaletteInterface_h__
#define PaletteInterface_h__

#include <vector>
#include "../General/Vec4.hpp"

namespace poca::core {

	class PaletteInterface {
	public:
		virtual ~PaletteInterface() = default;

		virtual void setColor(float, Color4uc) = 0;
		virtual void removeColorAt(unsigned int) = 0;
		virtual Color4uc colorAt(unsigned int) const = 0;
		virtual void setColorAt(unsigned int, Color4uc) = 0;
		virtual float colorPosition(unsigned int) const = 0;
		virtual void setColorPosition(unsigned int, float) = 0;
		virtual const Color4uc getColor(const float) const = 0;
		virtual const Color4uc getColorLUT(const float) const = 0;
		virtual const Color4uc getColorNoInterpolation(const float) const = 0;

		virtual void setFilterMinMax(const float, const float) = 0;
		virtual const float getFilterMin() const = 0;
		virtual const float getFilterMax() const = 0;

		virtual void setHiLow(const bool) = 0;
		virtual const bool isHiLow() const = 0;

		virtual void setThreshold(const bool) = 0;
		virtual const bool isThreshold() const = 0;

		virtual const std::string& getName() const = 0;
		virtual void setName(const std::string&) = 0;

		virtual const size_t size() const = 0;

		virtual PaletteInterface* copy() const = 0;
		virtual void setPalette(PaletteInterface*) = 0;
	};
}

#endif

