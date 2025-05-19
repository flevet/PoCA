/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ImageInterface.hpp
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

#ifndef ImageInterface_h__
#define ImageInterface_h__

#include <vector>
#include <array>

#include <General/BasicComponent.hpp>
#include <General/Misc.h>

namespace poca::core {
	class ImageInterface : public BasicComponent {
	public:
		virtual ~ImageInterface() = default;

		virtual BasicComponentInterface* copy() = 0;

		virtual void finalizeImage(const uint32_t, const uint32_t, const uint32_t) = 0;
		virtual void addFeatureLabels() = 0;

		virtual void uint8_normalisedData(std::vector <unsigned char>&) const = 0;
		virtual void uint16_normalisedData(std::vector <uint16_t>&) const = 0;
		virtual void uint16_labeledData(std::vector <uint16_t>&) const = 0;
		virtual void float_normalisedData(std::vector <float>&) const = 0;

		virtual void save(const std::string&) const = 0;
		virtual const void* getImagePtr(const uint32_t) const = 0;

		virtual const uint32_t dimension() const { return (m_depth > 1) ? 3 : 2; }
		virtual inline uint32_t width() const { return m_width; }
		virtual inline uint32_t height() const { return m_height; }
		virtual inline uint32_t depth() const { return m_depth; }
		virtual inline uint32_t nbPixels() const { return m_width * m_height * m_depth; }

		virtual inline float min() { return m_min; }
		virtual inline float max() { return m_max; }
		virtual inline float maxValue() { return m_maxValue; }

		virtual inline ImageType type() { return m_type; }
		virtual void setType(const ImageType _type) { m_type = _type; }

		virtual inline ImageType typeImage() { return m_typeImage; }
		virtual void setTypeImage(const ImageType _type) { m_typeImage = _type; }
		virtual bool isLabelImage() const { return m_typeImage == LABEL; }
		virtual bool isRawImage() const { return m_typeImage == RAW; }

		virtual std::vector <float>& volumes() { return m_volumes; }
		virtual const std::vector <float>& volumes() const { return m_volumes; }

		virtual inline int currentFrame() { return m_currentFrame; }
		virtual void setCurrentFrame(const int _frame) { m_currentFrame = _frame; }

	protected:
		ImageInterface(const ImageType _typeImage) :BasicComponent("Image"), m_typeImage(_typeImage) {}

	protected:
		uint32_t m_width, m_height, m_depth;
		float m_min, m_max, m_maxValue;
		ImageType m_type, m_typeImage;
		int m_currentFrame{ -1 };

		//Currently just store volumes for labels data, TODO think better about the structure
		std::vector <float> m_volumes;
	};
}

#endif

