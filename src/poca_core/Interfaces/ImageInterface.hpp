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
#include <cstddef>
#include <functional>
#include <cstdint>
#include <string>
#include <stdexcept>

#include <General/BasicComponent.hpp>
#include <General/Misc.h>
#include <General/Region3D.hpp>

namespace poca::core {
	struct ImageSpatialMetadata {
		std::array<double, 3> spacing{ 1.0, 1.0, 1.0 }; // x, y, z
		std::array<double, 3> origin{ 0.0, 0.0, 0.0 };  // x, y, z
		std::array<std::string, 3> units{ "", "", "" };
	};

	struct ImagePyramidLevelInfo {
		uint32_t width{ 0 }, height{ 0 }, depth{ 0 };
		bool available{ true }; // An unavailable entry retains its level index for generated fallback.
		// Optional actual stored-level calibration; never inferred from a level index.
		ImageSpatialMetadata spatial;
		bool hasSpatialMetadata{ false };
	};

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
		virtual bool hasPixels() const = 0;
		virtual bool canReloadPixels() const = 0;
		virtual void releasePixels() = 0;
		virtual bool canReadFullResolutionPlane() const = 0;
		virtual bool canReadFullResolutionRegion() const = 0;
		virtual bool readFullResolutionPlane(const uint64_t, void*, const std::size_t) const = 0;
		virtual bool readFullResolutionRegion(const Region3D&, void*, const std::size_t) const = 0;

		virtual bool outOfCoreEnabled() const { return m_outOfCoreEnabled; }
		virtual void setOutOfCoreEnabled(const bool _enabled) { m_outOfCoreEnabled = _enabled; }
		virtual bool pyramidalRenderingEnabled() const { return m_pyramidalRenderingEnabled; }
		virtual void setPyramidalRenderingEnabled(const bool _enabled) { m_pyramidalRenderingEnabled = _enabled; }
		virtual void invalidatePyramidCache() const {}
		virtual std::size_t pyramidCacheBytes() const { return 0; }

		// Core metadata only: does not affect display scale or bounding boxes.
		const ImageSpatialMetadata& spatialMetadata() const { return m_spatialMetadata; }
		void setSpatialMetadata(const ImageSpatialMetadata& _metadata) { m_spatialMetadata = _metadata; }

		// Configure before rendering. Level 0 describes the full-resolution image;
		// subsequent entries describe progressively coarser stored levels.
		virtual bool hasNativePyramid() const { return !m_nativePyramidLevels.empty(); }
		virtual std::size_t nativePyramidLevelCount() const { return m_nativePyramidLevels.size(); }
		virtual bool nativePyramidLevelInfo(const std::size_t _level, ImagePyramidLevelInfo& _info) const {
			if (_level >= m_nativePyramidLevels.size() || !m_nativePyramidLevels[_level].available)
				return false;
			_info = m_nativePyramidLevels[_level];
			return true;
		}
		virtual void setNativePyramidLevels(const std::vector<ImagePyramidLevelInfo>& _levels) {
			for (const auto& info : _levels)
				if (info.width == 0 || info.height == 0 || info.depth == 0)
					throw std::invalid_argument("Native pyramid dimensions must be positive");
			m_nativePyramidLevels = _levels;
			invalidatePyramidCache();
		}
		virtual bool canReadNativePyramid() const { return false; }
		virtual bool canReadNativePyramidRegion() const { return false; }
		virtual bool readNativePyramidRegion(uint32_t, const Region3D&, void*, std::size_t) const { return false; }

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
		ImageInterface(const ImageType _typeImage) :BasicComponent("Image", _typeImage == poca::core::RAW ? "LightGrayscale" : "HotCold2"), m_typeImage(_typeImage)
		{
		}

	protected:
		uint32_t m_width{ 0 }, m_height{ 0 }, m_depth{ 0 };
		float m_min{ 0.f }, m_max{ 0.f }, m_maxValue{ 0.f };
		ImageType m_type{ NONE }, m_typeImage{ RAW };
		int m_currentFrame{ -1 };
		bool m_outOfCoreEnabled{ false };
		bool m_pyramidalRenderingEnabled{ false };
		ImageSpatialMetadata m_spatialMetadata;
		std::vector<ImagePyramidLevelInfo> m_nativePyramidLevels;

		//Currently just store volumes for labels data, TODO think better about the structure
		std::vector <float> m_volumes;
	};
}

#endif

