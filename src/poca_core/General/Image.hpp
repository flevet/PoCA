/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Image.hpp
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

#ifndef Image_hpp__
#define Image_hpp__

#include <execution>
#include <unordered_map>     // [PYRAMID]
#include <cstdint>           // [PYRAMID]
#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <mutex>
#include <vector>
#include <limits>
#include <stdexcept>
#include <type_traits> 

#include <Interfaces/ImageInterface.hpp>
#include <General/ImagePyramidValidation.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <General/MyData.hpp>
#include <General/ArrayStatistics.hpp>
#include <General/Misc.h>
#include <Cuda/CoreMisc.h>
#include <General/Histogram.hpp>
#include <General/Engine.hpp>

namespace poca::core {

	template <class T>
	class Image : public ImageInterface {
	public:
		Image(const ImageType = RAW);
		Image(const Image&);
		~Image();

		BasicComponentInterface* copy();

		void finalizeImage(const uint32_t, const uint32_t, const uint32_t);
		// Metadata-only RAW image; _sample contains a coarse native level, never level zero.
		void initializeStorageBacked(const uint32_t, const uint32_t, const uint32_t,
			const std::vector<T>&, bool = false, float = 0.f, float = 0.f);
		void addFeatureLabels();

		void uint8_normalisedData(std::vector <unsigned char>&) const;
		void uint16_normalisedData(std::vector <uint16_t>&) const;
		void uint16_labeledData(std::vector <uint16_t>&) const;
		void float_normalisedData(std::vector <float>&) const;

		const T* getImage(const uint32_t) const;

		inline const T* data() const;
		inline T* data();
		inline const std::vector<T>& pixels() const;
		inline std::vector<T>& pixels();
		bool hasPixels() const override;
		bool canReloadPixels() const override;
		void releasePixels() override;
		bool canReadFullResolutionPlane() const override;
		bool canReadFullResolutionRegion() const override;
		bool readFullResolutionPlane(const uint64_t, void*, const std::size_t) const override;
		bool readFullResolutionRegion(const Region3D&, void*, const std::size_t) const override;
		void setPixelReloadCallback(std::function<void(std::vector<T>&)>);
		void setPlaneReaderCallback(std::function<bool(uint64_t, void*, std::size_t)>);
		void setRegionReaderCallback(std::function<bool(const Region3D&, void*, std::size_t)>);
		// Native values are contiguous, with x fastest, then y, then z.
		using NativePyramidReader = std::function<bool(uint32_t, std::vector<T>&)>;
		void setNativePyramidLevelReaderCallback(NativePyramidReader);
		using NativePyramidRegionReader = std::function<bool(uint32_t, const Region3D&, void*, std::size_t)>;
		void setNativePyramidRegionReaderCallback(NativePyramidRegionReader);
		bool canReadNativePyramidRegion() const override;
		bool readNativePyramidRegion(uint32_t, const Region3D&, void*, std::size_t) const override;
		void setNativePyramidLevels(const std::vector<ImagePyramidLevelInfo>&) override;
		bool canReadNativePyramid() const override;
		bool hasNativePyramid() const override {
			std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
			return ImageInterface::hasNativePyramid();
		}
		std::size_t nativePyramidLevelCount() const override {
			std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
			return ImageInterface::nativePyramidLevelCount();
		}
		bool nativePyramidLevelInfo(std::size_t _level, ImagePyramidLevelInfo& _info) const override {
			std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
			return ImageInterface::nativePyramidLevelInfo(_level, _info);
		}

		void save(const std::string&) const;

		const void* getImagePtr(const uint32_t) const;

		// ============================
		// [PYRAMID] Pyramidal cache API
		// ============================

		enum class DownsampleMode : uint8_t {
			Average,     // intensity
			Nearest,     // labels / fast
			MIP,         // maximum intensity / maximum ID in block
			Majority     // labels (slow but better IDs)
		};

		struct PyramidLevelView {
			uint32_t w = 0, h = 0, d = 0;
			const T* ptr = nullptr;
			std::size_t count = 0;
			const char* source = "generated";
			// Positive levels retain immutable storage across cache invalidation.
			// Level zero is borrowed; rendering uses copyFullResolutionLevel().
			std::shared_ptr<const void> owner;
		};

		// factors are applied per level (typically 2,2,2)
		PyramidLevelView getOrCreatePyramidLevel(
			int level,
			uint32_t fx = 2, uint32_t fy = 2, uint32_t fz = 2,
			DownsampleMode mode = DownsampleMode::Average
		) const;

		// Explicit native access for export, independent of display enablement/LOD.
		PyramidLevelView nativePyramidLevelView(uint32_t) const;

		// Export uses the same downsampler on an aligned, bounded parent region.
		// This operation is independent of rendering flags and does not cache pixels.
		static PyramidLevelView downsampleRegion(const T*, uint32_t, uint32_t, uint32_t);

		// A full-resolution snapshot uses Histogram's existing protected copy.
		PyramidLevelView copyFullResolutionLevel(std::vector<T>&) const;

		// Call this if the underlying pixels change (optional for now)
		void invalidatePyramidCache() const override;
		std::size_t pyramidCacheBytes() const override;

		PyramidLevelView getOrCreateDownsampled(
			uint32_t fx, uint32_t fy, uint32_t fz,
			DownsampleMode mode
		) const;

	private:
		// [PYRAMID] internal cache container
		struct PyramidKey {
			int level;
			uint32_t fx, fy, fz;
			uint8_t mode;

			bool operator==(const PyramidKey& o) const {
				return level == o.level && fx == o.fx && fy == o.fy && fz == o.fz && mode == o.mode;
			}
		};

		struct PyramidKeyHash {
			size_t operator()(const PyramidKey& k) const noexcept {
				size_t h = std::hash<int>{}(k.level);
				h ^= (std::hash<uint32_t>{}(k.fx) + 0x9e3779b9 + (h << 6) + (h >> 2));
				h ^= (std::hash<uint32_t>{}(k.fy) + 0x9e3779b9 + (h << 6) + (h >> 2));
				h ^= (std::hash<uint32_t>{}(k.fz) + 0x9e3779b9 + (h << 6) + (h >> 2));
				h ^= (std::hash<uint8_t>{}(k.mode) + 0x9e3779b9 + (h << 6) + (h >> 2));
				return h;
			}
		};

		struct PyramidLevel {
			uint32_t w = 0, h = 0, d = 0;
			std::vector<T> data;
		};

		// mutable: cache is logically const
		mutable std::unordered_map<PyramidKey, std::shared_ptr<const PyramidLevel>, PyramidKeyHash> m_pyramid; // [PYRAMID]
		mutable std::unordered_map<int, std::shared_ptr<const PyramidLevel>> m_nativePyramidCache;
		mutable std::recursive_mutex m_pyramidMutex;
		mutable uint64_t m_pyramidRevision{ 0 };

		PyramidLevelView makePyramidLevelView(const std::shared_ptr<const PyramidLevel>&, int, const char*) const;
		PyramidLevelView fullResolutionLevelView() const;
		PyramidLevelView getOrCreateNativePyramidLevel(int, ImagePyramidLevelInfo) const;
		PyramidLevelView getOrCreateGeneratedPyramidLevel(int, uint32_t, uint32_t, uint32_t, DownsampleMode) const;

		// [PYRAMID] downsample core
		static PyramidLevel downsampleLevel(
			const PyramidLevel& src,
			uint32_t fx, uint32_t fy, uint32_t fz,
			DownsampleMode mode
		);

		static PyramidLevel downsampleRaw(
			const T* srcData,
			uint32_t srcW, uint32_t srcH, uint32_t srcD,
			uint32_t fx, uint32_t fy, uint32_t fz,
			DownsampleMode mode
		);

	private:
		std::function<bool(uint64_t, void*, std::size_t)> m_planeReaderCallback;
		std::function<bool(const Region3D&, void*, std::size_t)> m_regionReaderCallback;
		NativePyramidReader m_nativePyramidReaderCallback;
		NativePyramidRegionReader m_nativePyramidRegionReaderCallback;
	};

	//maxValue is used for shaders. uint8_t & uint16_t textures are normalized so need to know the maxValue to find back the pixel value
	//float texture are not normalized, so we set maxValue at 1 to keep pixel value unchanged in shader
	template <class T>
	Image<T>::Image(const ImageType _typeImage) :ImageInterface(_typeImage)
	{
		if constexpr (std::is_same_v<T, uint8_t>) m_type = UINT8;
		else if constexpr (std::is_same_v<T, uint16_t>) m_type = UINT16;
		else if constexpr (std::is_same_v<T, uint32_t>) m_type = UINT32;
		else if constexpr (std::is_same_v<T, int32_t>) m_type = INT32;
		else if constexpr (std::is_same_v<T, float>) m_type = FLOAT;
		else throw std::invalid_argument("Unsupported image scalar type");
		m_data.insert(std::make_pair("intensity", new poca::core::MyData(new poca::core::Histogram<T>(), false)));
		std::string type = typeid(T).name();
		m_maxValue = (type == "float" || type == "unsigned int" || type == "int") ? 1 : std::numeric_limits<T>::max();
	}

	template <class T>
	Image<T>::Image(const Image& _o) : ImageInterface(_o)
	{
		std::lock_guard<std::recursive_mutex> lock(_o.m_pyramidMutex);
		m_pyramid = _o.m_pyramid;
		m_nativePyramidCache = _o.m_nativePyramidCache;
		m_nativePyramidReaderCallback = _o.m_nativePyramidReaderCallback;
		m_nativePyramidRegionReaderCallback = _o.m_nativePyramidRegionReaderCallback;
		m_outOfCoreEnabled = _o.m_outOfCoreEnabled;
		m_pyramidalRenderingEnabled = _o.m_pyramidalRenderingEnabled;
		m_planeReaderCallback = _o.m_planeReaderCallback;
		m_regionReaderCallback = _o.m_regionReaderCallback;
	}

	template <class T>
	Image<T>::~Image()
	{
	}

	template <class T>
	BasicComponentInterface* Image<T>::copy()
	{
		return new Image(*this);
	}

	template <class T>
	void Image<T>::initializeStorageBacked(const uint32_t _w, const uint32_t _h, const uint32_t _d,
		const std::vector<T>& _sample, bool _hasDisplayBounds, float _displayMin, float _displayMax)
	{
		std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
		if ((!isRawImage() && !isLabelImage()) || (isLabelImage() && !std::is_integral_v<T>) || m_width != 0 || !canReloadPixels())
			throw std::invalid_argument("Storage initialization requires a new scalar RAW or integer LABEL image with a reload callback");
		const std::size_t count = checkedPyramidElementCount(0, "storage/initialization", _w, _h, _d);
		checkedPyramidByteCount(0, "storage/initialization", _w, _h, _d, sizeof(T));
		auto* histogram = dynamic_cast<Histogram<T>*>(getOriginalHistogram("intensity"));
		histogram->initializeStorageBacked(count, _sample, _hasDisplayBounds, _displayMin, _displayMax);
		m_width = _w; m_height = _h; m_depth = _d;
		m_bbox.set(0, 0, 0, _w, _h, _d);
		m_selection.clear();
		setCurrentHistogramType("intensity");
		m_min = histogram->getMin(); m_max = histogram->getMax();
		m_outOfCoreEnabled = m_pyramidalRenderingEnabled = true;
		invalidatePyramidCache();
	}

	template <class T>
	void Image<T>::finalizeImage(const uint32_t _w, const uint32_t _h, const uint32_t _d)
	{
		poca::core::Engine* engine = poca::core::Engine::instance();
		
		m_width = _w; m_height = _h; m_depth = _d;
		const std::vector<T>& pixels = this->pixels();
		clock_t t1 = clock(), t2, t3 = clock(), t4;
		t4 = clock();
		long elapsed = ((double)t4 - t3) / CLOCKS_PER_SEC * 1000;
		if (engine->verbose())
			std::cout << "Time for finding max & min " << elapsed << std::endl;
		t3 = clock();
		m_bbox.set(0, 0, 0, m_width, m_height, m_depth);
		m_data["intensity"]->finalizeData();// = new poca::core::MyData(m_pixels, false);
		t4 = clock();
		elapsed = ((double)t4 - t3) / CLOCKS_PER_SEC * 1000;
		if (engine->verbose())
			std::cout << "Time for creating my data " << elapsed << std::endl;
		t3 = clock();
		m_selection.clear();// .resize(pixels.size(), true);
		setCurrentHistogramType("intensity");
		m_min = getCurrentHistogram()->getMin();
		m_max = getCurrentHistogram()->getMax();
		t4 = clock();
		elapsed = ((double)t4 - t3) / CLOCKS_PER_SEC * 1000;
		if (engine->verbose())
			std::cout << "Time forsetting the histogram " << elapsed << ", min = " << m_min << ", max = " << m_max << std::endl;
		t3 = clock();
		if (engine->verbose())
			std::cout << "Bounding box image " << m_bbox << std::endl;
		t2 = clock();
		elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		if (engine->verbose())
			std::cout << "Time for creating image " << elapsed << std::endl;
		if(m_typeImage == poca::core::LABEL)
			addFeatureLabels();

		// [PYRAMID] pixels are now finalized, clear cache
		invalidatePyramidCache();
	}

	// =====================================
	// [PYRAMID] Implementation of cache API
	// =====================================

	template <class T>
	void Image<T>::invalidatePyramidCache() const
	{
		std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
		++m_pyramidRevision;
		m_pyramid.clear();
		m_nativePyramidCache.clear();
	}

	template <class T>
	std::size_t Image<T>::pyramidCacheBytes() const
	{
		std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
		std::size_t bytes = 0;
		for (const auto& kv : m_pyramid)
			bytes += kv.second->data.size() * sizeof(T);
		for (const auto& kv : m_nativePyramidCache)
			bytes += kv.second->data.size() * sizeof(T);
		return bytes;
	}

	template <class T>
	typename Image<T>::PyramidLevel Image<T>::downsampleRaw(
		const T* srcData,
		uint32_t srcW, uint32_t srcH, uint32_t srcD,
		uint32_t fx, uint32_t fy, uint32_t fz,
		DownsampleMode mode
	)
	{
		PyramidLevel src;
		src.w = srcW;
		src.h = srcH;
		src.d = srcD;

		PyramidLevel dst;
		dst.w = std::max(1u, src.w / fx);
		dst.h = std::max(1u, src.h / fy);
		dst.d = std::max(1u, src.d / fz);
		dst.data.resize(checkedPyramidElementCount(0, "generated/allocation", dst.w, dst.h, dst.d));

		auto idxSrc = [&](uint32_t x, uint32_t y, uint32_t z) -> size_t {
			return (size_t(z) * src.h + y) * src.w + x;
		};
		auto idxDst = [&](uint32_t x, uint32_t y, uint32_t z) -> size_t {
			return (size_t(z) * dst.h + y) * dst.w + x;
		};

		for (uint32_t z = 0; z < dst.d; ++z) {
			for (uint32_t y = 0; y < dst.h; ++y) {
				for (uint32_t x = 0; x < dst.w; ++x) {
					const uint32_t sx0 = x * fx;
					const uint32_t sy0 = y * fy;
					const uint32_t sz0 = z * fz;

					if (mode == DownsampleMode::Nearest) {
						const uint32_t sx = std::min(sx0, src.w - 1);
						const uint32_t sy = std::min(sy0, src.h - 1);
						const uint32_t sz = std::min(sz0, src.d - 1);
						dst.data[idxDst(x, y, z)] = srcData[idxSrc(sx, sy, sz)];
						continue;
					}

					if (mode == DownsampleMode::MIP) {
						T maxVal = srcData[idxSrc(std::min(sx0, src.w - 1), std::min(sy0, src.h - 1), std::min(sz0, src.d - 1))];
						for (uint32_t dz = 0; dz < fz; ++dz) {
							uint32_t sz = std::min(sz0 + dz, src.d - 1);
							for (uint32_t dy = 0; dy < fy; ++dy) {
								uint32_t sy = std::min(sy0 + dy, src.h - 1);
								for (uint32_t dx = 0; dx < fx; ++dx) {
									uint32_t sx = std::min(sx0 + dx, src.w - 1);
									maxVal = std::max(maxVal, srcData[idxSrc(sx, sy, sz)]);
								}
							}
						}
						dst.data[idxDst(x, y, z)] = maxVal;
						continue;
					}

					if (mode == DownsampleMode::Majority) {
						if constexpr (!std::is_integral_v<T>) {
							const uint32_t sx = std::min(sx0, src.w - 1);
							const uint32_t sy = std::min(sy0, src.h - 1);
							const uint32_t sz = std::min(sz0, src.d - 1);
							dst.data[idxDst(x, y, z)] = srcData[idxSrc(sx, sy, sz)];
						}
						else {
							T bestVal = 0;
							int bestCount = -1;
							std::vector<T> vals;
							vals.reserve(size_t(fx) * fy * fz);
							for (uint32_t dz = 0; dz < fz; ++dz) {
								uint32_t sz = std::min(sz0 + dz, src.d - 1);
								for (uint32_t dy = 0; dy < fy; ++dy) {
									uint32_t sy = std::min(sy0 + dy, src.h - 1);
									for (uint32_t dx = 0; dx < fx; ++dx) {
										uint32_t sx = std::min(sx0 + dx, src.w - 1);
										vals.push_back(srcData[idxSrc(sx, sy, sz)]);
									}
								}
							}
							for (size_t i = 0; i < vals.size(); ++i) {
								int count = 0;
								for (size_t j = 0; j < vals.size(); ++j)
									count += (vals[j] == vals[i]) ? 1 : 0;
								if (count > bestCount) {
									bestCount = count;
									bestVal = vals[i];
								}
							}
							dst.data[idxDst(x, y, z)] = bestVal;
						}
						continue;
					}

					if constexpr (std::is_integral_v<T>) {
						using Sum = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
						Sum sum = 0;
						uint64_t cnt = 0;
						for (uint32_t dz = 0; dz < fz; ++dz) {
							uint32_t sz = std::min(sz0 + dz, src.d - 1);
							for (uint32_t dy = 0; dy < fy; ++dy) {
								uint32_t sy = std::min(sy0 + dy, src.h - 1);
								for (uint32_t dx = 0; dx < fx; ++dx) {
									uint32_t sx = std::min(sx0 + dx, src.w - 1);
									sum += static_cast<Sum>(srcData[idxSrc(sx, sy, sz)]);
									++cnt;
								}
							}
						}
						dst.data[idxDst(x, y, z)] = T(sum / static_cast<Sum>(std::max<uint64_t>(1, cnt)));
					}
					else {
						double sum = 0.0;
						double cnt = 0.0;
						for (uint32_t dz = 0; dz < fz; ++dz) {
							uint32_t sz = std::min(sz0 + dz, src.d - 1);
							for (uint32_t dy = 0; dy < fy; ++dy) {
								uint32_t sy = std::min(sy0 + dy, src.h - 1);
								for (uint32_t dx = 0; dx < fx; ++dx) {
									uint32_t sx = std::min(sx0 + dx, src.w - 1);
									sum += double(srcData[idxSrc(sx, sy, sz)]);
									cnt += 1.0;
								}
							}
						}
						dst.data[idxDst(x, y, z)] = T(sum / std::max(1.0, cnt));
					}
				}
			}
		}
		return dst;
	}

	template <class T>
	typename Image<T>::PyramidLevel Image<T>::downsampleLevel(
		const PyramidLevel& src,
		uint32_t fx, uint32_t fy, uint32_t fz,
		DownsampleMode mode
	)
	{
		PyramidLevel dst;
		dst.w = std::max(1u, src.w / fx);
		dst.h = std::max(1u, src.h / fy);
		dst.d = std::max(1u, src.d / fz);

		dst.data.resize(checkedPyramidElementCount(0, "generated/allocation", dst.w, dst.h, dst.d));

		auto idxSrc = [&](uint32_t x, uint32_t y, uint32_t z) -> size_t {
			return (size_t(z) * src.h + y) * src.w + x;
			};
		auto idxDst = [&](uint32_t x, uint32_t y, uint32_t z) -> size_t {
			return (size_t(z) * dst.h + y) * dst.w + x;
			};

		// Average: good for intensity
		// Nearest: fast and safe for labels
		// MIP: preserves bright/raw maxima and can keep non-zero labels visible
		// Majority: more correct for labels (IDs), more expensive
		for (uint32_t z = 0; z < dst.d; ++z) {
			for (uint32_t y = 0; y < dst.h; ++y) {
				for (uint32_t x = 0; x < dst.w; ++x) {
					const uint32_t sx0 = x * fx;
					const uint32_t sy0 = y * fy;
					const uint32_t sz0 = z * fz;

					if (mode == DownsampleMode::Nearest) {
						const uint32_t sx = std::min(sx0, src.w - 1);
						const uint32_t sy = std::min(sy0, src.h - 1);
						const uint32_t sz = std::min(sz0, src.d - 1);
						dst.data[idxDst(x, y, z)] = src.data[idxSrc(sx, sy, sz)];
						continue;
					}

					if (mode == DownsampleMode::MIP) {
						T maxVal = src.data[idxSrc(std::min(sx0, src.w - 1), std::min(sy0, src.h - 1), std::min(sz0, src.d - 1))];
						for (uint32_t dz = 0; dz < fz; ++dz) {
							uint32_t sz = std::min(sz0 + dz, src.d - 1);
							for (uint32_t dy = 0; dy < fy; ++dy) {
								uint32_t sy = std::min(sy0 + dy, src.h - 1);
								for (uint32_t dx = 0; dx < fx; ++dx) {
									uint32_t sx = std::min(sx0 + dx, src.w - 1);
									maxVal = std::max(maxVal, src.data[idxSrc(sx, sy, sz)]);
								}
							}
						}
						dst.data[idxDst(x, y, z)] = maxVal;
						continue;
					}

					if (mode == DownsampleMode::Majority) {
						// tiny fixed neighborhood; implement with small vector counting
						// Works best if T is integer-ish. For float, falls back to nearest.
						if constexpr (!std::is_integral_v<T>) {
							const uint32_t sx = std::min(sx0, src.w - 1);
							const uint32_t sy = std::min(sy0, src.h - 1);
							const uint32_t sz = std::min(sz0, src.d - 1);
							dst.data[idxDst(x, y, z)] = src.data[idxSrc(sx, sy, sz)];
						}
						else {
							// majority vote in fx*fy*fz
							// for small blocks, O(n^2) is fine
							T bestVal = 0;
							int bestCount = -1;
							std::vector<T> vals;
							vals.reserve(size_t(fx) * fy * fz);

							for (uint32_t dz = 0; dz < fz; ++dz) {
								uint32_t sz = std::min(sz0 + dz, src.d - 1);
								for (uint32_t dy = 0; dy < fy; ++dy) {
									uint32_t sy = std::min(sy0 + dy, src.h - 1);
									for (uint32_t dx = 0; dx < fx; ++dx) {
										uint32_t sx = std::min(sx0 + dx, src.w - 1);
										vals.push_back(src.data[idxSrc(sx, sy, sz)]);
									}
								}
							}

							for (size_t i = 0; i < vals.size(); ++i) {
								int count = 0;
								for (size_t j = 0; j < vals.size(); ++j)
									count += (vals[j] == vals[i]) ? 1 : 0;
								if (count > bestCount) {
									bestCount = count;
									bestVal = vals[i];
								}
							}
							dst.data[idxDst(x, y, z)] = bestVal;
						}
						continue;
					}

					// Average
					// Use wider accumulator for integer types
					if constexpr (std::is_integral_v<T>) {
						using Sum = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
						Sum sum = 0;
						uint64_t cnt = 0;
						for (uint32_t dz = 0; dz < fz; ++dz) {
							uint32_t sz = std::min(sz0 + dz, src.d - 1);
							for (uint32_t dy = 0; dy < fy; ++dy) {
								uint32_t sy = std::min(sy0 + dy, src.h - 1);
								for (uint32_t dx = 0; dx < fx; ++dx) {
									uint32_t sx = std::min(sx0 + dx, src.w - 1);
									sum += static_cast<Sum>(src.data[idxSrc(sx, sy, sz)]);
									++cnt;
								}
							}
						}
						dst.data[idxDst(x, y, z)] = T(sum / static_cast<Sum>(std::max<uint64_t>(1, cnt)));
					}
					else {
						double sum = 0.0;
						double cnt = 0.0;
						for (uint32_t dz = 0; dz < fz; ++dz) {
							uint32_t sz = std::min(sz0 + dz, src.d - 1);
							for (uint32_t dy = 0; dy < fy; ++dy) {
								uint32_t sy = std::min(sy0 + dy, src.h - 1);
								for (uint32_t dx = 0; dx < fx; ++dx) {
									uint32_t sx = std::min(sx0 + dx, src.w - 1);
									sum += double(src.data[idxSrc(sx, sy, sz)]);
									cnt += 1.0;
								}
							}
						}
						dst.data[idxDst(x, y, z)] = T(sum / std::max(1.0, cnt));
					}
				}
			}
		}

		return dst;
	}

	template <class T>
	typename Image<T>::PyramidLevelView Image<T>::makePyramidLevelView(
		const std::shared_ptr<const PyramidLevel>& _storage, int _level, const char* _source) const
	{
		PyramidLevelView view;
		view.w = _storage->w; view.h = _storage->h; view.d = _storage->d;
		view.ptr = _storage->data.data(); view.count = _storage->data.size();
		view.source = _source; view.owner = _storage;
		validatePyramidBuffer(_level, _source, view.w, view.h, view.d, view.count, view.ptr);
		return view;
	}

	template <class T>
	typename Image<T>::PyramidLevelView Image<T>::fullResolutionLevelView() const
	{
		const auto& values = this->pixels();
		PyramidLevelView view;
		view.w = this->width(); view.h = this->height(); view.d = this->depth();
		view.ptr = values.data(); view.count = values.size();
		validatePyramidBuffer(0, "generated/full-resolution", view.w, view.h, view.d, view.count, view.ptr);
		return view;
	}

	template <class T>
	typename Image<T>::PyramidLevelView Image<T>::copyFullResolutionLevel(std::vector<T>& _values) const
	{
		std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
		auto* histogram = dynamic_cast<Histogram<T>*>(getOriginalHistogram("intensity"));
		if (histogram == nullptr)
			throw std::runtime_error("Image pyramid full-resolution histogram type mismatch");
		histogram->copyValues(_values);
		PyramidLevelView view;
		view.w = this->width(); view.h = this->height(); view.d = this->depth();
		view.ptr = _values.data(); view.count = _values.size();
		validatePyramidBuffer(0, "generated/full-resolution", view.w, view.h, view.d, view.count, view.ptr);
		return view;
	}

	template <class T>
	typename Image<T>::PyramidLevelView Image<T>::getOrCreatePyramidLevel(
		int level, uint32_t fx, uint32_t fy, uint32_t fz, DownsampleMode mode) const
	{
		std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
		if (level < 0)
			failPyramidBuffer(level, "generated", width(), height(), depth(), "unknown", 0, "negative requested level");
		if (!m_pyramidalRenderingEnabled)
			level = 0;
		if (level == 0)
			return fullResolutionLevelView();
		ImagePyramidLevelInfo info;
		if (m_nativePyramidReaderCallback && nativePyramidLevelInfo(static_cast<std::size_t>(level), info))
			return getOrCreateNativePyramidLevel(level, info);
		if (level > 31 || fx == 0 || fy == 0 || fz == 0)
			failPyramidBuffer(level, "generated", width(), height(), depth(), "unknown", 0, "invalid level or zero factor");
		return getOrCreateGeneratedPyramidLevel(level, fx, fy, fz, mode);
	}

	template <class T>
	typename Image<T>::PyramidLevelView Image<T>::getOrCreateGeneratedPyramidLevel(
		int level, uint32_t fx, uint32_t fy, uint32_t fz, DownsampleMode mode) const
	{
		// All recursion is locked and uses generated parents only. Rehashing an
		// unordered_map does not invalidate its elements; clearing did invalidate
		// the old borrowed views, so views now retain immutable shared storage.
		PyramidKey key{ level, fx, fy, fz, uint8_t(mode) };
		auto it = m_pyramid.find(key);
		if (it != m_pyramid.end())
			return makePyramidLevelView(it->second, level, "generated");
		PyramidLevel result;
		if (level == 1) {
			const auto source = fullResolutionLevelView();
			result = downsampleRaw(source.ptr, source.w, source.h, source.d, fx, fy, fz, mode);
		}
		else {
			getOrCreateGeneratedPyramidLevel(level - 1, fx, fy, fz, mode);
			const PyramidKey parentKey{ level - 1, fx, fy, fz, uint8_t(mode) };
			const auto parent = m_pyramid.find(parentKey);
			if (parent == m_pyramid.end())
				throw std::runtime_error("Missing generated image pyramid parent at level " + std::to_string(level - 1));
			result = downsampleLevel(*parent->second, fx, fy, fz, mode);
		}
		auto storage = std::make_shared<const PyramidLevel>(std::move(result));
		const auto view = makePyramidLevelView(storage, level, "generated");
		m_pyramid.emplace(key, std::move(storage));
		return view;
	}

	template <class T>
	typename Image<T>::PyramidLevelView Image<T>::getOrCreateDownsampled(
		uint32_t fx, uint32_t fy, uint32_t fz, DownsampleMode mode) const
	{
		std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
		if (!m_pyramidalRenderingEnabled)
			return fullResolutionLevelView();
		fx = std::max(1u, fx); fy = std::max(1u, fy); fz = std::max(1u, fz);
		const PyramidKey key{ -1, fx, fy, fz, uint8_t(mode) };
		auto it = m_pyramid.find(key);
		if (it != m_pyramid.end())
			return makePyramidLevelView(it->second, 0, "generated/downsampled");
		const auto source = fullResolutionLevelView();
		auto storage = std::make_shared<const PyramidLevel>(
			downsampleRaw(source.ptr, source.w, source.h, source.d, fx, fy, fz, mode));
		const auto view = makePyramidLevelView(storage, 0, "generated/downsampled");
		m_pyramid.emplace(key, std::move(storage));
		return view;
	}

	template <class T>
	void Image<T>::addFeatureLabels()
	{
		poca::core::Engine* engine = poca::core::Engine::instance();
		
		if (engine->verbose())
			std::cout << __LINE__ << std::endl;
		std::vector <float> labels(m_volumes.size());
		std::iota(std::begin(labels), std::end(labels), 1);
		addFeature("label", poca::core::generateDataWithLog(labels));
		addFeature("volume", poca::core::generateDataWithLog(m_volumes));
		setCurrentHistogramType("label");
		if (engine->verbose())
			std::cout << __LINE__ << std::endl;
	}

	template <class T>
	void Image<T>::uint8_normalisedData(std::vector <unsigned char>& _normalData) const
	{
		// Normalise and cast
		const std::vector<T>& pixels = this->pixels();
		_normalData.clear();
		_normalData.resize(pixels.size());
		float minV = (float)*std::min_element(pixels.begin(), pixels.end()), maxV = (float)*std::max_element(pixels.begin(), pixels.end()), inter = maxV - minV;
#pragma omp parallel for
		for (auto i = 0; i < pixels.size(); ++i) {
			_normalData[i] = static_cast<unsigned char>(255 * ((float)pixels[i] - minV) / inter);
			//_normalData[i] = static_cast<unsigned char>(m_pixels[i]);
		}
	}

	template <class T>
	void Image<T>::uint16_normalisedData(std::vector <uint16_t>& _normalData) const
	{
		const std::vector<T>& pixels = this->pixels();
		// Normalise and cast
		_normalData.clear();
		_normalData.resize(pixels.size());
		float minV = (float)*std::min_element(pixels.begin(), pixels.end()), maxV = (float)*std::max_element(pixels.begin(), pixels.end()), inter = maxV - minV;
#pragma omp parallel for
		for (auto i = 0; i < pixels.size(); ++i) {
			_normalData[i] = static_cast<uint16_t>(65535 * ((float)pixels[i] - minV) / inter);
		}
	}

	template <class T>
	void Image<T>::uint16_labeledData(std::vector <uint16_t>& _data) const
	{
		const std::vector<T>& pixels = this->pixels();
		_data.clear();
		_data.resize(pixels.size());
#pragma omp parallel for
		for (auto i = 0; i < pixels.size(); ++i) {
			_data[i] = static_cast<uint16_t>(pixels[i]);
		}
	}

	template <class T>
	void Image<T>::float_normalisedData(std::vector <float>& _normalData) const
	{
		const std::vector<T>& pixels = this->pixels();
		// Normalise and cast
		_normalData.clear();
		_normalData.resize(pixels.size());
		float minV = (float)*std::min_element(pixels.begin(), pixels.end()), maxV = (float)*std::max_element(pixels.begin(), pixels.end()), inter = maxV - minV;
#pragma omp parallel for
		for (auto i = 0; i < pixels.size(); ++i) {
			_normalData[i] = ((float)pixels[i] - minV) / inter;
			//_normalData[i] = static_cast<unsigned char>(m_pixels[i]);
		}
	}

	template <class T>
	const T* Image<T>::getImage(const uint32_t _index) const
	{
		const std::vector<T>& pixels = this->pixels();
		auto wh = m_width * m_height;
		return pixels.data() + _index * wh;
	}

	template <class T>
	inline const T* Image<T>::data() const
	{
		return pixels().data();
	}

	template <class T>
	inline T* Image<T>::data()
	{
		return pixels().data();
	}

	template <class T>
	inline const std::vector<T>& Image<T>::pixels() const
	{
		return dynamic_cast<Histogram<T>*>(getOriginalHistogram("intensity"))->getValues();
	}

	template <class T>
	inline std::vector<T>& Image<T>::pixels()
	{
		return dynamic_cast<Histogram<T>*>(getOriginalHistogram("intensity"))->getValues();
	}

	template <class T>
	bool Image<T>::hasPixels() const
	{
		return dynamic_cast<Histogram<T>*>(getOriginalHistogram("intensity"))->hasValues();
	}

	template <class T>
	bool Image<T>::canReloadPixels() const
	{
		return dynamic_cast<Histogram<T>*>(getOriginalHistogram("intensity"))->canMaterializeValues();
	}

	template <class T>
	void Image<T>::releasePixels()
	{
		std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
		if (!m_outOfCoreEnabled || !canReloadPixels())
			return;

		dynamic_cast<Histogram<T>*>(getOriginalHistogram("intensity"))->releaseValues();
	}

	template <class T>
	void Image<T>::setPixelReloadCallback(std::function<void(std::vector<T>&)> _callback)
	{
		dynamic_cast<Histogram<T>*>(getOriginalHistogram("intensity"))->setMaterializeValuesCallback(std::move(_callback));
	}

	template <class T>
	bool Image<T>::canReadFullResolutionPlane() const
	{
		return static_cast<bool>(m_planeReaderCallback) || hasPixels();
	}

	template <class T>
	bool Image<T>::canReadFullResolutionRegion() const
	{
		return static_cast<bool>(m_regionReaderCallback) || canReadFullResolutionPlane() || hasPixels();
	}

	template <class T>
	bool Image<T>::readFullResolutionPlane(const uint64_t _planeIndex, void* _dst, const std::size_t _bytes) const
	{
		if (_planeIndex >= m_depth)
			return false;

		const std::size_t planeBytes = static_cast<std::size_t>(m_width) * static_cast<std::size_t>(m_height) * sizeof(T);
		if (_bytes < planeBytes)
			return false;

		if (m_planeReaderCallback)
			return m_planeReaderCallback(_planeIndex, _dst, _bytes);

		if (!hasPixels())
			return false;

		const std::size_t offset = static_cast<std::size_t>(_planeIndex) * static_cast<std::size_t>(m_width) * static_cast<std::size_t>(m_height);
		std::memcpy(_dst, pixels().data() + offset, planeBytes);
		return true;
	}

	template <class T>
	bool Image<T>::readFullResolutionRegion(const Region3D& _region, void* _dst, const std::size_t _bytes) const
	{
		if (_region.empty() || _region.endX() > m_width || _region.endY() > m_height || _region.endZ() > m_depth)
			return false;

		const std::size_t required = _region.nbVoxels() * sizeof(T);
		if (_bytes < required)
			return false;

		if (m_regionReaderCallback)
			return m_regionReaderCallback(_region, _dst, _bytes);

		if (hasPixels()) {
			const std::vector<T>& vals = pixels();
			T* dst = static_cast<T*>(_dst);
			const std::size_t slice = static_cast<std::size_t>(m_width) * static_cast<std::size_t>(m_height);
			for (uint64_t z = 0; z < _region.depth; ++z) {
				for (uint64_t y = 0; y < _region.height; ++y) {
					const std::size_t srcOffset = static_cast<std::size_t>(_region.z + z) * slice + static_cast<std::size_t>(_region.y + y) * static_cast<std::size_t>(m_width) + static_cast<std::size_t>(_region.x);
					std::memcpy(dst, vals.data() + srcOffset, static_cast<std::size_t>(_region.width) * sizeof(T));
					dst += _region.width;
				}
			}
			return true;
		}

		if (!canReadFullResolutionPlane())
			return false;

		std::vector<T> plane(static_cast<std::size_t>(m_width) * static_cast<std::size_t>(m_height));
		T* dst = static_cast<T*>(_dst);
		for (uint64_t z = 0; z < _region.depth; ++z) {
			if (!readFullResolutionPlane(_region.z + z, plane.data(), plane.size() * sizeof(T)))
				return false;
			for (uint64_t y = 0; y < _region.height; ++y) {
				const std::size_t srcOffset = static_cast<std::size_t>(_region.y + y) * static_cast<std::size_t>(m_width) + static_cast<std::size_t>(_region.x);
				std::memcpy(dst, plane.data() + srcOffset, static_cast<std::size_t>(_region.width) * sizeof(T));
				dst += _region.width;
			}
		}
		return true;
	}

	template <class T>
	void Image<T>::setPlaneReaderCallback(std::function<bool(uint64_t, void*, std::size_t)> _callback)
	{
		m_planeReaderCallback = std::move(_callback);
	}

	template <class T>
	void Image<T>::setRegionReaderCallback(std::function<bool(const Region3D&, void*, std::size_t)> _callback)
	{
		m_regionReaderCallback = std::move(_callback);
	}

	template <class T>
	void Image<T>::setNativePyramidLevelReaderCallback(NativePyramidReader _callback)
	{
		std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
		m_nativePyramidReaderCallback = std::move(_callback);
		invalidatePyramidCache();
	}

	template <class T>
	void Image<T>::setNativePyramidRegionReaderCallback(NativePyramidRegionReader _callback)
	{
		std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
		m_nativePyramidRegionReaderCallback = std::move(_callback);
		invalidatePyramidCache();
	}

	template <class T>
	bool Image<T>::canReadNativePyramidRegion() const
	{
		std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
		return static_cast<bool>(m_nativePyramidRegionReaderCallback);
	}

	template <class T>
	bool Image<T>::readNativePyramidRegion(uint32_t _level, const Region3D& _region, void* _dst, std::size_t _bytes) const
	{
		std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
		ImagePyramidLevelInfo info;
		if (!ImageInterface::nativePyramidLevelInfo(_level, info) || !m_nativePyramidRegionReaderCallback)
			return false;
		if (_region.empty() || _region.x >= info.width || _region.y >= info.height || _region.z >= info.depth ||
			_region.width > info.width - _region.x || _region.height > info.height - _region.y || _region.depth > info.depth - _region.z)
			throw std::invalid_argument("Native pyramid region is outside its level");
		const auto bytes = checkedPyramidByteCount(_level, "native/region", static_cast<uint32_t>(_region.width),
			static_cast<uint32_t>(_region.height), static_cast<uint32_t>(_region.depth), sizeof(T));
		if (_dst == nullptr || _bytes != bytes)
			throw std::invalid_argument("Native pyramid region buffer size mismatch");
		const auto reader = m_nativePyramidRegionReaderCallback;
		const auto revision = m_pyramidRevision;
		const bool read = reader(_level, _region, _dst, _bytes);
		if (revision != m_pyramidRevision)
			throw std::runtime_error("Native region reader changed image pyramid configuration");
		return read;
	}

	template <class T>
	typename Image<T>::PyramidLevelView Image<T>::nativePyramidLevelView(uint32_t _level) const
	{
		std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
		ImagePyramidLevelInfo info;
		if (_level == 0 || _level > static_cast<uint32_t>((std::numeric_limits<int>::max)()) ||
			!ImageInterface::nativePyramidLevelInfo(_level, info) || !m_nativePyramidReaderCallback)
			throw std::invalid_argument("Missing positive native pyramid level/reader");
		return getOrCreateNativePyramidLevel(static_cast<int>(_level), info);
	}

	template <class T>
	typename Image<T>::PyramidLevelView Image<T>::downsampleRegion(const T* _data, uint32_t _w, uint32_t _h, uint32_t _d)
	{
		checkedPyramidByteCount(0, "export/downsample", _w, _h, _d, sizeof(T));
		if (_data == nullptr) throw std::invalid_argument("Null export downsample source");
		auto storage = std::make_shared<const PyramidLevel>(downsampleRaw(_data, _w, _h, _d, 2, 2, 2, DownsampleMode::Average));
		PyramidLevelView view;
		view.w = storage->w; view.h = storage->h; view.d = storage->d;
		view.ptr = storage->data.data(); view.count = storage->data.size(); view.owner = storage;
		return view;
	}

	template <class T>
	void Image<T>::setNativePyramidLevels(const std::vector<ImagePyramidLevelInfo>& _levels)
	{
		std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
		ImageInterface::setNativePyramidLevels(_levels);
	}

	template <class T>
	bool Image<T>::canReadNativePyramid() const
	{
		std::lock_guard<std::recursive_mutex> lock(m_pyramidMutex);
		return static_cast<bool>(m_nativePyramidReaderCallback);
	}

	template <class T>
	typename Image<T>::PyramidLevelView Image<T>::getOrCreateNativePyramidLevel(int _level, ImagePyramidLevelInfo _info) const
	{
		// Metadata and the callable are copied before invoking user code; a
		// reentrant callback cannot destroy its own invocation or metadata.
		auto it = m_nativePyramidCache.find(_level);
		if (it != m_nativePyramidCache.end())
			return makePyramidLevelView(it->second, _level, "native");
		PyramidLevel native;
		native.w = _info.width; native.h = _info.height; native.d = _info.depth;
		const auto count = checkedPyramidElementCount(_level, "native", native.w, native.h, native.d);
		if (count > native.data.max_size())
			failPyramidBuffer(_level, "native", native.w, native.h, native.d, "too large", 0, "vector capacity exceeded");
		const auto reader = m_nativePyramidReaderCallback;
		const auto revision = m_pyramidRevision;
		if (!reader(static_cast<uint32_t>(_level), native.data))
			throw std::runtime_error("Could not read native pyramid level " + std::to_string(_level));
		if (revision != m_pyramidRevision)
			failPyramidBuffer(_level, "native", native.w, native.h, native.d,
				std::to_string(count).c_str(), native.data.size(), "native reader changed image pyramid configuration");
		auto storage = std::make_shared<const PyramidLevel>(std::move(native));
		const auto view = makePyramidLevelView(storage, _level, "native");
		m_nativePyramidCache.emplace(_level, std::move(storage));
		return view;
	}

	template <class T>
	void Image<T>::save(const std::string& _filename) const
	{
		/*std::pair <uint16_t, TinyTIFFWriterSampleFormat> infos = getTinyTiffHeaderInfo(m_type);
		TinyTIFFWriterFile* tif = TinyTIFFWriter_open(filename.toStdString().c_str(), infos.first, infos.second, 1, image->width(), image->height(), TinyTIFFWriter_Greyscale);
		if (tif) {
			for (uint32_t frame = 0; frame < image->depth(); frame++) {
				const float* data = image->getImage(frame);
				TinyTIFFWriter_writeImage(tif, data);
			}
			TinyTIFFWriter_close(tif);
			std::cout << "Image " << filename.toStdString() << "saved" << std::endl;
		}*/
	}

	template <class T>
	const void* Image<T>::getImagePtr(const uint32_t _index) const
	{
		const std::vector<T>& pixels = this->pixels();
		auto wh = m_width * m_height;
		return (void *)(pixels.data() + _index * wh);
	}

	typedef Image<uint8_t> ImageU8;
	typedef Image<uint16_t> ImageU16;
	typedef Image<float> ImageF;
}

#endif
