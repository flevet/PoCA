/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#ifndef TiffRegionReader_hpp__
#define TiffRegionReader_hpp__
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <tinytiffreader.h>
#include "ImagePyramidValidation.hpp"
#include "Region3D.hpp"
namespace poca::core::tiff {
	struct TiffDimensions { uint32_t width{ 0 }, height{ 0 }, depth{ 0 }; };
	struct TiffReadMetrics { uint32_t openCount{ 0 }; uint64_t planesDecoded{ 0 }; };
	// Only APIs already used by PoCA. The adapter also permits source-only fake-reader fixtures.
	struct TinyTiffReaderApi {
		using File = TinyTIFFReaderFile;
		static File* open(const char* name) { return TinyTIFFReader_open(name); }
		static void close(File* file) { TinyTIFFReader_close(file); }
		static uint32_t width(File* f) { return TinyTIFFReader_getWidth(f); }
		static uint32_t height(File* f) { return TinyTIFFReader_getHeight(f); }
		static uint32_t depth(File* f) { return TinyTIFFReader_countFrames(f); }
		static uint16_t bits(File* f) { return TinyTIFFReader_getBitsPerSample(f, 0); }
		static uint16_t format(File* f) { return TinyTIFFReader_getSampleFormat(f); }
		static uint16_t samples(File* f) { return TinyTIFFReader_getSamplesPerPixel(f); }
		static bool next(File* f) { return TinyTIFFReader_readNext(f) != 0; }
		static bool error(File* f) { return TinyTIFFReader_wasError(f) != 0; }
		static void decode(File* f, void* dst) { TinyTIFFReader_getSampleData(f, dst, 0); }
	};
	template <class T, class Api = TinyTiffReaderApi> class TiffRegionReader {
	public:
		explicit TiffRegionReader(std::string filename, TiffDimensions expected = {})
			: m_filename(std::move(filename)), m_dims(expected) {
			reopen();
			if (!m_dims.width) m_dims = { Api::width(m_file.get()), Api::height(m_file.get()), Api::depth(m_file.get()) };
			checkedPyramidByteCount(0, "TIFF plane", m_dims.width, m_dims.height, 1, sizeof(T));
			if (!m_dims.depth) throw std::invalid_argument("Empty TIFF stack");
			validatePlane();
		}
		TiffDimensions dimensions() const { return m_dims; }
		TiffReadMetrics metrics() const { return m_metrics; }
		bool read(const Region3D& region, void* destination, std::size_t bytes) {
			if (!destination || region.empty() || region.x >= m_dims.width || region.y >= m_dims.height || region.z >= m_dims.depth ||
				region.width > m_dims.width - region.x || region.height > m_dims.height - region.y || region.depth > m_dims.depth - region.z)
				return false;
			const auto required = checkedPyramidByteCount(0, "TIFF region", uint32_t(region.width), uint32_t(region.height), uint32_t(region.depth), sizeof(T));
			if (bytes < required) return false;
			auto* dst = static_cast<uint8_t*>(destination);
			for (uint64_t z = 0; z < region.depth; ++z) {
				const auto index = region.z + z;
				// Ordinary adaptive slabs advance monotonically. Majority/scientific
				// arbitrary reads may rewind explicitly; never silently return a wrong plane.
				if (index < m_frame) reopen();
				while (m_frame < index) {
					if (!Api::next(m_file.get()) || Api::error(m_file.get())) return false;
					++m_frame;
				}
				if (m_decoded != index) {
					validatePlane();
					const auto count = checkedPyramidElementCount(0, "TIFF plane", m_dims.width, m_dims.height, 1);
					m_plane.resize(count);
					if (Api::format(m_file.get()) == 2 && (std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t>)) {
						m_signed.resize(count); Api::decode(m_file.get(), m_signed.data());
						for (std::size_t i = 0; i < count; ++i) {
							if constexpr (std::is_same_v<T, uint16_t>) {
								int16_t value; std::memcpy(&value, &m_signed[i], sizeof(value)); m_plane[i] = value < 0 ? 0 : T(value);
							}
							else if constexpr (std::is_same_v<T, uint32_t>) {
								int32_t value; std::memcpy(&value, &m_signed[i], sizeof(value)); m_plane[i] = value < 0 ? 0 : T(value);
							}
						}
					}
					else Api::decode(m_file.get(), m_plane.data());
					if (Api::error(m_file.get())) return false;
					m_decoded = index; ++m_metrics.planesDecoded;
				}
				for (uint64_t y = 0; y < region.height; ++y) {
					const auto offset = (std::size_t(region.y + y) * m_dims.width) + region.x;
					std::memcpy(dst, m_plane.data() + offset, std::size_t(region.width) * sizeof(T));
					dst += std::size_t(region.width) * sizeof(T);
				}
			}
			return true;
		}
	private:
		struct Close { void operator()(typename Api::File* file) const { if (file) Api::close(file); } };
		void reopen() {
			m_file.reset(Api::open(m_filename.c_str()));
			if (!m_file || Api::error(m_file.get())) throw std::runtime_error("Cannot open TIFF regional source: " + m_filename);
			++m_metrics.openCount; m_frame = 0; m_decoded = UINT64_MAX;
		}
		void validatePlane() const {
			const auto format = Api::format(m_file.get());
			const bool validFormat = std::is_floating_point_v<T> ? format == 3 :
				(std::is_signed_v<T> ? format == 2 : (format == 1 || (format == 2 && sizeof(T) > 1)));
			if (Api::width(m_file.get()) != m_dims.width || Api::height(m_file.get()) != m_dims.height ||
				Api::bits(m_file.get()) != sizeof(T) * 8 || Api::samples(m_file.get()) != 1 || !validFormat)
				throw std::runtime_error("TIFF regional source dimensions/type/samples changed or are unsupported");
		}
		std::string m_filename;
		TiffDimensions m_dims;
		std::unique_ptr<typename Api::File, Close> m_file;
		std::vector<T> m_plane, m_signed;
		uint64_t m_frame{ 0 }, m_decoded{ UINT64_MAX };
		TiffReadMetrics m_metrics;
	};
}
#endif
