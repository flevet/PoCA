/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      TiffImageIO.hpp
*
* Copyright: Florian Levet (2020-2026)
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*/

#ifndef TiffImageIO_hpp__
#define TiffImageIO_hpp__

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <tinytiffreader.h>

#include "Region3D.hpp"
#include "TiffRegionReader.hpp"

namespace poca::core::tiff {
	template <class T, class M>
	inline void convertSignedToUnsigned(uint8_t* _src, uint8_t* _dst, size_t _numElements)
	{
		T* srcT = reinterpret_cast<T*>(_src);
		M* dstM = reinterpret_cast<M*>(_dst);
		std::transform(srcT, srcT + _numElements, dstM, [](T _val) { return _val < 0 ? 0 : static_cast<M>(_val); });
	}

	template <class T>
	inline bool readPixels(const std::string& _filename, std::vector<T>& _pixels, uint32_t& _width, uint32_t& _height, uint32_t& _depth, uint16_t& _bitsPerSample, uint16_t& _sampleFormat)
	{
		TinyTIFFReaderFile* tiffr = TinyTIFFReader_open(_filename.c_str());
		if (!tiffr)
			return false;

		_width = TinyTIFFReader_getWidth(tiffr);
		_height = TinyTIFFReader_getHeight(tiffr);
		_bitsPerSample = TinyTIFFReader_getBitsPerSample(tiffr, 0);
		_sampleFormat = TinyTIFFReader_getSampleFormat(tiffr);
		_depth = TinyTIFFReader_countFrames(tiffr);

		_pixels.resize(static_cast<size_t>(_width) * static_cast<size_t>(_height) * static_cast<size_t>(_depth));
		uint8_t* stack = reinterpret_cast<uint8_t*>(_pixels.data());
		uint8_t* tmpImage = NULL;

		if (_sampleFormat == 2) {
			if (_bitsPerSample == 16)
				tmpImage = reinterpret_cast<uint8_t*>(new uint16_t[_width * _height * (_bitsPerSample / 8)]);
			else if (_bitsPerSample == 32)
				tmpImage = reinterpret_cast<uint8_t*>(new uint32_t[_width * _height * (_bitsPerSample / 8)]);
		}

		uint32_t frame = 0;
		for (uint32_t n = 0; n < _depth; n++) {
			const uint16_t samples = TinyTIFFReader_getSamplesPerPixel(tiffr);
			uint8_t* image = &stack[frame * _width * _height * (_bitsPerSample / 8)];
			frame++;

			for (uint16_t sample = 0; sample < samples; sample++) {
				if (tmpImage != NULL) {
					TinyTIFFReader_getSampleData(tiffr, tmpImage, sample);
					if (_bitsPerSample == 16)
						convertSignedToUnsigned<int16_t, uint16_t>(tmpImage, image, _width * _height);
					else if (_bitsPerSample == 32)
						convertSignedToUnsigned<int32_t, uint32_t>(tmpImage, image, _width * _height);
				}
				else {
					TinyTIFFReader_getSampleData(tiffr, image, sample);
				}

				if (TinyTIFFReader_wasError(tiffr)) {
					if (tmpImage != NULL) delete[] tmpImage;
					TinyTIFFReader_close(tiffr);
					return false;
				}
			}

			TinyTIFFReader_readNext(tiffr);
		}

		if (tmpImage != NULL) delete[] tmpImage;
		TinyTIFFReader_close(tiffr);
		return true;
	}

	template <class T>
	inline bool readPlane(const std::string& _filename, const uint64_t _planeIndex, std::vector<T>& _plane, uint32_t& _width, uint32_t& _height, uint32_t& _depth)
	{
		TiffRegionReader<T> session(_filename);
		const auto dims = session.dimensions();
		_width = dims.width; _height = dims.height; _depth = dims.depth;
		if (_planeIndex >= _depth) return false;
		_plane.resize(checkedPyramidElementCount(0, "TIFF plane", _width, _height, 1));
		return session.read({ 0, 0, _planeIndex, _width, _height, 1 }, _plane.data(),
			checkedPyramidByteCount(0, "TIFF plane", _width, _height, 1, sizeof(T)));
	}

	template <class T>
	inline bool readRegion(const std::string& _filename, const Region3D& _region, std::vector<T>& _regionValues, uint32_t& _width, uint32_t& _height, uint32_t& _depth)
	{
		if (_region.empty()) return false;
		TiffRegionReader<T> session(_filename);
		const auto dims = session.dimensions();
		_width = dims.width; _height = dims.height; _depth = dims.depth;
		if (_region.x >= _width || _region.y >= _height || _region.z >= _depth ||
			_region.width > _width - _region.x || _region.height > _height - _region.y || _region.depth > _depth - _region.z) return false;
		const auto bytes = checkedPyramidByteCount(0, "TIFF region", uint32_t(_region.width), uint32_t(_region.height), uint32_t(_region.depth), sizeof(T));
		_regionValues.resize(bytes / sizeof(T));
		return session.read(_region, _regionValues.data(), bytes);
	}
}

#endif // TiffImageIO_hpp__
