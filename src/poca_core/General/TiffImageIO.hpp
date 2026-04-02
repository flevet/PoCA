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
		uint16_t bitsPerSample = 0, sampleFormat = 0;
		TinyTIFFReaderFile* tiffr = TinyTIFFReader_open(_filename.c_str());
		if (!tiffr)
			return false;

		_width = TinyTIFFReader_getWidth(tiffr);
		_height = TinyTIFFReader_getHeight(tiffr);
		_depth = TinyTIFFReader_countFrames(tiffr);
		bitsPerSample = TinyTIFFReader_getBitsPerSample(tiffr, 0);
		sampleFormat = TinyTIFFReader_getSampleFormat(tiffr);
		if (_planeIndex >= _depth) {
			TinyTIFFReader_close(tiffr);
			return false;
		}

		for (uint64_t n = 0; n < _planeIndex; ++n) {
			if (!TinyTIFFReader_readNext(tiffr)) {
				TinyTIFFReader_close(tiffr);
				return false;
			}
		}

		_plane.resize(static_cast<size_t>(_width) * static_cast<size_t>(_height));
		uint8_t* tmpImage = NULL;
		if (sampleFormat == 2) {
			if (bitsPerSample == 16)
				tmpImage = reinterpret_cast<uint8_t*>(new uint16_t[_width * _height * (bitsPerSample / 8)]);
			else if (bitsPerSample == 32)
				tmpImage = reinterpret_cast<uint8_t*>(new uint32_t[_width * _height * (bitsPerSample / 8)]);
		}

		if (tmpImage != NULL) {
			TinyTIFFReader_getSampleData(tiffr, tmpImage, 0);
			if (bitsPerSample == 16)
				convertSignedToUnsigned<int16_t, uint16_t>(tmpImage, reinterpret_cast<uint8_t*>(_plane.data()), _width * _height);
			else if (bitsPerSample == 32)
				convertSignedToUnsigned<int32_t, uint32_t>(tmpImage, reinterpret_cast<uint8_t*>(_plane.data()), _width * _height);
			delete[] tmpImage;
		}
		else {
			TinyTIFFReader_getSampleData(tiffr, reinterpret_cast<uint8_t*>(_plane.data()), 0);
		}

		const bool ok = !TinyTIFFReader_wasError(tiffr);
		TinyTIFFReader_close(tiffr);
		return ok;
	}

	template <class T>
	inline bool readRegion(const std::string& _filename, const Region3D& _region, std::vector<T>& _regionValues, uint32_t& _width, uint32_t& _height, uint32_t& _depth)
	{
		if (_region.empty())
			return false;

		_regionValues.resize(_region.nbVoxels());
		std::vector<T> plane;
		T* dst = _regionValues.data();
		for (uint64_t z = 0; z < _region.depth; ++z) {
			if (!readPlane(_filename, _region.z + z, plane, _width, _height, _depth))
				return false;
			for (uint64_t y = 0; y < _region.height; ++y) {
				const std::size_t srcOffset = static_cast<std::size_t>((_region.y + y) * _width + _region.x);
				std::memcpy(dst, plane.data() + srcOffset, static_cast<std::size_t>(_region.width) * sizeof(T));
				dst += _region.width;
			}
		}
		return true;
	}
}

#endif // TiffImageIO_hpp__
