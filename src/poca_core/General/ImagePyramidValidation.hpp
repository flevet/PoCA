/*
* Software: PoCA: Point Cloud Analyst
* File: ImagePyramidValidation.hpp
* Copyright: Florian Levet (2026)
* License: LGPL v3
*/
#ifndef ImagePyramidValidation_hpp__
#define ImagePyramidValidation_hpp__

#include <cstdint>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <stdexcept>

namespace poca::core {
	inline void failPyramidBuffer(int64_t _level, const char* _source,
		uint32_t _w, uint32_t _h, uint32_t _d, const char* _expected,
		std::size_t _actual, const char* _reason)
	{
		std::ostringstream message;
		message << "Invalid image pyramid: level=" << _level << " source=" << _source
			<< " dimensions=" << _w << "x" << _h << "x" << _d
			<< " expected=" << _expected << " actual=" << _actual << " reason=" << _reason;
		throw std::runtime_error(message.str());
	}

	inline std::size_t checkedPyramidElementCount(int64_t _level, const char* _source,
		uint32_t _w, uint32_t _h, uint32_t _d, std::size_t _actual = 0)
	{
		if (_w == 0 || _h == 0 || _d == 0)
			failPyramidBuffer(_level, _source, _w, _h, _d, "0", _actual, "nonpositive dimension");
		std::size_t count = _w;
		const std::size_t limit = (std::numeric_limits<std::size_t>::max)();
		if (count > limit / _h || count * _h > limit / _d)
			failPyramidBuffer(_level, _source, _w, _h, _d, "overflow", _actual, "element count overflow");
		count *= _h;
		count *= _d;
		if (_level < 0 || _level > (std::numeric_limits<int>::max)()) {
			const std::string expected = std::to_string(count);
			failPyramidBuffer(_level, _source, _w, _h, _d, expected.c_str(), _actual, "invalid requested level");
		}
		return count;
	}

	inline std::size_t validatePyramidBuffer(int64_t _level, const char* _source,
		uint32_t _w, uint32_t _h, uint32_t _d, std::size_t _actual, const void* _ptr)
	{
		const std::size_t count = checkedPyramidElementCount(_level, _source, _w, _h, _d, _actual);
		if (_actual != count || _ptr == nullptr) {
			const std::string expected = std::to_string(count);
			failPyramidBuffer(_level, _source, _w, _h, _d, expected.c_str(), _actual,
				_ptr == nullptr ? "null data pointer" : "buffer size mismatch");
		}
		return count;
	}

	inline std::size_t checkedPyramidByteCount(int64_t _level, const char* _source,
		uint32_t _w, uint32_t _h, uint32_t _d, std::size_t _elementBytes)
	{
		const std::size_t count = checkedPyramidElementCount(_level, _source, _w, _h, _d);
		if (_elementBytes == 0 || count > (std::numeric_limits<std::size_t>::max)() / _elementBytes)
			failPyramidBuffer(_level, _source, _w, _h, _d, "overflow", 0, "byte count overflow");
		return count * _elementBytes;
	}
}
#endif // ImagePyramidValidation_hpp__
