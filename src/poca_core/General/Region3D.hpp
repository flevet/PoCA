/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Region3D.hpp
*
* Copyright: Florian Levet (2020-2026)
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*/

#ifndef Region3D_hpp__
#define Region3D_hpp__

#include <cstddef>
#include <cstdint>

namespace poca::core {
	struct Region3D {
		uint64_t x{ 0 }, y{ 0 }, z{ 0 };
		uint64_t width{ 0 }, height{ 0 }, depth{ 0 };

		constexpr bool empty() const { return width == 0 || height == 0 || depth == 0; }
		constexpr uint64_t endX() const { return x + width; }
		constexpr uint64_t endY() const { return y + height; }
		constexpr uint64_t endZ() const { return z + depth; }
		constexpr std::size_t nbVoxels() const { return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(depth); }
	};
}

#endif // Region3D_hpp__
