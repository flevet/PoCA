/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#ifndef ImageStreamPolicy_hpp__
#define ImageStreamPolicy_hpp__
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <glm/glm.hpp>
#include <General/Region3D.hpp>
namespace poca::opengl {
	struct ImageStreamCanceled : std::exception { const char* what() const noexcept override { return "Obsolete image stream preparation"; } };
	struct ImageStreamPolicy {
		std::size_t gpuBytes{ 512ull * 1024 * 1024 };
		std::size_t cpuBytes{ 256ull * 1024 * 1024 };
		std::size_t textureBytes{ 64ull * 1024 * 1024 };
		std::size_t scratchBytes{ 1024ull * 1024 };
		std::size_t uploadBytes{ 16ull * 1024 * 1024 };
		uint32_t previewEdge{ 64 }, displayEdge{ 512 }, interactiveEdge{ 128 };
		uint32_t regionQuantum{ 16 }, offscreenGraceFrames{ 12 };
		double guardFraction{ .25 }, safeFraction{ .10 };
	};
	inline const ImageStreamPolicy& imageStreamPolicy() { static const ImageStreamPolicy policy; return policy; }
	inline std::size_t streamMultiply(std::size_t _a, std::size_t _b) {
		if (_b && _a > (std::numeric_limits<std::size_t>::max)() / _b)
			throw std::overflow_error("Image stream byte/element count overflow");
		return _a * _b;
	}
	inline std::size_t streamAdd(std::size_t _a, std::size_t _b) {
		if (_a > (std::numeric_limits<std::size_t>::max)() - _b)
			throw std::overflow_error("Image stream byte sum overflow");
		return _a + _b;
	}
	inline std::size_t streamBytes(const glm::uvec3& _dims, std::size_t _elementBytes) {
		if (!_dims.x || !_dims.y || !_dims.z || !_elementBytes) throw std::invalid_argument("Empty image stream dimensions");
		return streamMultiply(streamMultiply(streamMultiply(_dims.x, _dims.y), _dims.z), _elementBytes);
	}
	inline bool sameRegion(const poca::core::Region3D& _a, const poca::core::Region3D& _b) {
		return _a.x == _b.x && _a.y == _b.y && _a.z == _b.z &&
			_a.width == _b.width && _a.height == _b.height && _a.depth == _b.depth;
	}
	inline void validateStreamRegion(const poca::core::Region3D& _r, const glm::uvec3& _dims) {
		if (_r.empty() || _r.x >= _dims.x || _r.y >= _dims.y || _r.z >= _dims.z ||
			_r.width > _dims.x - _r.x || _r.height > _dims.y - _r.y || _r.depth > _dims.z - _r.z)
			throw std::invalid_argument("Image stream region outside source dimensions");
	}
}
#endif
