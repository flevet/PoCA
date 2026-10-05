/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#ifndef ImageStorageSampling_hpp__
#define ImageStorageSampling_hpp__
#include <algorithm>
#include <array>
#include <vector>
#include <General/ImagePyramidValidation.hpp>
#include <General/Region3D.hpp>
namespace poca::core {
	// At most 32,768 voxels / 128 KiB for all supported scalar types, distributed
	// across up to 4x4x4 non-overlapping small regions of the coarsest stored level.
	template <class T, class Reader>
	std::vector<T> readStorageStatisticsSample(uint32_t _width, uint32_t _height, uint32_t _depth, Reader _reader) {
		const uint32_t dims[3]{ _width, _height, _depth };
		std::array<uint32_t, 3> grid{}, block{};
		for (int a = 0; a < 3; ++a) {
			if (!dims[a]) throw std::invalid_argument("Empty storage statistics level");
			block[a] = std::min(8u, dims[a]); grid[a] = std::clamp(dims[a] / block[a], 1u, 4u);
		}
		const auto tileCount = checkedPyramidElementCount(0, "statistics sample", block[0], block[1], block[2]);
		const auto sampleCount = tileCount * grid[0] * grid[1] * grid[2]; // Each factor is bounded above.
		if (sampleCount > 32768 || sampleCount * sizeof(T) > 128 * 1024) throw std::logic_error("Statistics sample exceeds policy");
		std::vector<T> sample; sample.reserve(sampleCount);
		std::vector<T> tile(tileCount);
		for (uint32_t z = 0; z < grid[2]; ++z)
			for (uint32_t y = 0; y < grid[1]; ++y)
				for (uint32_t x = 0; x < grid[0]; ++x) {
					const uint32_t index[3]{ x, y, z }; std::array<uint64_t, 3> start{};
					for (int a = 0; a < 3; ++a) start[a] = grid[a] == 1 ? (dims[a] - block[a]) / 2 : uint64_t(dims[a] - block[a]) * index[a] / (grid[a] - 1);
					if (!_reader(Region3D{ start[0], start[1], start[2], block[0], block[1], block[2] }, tile.data(), tile.size() * sizeof(T)))
						throw std::runtime_error("Storage statistics regional sample read failed");
					sample.insert(sample.end(), tile.begin(), tile.end());
				}
		return sample;
	}
}
#endif
