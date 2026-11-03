/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "DatasetPlacement.hpp"
#include "Misc.h"
#include "stb_rect_pack.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <set>
namespace poca::core {
	std::vector<poca::core::BoundingBox> computeRegularGridPlacement(const std::vector<poca::core::BoundingBox>& _base, int _columns, int _rows)
	{
		std::vector<poca::core::BoundingBox> result = _base;
		if (_base.empty()) return result;
		_columns = std::max(1, _columns);
		_rows = std::max(1, _rows);
		float cellSide = 1.f;
		for (const auto& b : _base)
			cellSide = std::max(cellSide, std::max(b.realWidth(), b.realHeight()));
		for (size_t i = 0; i < result.size(); ++i) {
			const int col = int(i % _columns);
			const int row = int(i / _columns);
			const float x = col * cellSide + (cellSide - _base[i].realWidth()) * 0.5f;
			const float y = row * cellSide + (cellSide - _base[i].realHeight()) * 0.5f;
			result[i].set(x, y, 0.f, x + _base[i].realWidth(), y + _base[i].realHeight(), _base[i].realThick());
		}
		return result;
	}

	std::vector<poca::core::BoundingBox> computePackPlacement(const std::vector<poca::core::BoundingBox>& _base, double _initialMultiplier)
	{
		std::vector<poca::core::BoundingBox> result = _base;
		if (_base.empty()) return result;
		size_t total = 0, maxD = 1;
		std::vector<stbrp_rect> rects;
		for (size_t i = 0; i < _base.size(); ++i) {
			const double width = std::ceil(_base[i].realWidth()), height = std::ceil(_base[i].realHeight());
			const auto limit = (std::numeric_limits<int>::max)();
			if (!std::isfinite(width) || !std::isfinite(height) || width <= 0 || height <= 0 ||
				width > limit || height > limit || _base.size() > size_t(limit) ||
				total + (std::max)(width,height) > limit)
				throw std::overflow_error("Rectangle placement exceeds integer packing limits");
			const int w = std::max(1, int(width));
			const int h = std::max(1, int(std::ceil(_base[i].realHeight())));
			rects.push_back({ int(i), w, h, 0, 0, 0 });
			const size_t d = size_t(std::max(w, h));
			total += d;
			maxD = std::max(maxD, d);
		}
		if (!std::isfinite(_initialMultiplier) || _initialMultiplier < 1. ||
			double(maxD)*_initialMultiplier > (std::numeric_limits<int>::max)())
			throw std::overflow_error("Invalid rectangle packing multiplier/extent");
		Bin bin{ int(maxD*_initialMultiplier), int(maxD*_initialMultiplier) };
		const int maxW = std::max(bin.w, int(std::max<size_t>(total, maxD)));
		const int maxH = std::max(bin.h, int(std::max<size_t>(total, maxD)));
		while (true) {
			auto work = rects;
			if (try_pack(bin, work)) {
				for (const auto& r : work)
					result[r.id].set(float(r.x), float(r.y), 0.f, float(r.x) + _base[r.id].realWidth(), float(r.y) + _base[r.id].realHeight(), _base[r.id].realThick());
				break;
			}
			Bin next = grow(bin, maxW, maxH, 1.25f);
			if (next.w == bin.w && next.h == bin.h)
				throw std::runtime_error("Rectangle placement cannot fit within packing limits");
			bin = next;
		}
		return result;
	}

	std::vector<poca::core::BoundingBox> computeCirclePlacement(const std::vector<poca::core::BoundingBox>& _base, const std::vector<std::vector<size_t>>& _groups, const std::vector<std::string>& _groupLabels, std::vector<PlacementCircle>* _previewGroups)
	{
		std::vector<poca::core::BoundingBox> result = _base;
		if (_base.empty()) return result;
		std::set<size_t> membership;
		for (const auto& group : _groups)
			for (size_t id : group)
				if (id >= _base.size() || !membership.insert(id).second)
					throw std::invalid_argument("Circle placement groups must partition the child footprints");
		if (membership.size() != _base.size()) throw std::invalid_argument("Circle placement has ungrouped children");
		std::vector<std::vector<poca::geometry::PackingCircle>> circlesGroups;
		for (const auto& group : _groups) {
			if (group.empty()) continue;
			circlesGroups.push_back({});
			auto& circles = circlesGroups.back();
			for (size_t id : group) {
				if (id >= _base.size()) continue;
				const float radius = 0.5f * std::sqrt(_base[id].realWidth() * _base[id].realWidth() + _base[id].realHeight() * _base[id].realHeight()) + 1.f;
				circles.push_back({ radius, radius, std::max(1.f, radius), int(id) });
			}
			poca::geometry::packCirclesFast(circles, 200, 0.0015f, 0.55f);
		}
		std::vector<poca::geometry::PackingCircle> groupCircles;
		std::vector<poca::core::BoundingBox> groupBounds;
		for (size_t g = 0; g < circlesGroups.size(); ++g) {
			poca::core::BoundingBox bbox = poca::core::BoundingBox::initBBox();
			for (const auto& c : circlesGroups[g]) {
				bbox.addPointBBox(c.x - c.r, c.y - c.r, 0.f);
				bbox.addPointBBox(c.x + c.r, c.y + c.r, 0.f);
			}
			groupBounds.push_back(bbox);
			const float radius = std::hypot(bbox.realWidth(), bbox.realHeight()) / 2.f;
			groupCircles.push_back({ radius, radius, std::max(1.f, radius), int(g) });
		}
		poca::geometry::packCirclesFast(groupCircles, 200, 0.0015f, 0.55f);
		if (_previewGroups) _previewGroups->clear();
		for (const auto& cg : groupCircles) {
			const int g = cg.id;
			const auto& circles = circlesGroups[g];
			const auto& bbox = groupBounds[g];
			const float gx = cg.x - bbox.realWidth() * .5f;
			const float gy = cg.y - bbox.realHeight() * .5f;
			if (_previewGroups) _previewGroups->push_back({ cg.x, cg.y, cg.r, g >= 0 && size_t(g) < _groupLabels.size() ? _groupLabels[g] : "Group " + std::to_string(g + 1) });
			for (const auto& c : circles) {
				const size_t id = size_t(c.id);
				const float x = c.x - _base[id].realWidth() * .5f - bbox.x() + gx;
				const float y = c.y - _base[id].realHeight() * .5f - bbox.y() + gy;
				result[id].set(x, y, 0.f, x + _base[id].realWidth(), y + _base[id].realHeight(), _base[id].realThick());
			}
		}
		return result;
	}


}
