/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#ifndef ImageVolumeResidency_hpp__
#define ImageVolumeResidency_hpp__
#include <functional>
#include <unordered_map>
#include "ImageStreamPolicy.hpp"
namespace poca::opengl {
	// Render-thread only. Callbacks release volume textures, never palette/command state.
	class ImageVolumeResidency {
	public:
		struct Entry {
			std::size_t bytes{ 0 };
			uint64_t lastVisible{ 0 };
			float priority{ 0 };
			bool visible{ false }, pinned{ false };
			std::function<void()> evict;
		};
		explicit ImageVolumeResidency(std::size_t _budget = imageStreamPolicy().gpuBytes) : m_budget(_budget) {}
		void beginFrame() { m_previousVisibleCount = 0; for (const auto& item : m_entries) m_previousVisibleCount += item.second.visible; ++m_frame; for (auto& item : m_entries) item.second.visible = item.second.pinned = false; }
		void touch(uint64_t _id, bool _visible, float _priority, bool _pinned, std::function<void()> _evict) {
			auto& entry = m_entries[_id];
			entry.visible = _visible; entry.priority = _priority; entry.pinned = _pinned && _visible; entry.evict = std::move(_evict);
			if (_visible) entry.lastVisible = m_frame;
		}
		void remove(uint64_t _id) { m_entries.erase(_id); }
		void setBytes(uint64_t _id, std::size_t _bytes) { m_entries.at(_id).bytes = _bytes; }
		std::size_t bytes() const { std::size_t sum = 0; for (const auto& item : m_entries) sum = streamAdd(sum, item.second.bytes); return sum; }
		std::size_t visibleCount() const { std::size_t count = 0; for (const auto& item : m_entries) count += item.second.visible; return std::max(count, m_previousVisibleCount); }
		bool visible(uint64_t _id) const { auto it = m_entries.find(_id); return it != m_entries.end() && it->second.visible; }
		bool admit(uint64_t _id, std::size_t _newBytes, std::size_t _replacedBytes = 0) {
			// Include the old texture during atomic replacement; never credit it before successful upload.
			if (_newBytes > m_budget) return false;
			const auto steadyLimit = m_budget - std::min(imageStreamPolicy().textureBytes, m_budget / 4);
			while (bytes() > m_budget - _newBytes || streamAdd(bytes() - std::min(bytes(), _replacedBytes), _newBytes) > steadyLimit) {
				auto victim = m_entries.end();
				for (auto it = m_entries.begin(); it != m_entries.end(); ++it) {
					if (it->first == _id || !it->second.bytes || it->second.visible || it->second.pinned) continue;
					if (victim == m_entries.end() || it->second.lastVisible < victim->second.lastVisible ||
						(it->second.lastVisible == victim->second.lastVisible && it->first < victim->first)) victim = it;
				}
				if (victim == m_entries.end()) return false;
				const auto callback = victim->second.evict;
				victim->second.bytes = 0;
				if (callback) callback();
			}
			return true;
		}
		void endFrame() {
			for (auto& item : m_entries) {
				auto& entry = item.second;
				if (entry.bytes && !entry.visible && m_frame - entry.lastVisible > imageStreamPolicy().offscreenGraceFrames) {
					const auto callback = entry.evict; entry.bytes = 0;
					if (callback) callback();
				}
			}
		}
		bool needsGraceFrames() const { for (const auto& item : m_entries) if (item.second.bytes && !item.second.visible) return true; return false; }
		void setBudget(std::size_t _budget) { m_budget = _budget; }
		std::size_t budget() const { return m_budget; }
		uint64_t frame() const { return m_frame; }
	private:
		std::unordered_map<uint64_t, Entry> m_entries;
		std::size_t m_budget;
		uint64_t m_frame{ 0 };
		std::size_t m_previousVisibleCount{ 0 };
	};
	struct ImageResidencyFrame {
		ImageVolumeResidency& manager;
		std::function<void()> finish;
		explicit ImageResidencyFrame(ImageVolumeResidency& _manager, std::function<void()> _finish) : manager(_manager), finish(std::move(_finish)) { manager.beginFrame(); }
		~ImageResidencyFrame() noexcept(false) { manager.endFrame(); if (finish) finish(); }
	};
}
#endif
