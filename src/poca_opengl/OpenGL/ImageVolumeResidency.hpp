/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#ifndef ImageVolumeResidency_hpp__
#define ImageVolumeResidency_hpp__
#include <functional>
#include <General/PerformanceProfiler.hpp>
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
			bool visible{ false }, pinned{ false }, highResolution{ false };
			std::function<void()> evict;
			std::function<void()> trimDetail;
		};
		explicit ImageVolumeResidency(std::size_t _budget = imageStreamPolicy().gpuBytes) : m_budget(_budget) {}
		void beginFrame() {
			m_previousVisibleCount = m_visibleCount; m_visibleCount = m_forcedCount = 0; ++m_frame;
			poca::core::PerformanceProfiler::renderCount(poca::core::PerformanceProfiler::RenderMetric::ResidencyEntriesExamined, m_entries.size());
			for (auto& item : m_entries) {
				item.second.visible = item.second.pinned = false;
				m_forcedCount += forcedCandidate(item.second);
			}
		}
		void touch(uint64_t _id, bool _visible, float _priority, bool _pinned, std::function<void()> _evict, std::function<void()> _trimDetail = {}) {
			auto& entry = m_entries[_id];
			m_visibleCount -= entry.visible; m_forcedCount -= forcedCandidate(entry);
			entry.visible = _visible; entry.priority = _priority; entry.pinned = _pinned && _visible; entry.evict = std::move(_evict); entry.trimDetail = std::move(_trimDetail);
			if (_visible) entry.lastVisible = m_frame;
			m_visibleCount += entry.visible; m_forcedCount += forcedCandidate(entry);
		}
		void remove(uint64_t _id) {
			auto found = m_entries.find(_id); if (found == m_entries.end()) return;
			m_visibleCount -= found->second.visible; m_forcedCount -= forcedCandidate(found->second); m_entries.erase(found);
		}
		void setBytes(uint64_t _id, std::size_t _bytes) { m_entries.at(_id).bytes = _bytes; }
		std::size_t bytes() const { poca::core::PerformanceProfiler::renderCount(poca::core::PerformanceProfiler::RenderMetric::ResidencyEntriesExamined, m_entries.size()); std::size_t sum = 0; for (const auto& item : m_entries) sum = streamAdd(sum, item.second.bytes); return sum; }
		std::size_t visibleCount() const { return std::max(m_visibleCount, m_previousVisibleCount); }
		bool visible(uint64_t _id) const { auto it = m_entries.find(_id); return it != m_entries.end() && it->second.visible; }
		bool admit(uint64_t _id, std::size_t _newBytes, std::size_t _replacedBytes = 0) {
			// Include the old texture during atomic replacement; never credit it before successful upload.
			if (_newBytes > m_budget) return false;
			const auto steadyLimit = m_budget - std::min(imageStreamPolicy().textureBytes, m_budget / 4);
			while (bytes() > m_budget - _newBytes || streamAdd(bytes() - std::min(bytes(), _replacedBytes), _newBytes) > steadyLimit) {
				auto victim = m_entries.end();
				poca::core::PerformanceProfiler::renderCount(poca::core::PerformanceProfiler::RenderMetric::ResidencyEntriesExamined, m_entries.size());
				for (auto it = m_entries.begin(); it != m_entries.end(); ++it) {
					if (it->first == _id || !it->second.bytes || it->second.visible || it->second.pinned) continue;
					if (victim == m_entries.end() || it->second.lastVisible < victim->second.lastVisible ||
						(it->second.lastVisible == victim->second.lastVisible && it->first < victim->first)) victim = it;
				}
				if (victim == m_entries.end()) {
					// Forced detail can retire other visible Detail allocations, preserving
					// their current previews and all scientific/command state.
					if (!m_entries.at(_id).highResolution) return false;
					poca::core::PerformanceProfiler::renderCount(poca::core::PerformanceProfiler::RenderMetric::ResidencyEntriesExamined, m_entries.size());
					for (auto it = m_entries.begin(); it != m_entries.end(); ++it) {
						if (it->first == _id || !it->second.bytes || it->second.highResolution || !it->second.trimDetail) continue;
						if (victim == m_entries.end() || it->second.priority < victim->second.priority ||
							(it->second.priority == victim->second.priority && it->first < victim->first)) victim = it;
					}
					if (victim == m_entries.end()) return false;
					const auto trim = victim->second.trimDetail; victim->second.trimDetail = {};
					trim();
					continue;
				}
				const auto callback = victim->second.evict;
				victim->second.bytes = 0;
				if (callback) callback();
			}
			return true;
		}
		void endFrame() {
			poca::core::PerformanceProfiler::renderCount(poca::core::PerformanceProfiler::RenderMetric::ResidencyEntriesExamined, m_entries.size());
			for (auto& item : m_entries) {
				auto& entry = item.second;
				if (entry.bytes && !entry.visible && m_frame - entry.lastVisible > imageStreamPolicy().offscreenGraceFrames) {
					const auto callback = entry.evict; entry.bytes = 0;
					if (callback) callback();
				}
			}
		}
		bool needsGraceFrames() const {
			std::size_t examined = 0;
			for (const auto& item : m_entries) {
				++examined;
				if (item.second.bytes && !item.second.visible) {
					poca::core::PerformanceProfiler::renderCount(poca::core::PerformanceProfiler::RenderMetric::ResidencyEntriesExamined, examined); return true;
				}
			}
			poca::core::PerformanceProfiler::renderCount(poca::core::PerformanceProfiler::RenderMetric::ResidencyEntriesExamined, examined); return false;
		}
		// Forced images share half the pool; all images share the other half of the
		// pool. The previous frame stabilizes shares during visibility traversal.
		std::size_t imageShare(uint64_t id, bool highResolution) const {
			poca::core::PerformanceProfiler::renderCount(poca::core::PerformanceProfiler::RenderMetric::ImageShareCalls);
			const auto count = std::max(std::size_t(1), visibleCount());
			if (!m_forcedCount) return m_budget / count;
			const auto fair = (m_budget - m_budget / 2) / count;
			return highResolution && m_entries.count(id) ? streamAdd(fair, m_budget / 2 / m_forcedCount) : fair;
		}
		void prioritize(uint64_t id, bool enabled) {
			auto& entry = m_entries.at(id);
			m_forcedCount -= forcedCandidate(entry); entry.highResolution = enabled; m_forcedCount += forcedCandidate(entry);
		}
		void setBudget(std::size_t _budget) { m_budget = _budget; }
		std::size_t budget() const { return m_budget; }
		uint64_t frame() const { return m_frame; }
	private:
		bool forcedCandidate(const Entry& entry) const { return entry.highResolution && (entry.visible || (m_frame > 0 && entry.lastVisible == m_frame - 1)); }
		std::size_t m_visibleCount{ 0 }, m_forcedCount{ 0 };
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
