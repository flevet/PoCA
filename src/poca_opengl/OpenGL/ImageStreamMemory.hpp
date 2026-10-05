/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#ifndef ImageStreamMemory_hpp__
#define ImageStreamMemory_hpp__
#include <memory>
#include <mutex>
#include "ImageStreamPolicy.hpp"
namespace poca::opengl {
	// Shared accounting survives a manager and any drained-but-still-owned ready payload.
	class ImageStreamMemory : public std::enable_shared_from_this<ImageStreamMemory> {
	public:
		struct Usage { std::size_t preparing{ 0 }, ready{ 0 }; };
		class Reservation {
		public:
			Reservation(std::shared_ptr<ImageStreamMemory> _owner, std::size_t _bytes) : m_owner(std::move(_owner)), m_bytes(_bytes) {}
			~Reservation() {
				std::lock_guard<std::mutex> lock(m_owner->m_mutex);
				(m_ready ? m_owner->m_usage.ready : m_owner->m_usage.preparing) -= m_bytes;
			}
			void ready(std::size_t _retainedBytes) {
				std::lock_guard<std::mutex> lock(m_owner->m_mutex);
				if (m_ready || _retainedBytes > m_bytes) throw std::logic_error("Invalid image stream reservation transition");
				m_owner->m_usage.preparing -= m_bytes;
				m_bytes = _retainedBytes;
				m_owner->m_usage.ready += m_bytes;
				m_ready = true;
			}
			Reservation(const Reservation&) = delete;
			Reservation& operator=(const Reservation&) = delete;
		private:
			std::shared_ptr<ImageStreamMemory> m_owner;
			std::size_t m_bytes;
			bool m_ready{ false };
		};
		explicit ImageStreamMemory(std::size_t _limit = imageStreamPolicy().cpuBytes) : m_limit(_limit) {}
		std::shared_ptr<Reservation> reserve(std::size_t _bytes) {
			std::lock_guard<std::mutex> lock(m_mutex);
			const auto used = streamAdd(m_usage.preparing, m_usage.ready);
			if (!_bytes || _bytes > m_limit || _bytes > m_limit - used) return {};
			auto result = std::make_shared<Reservation>(shared_from_this(), _bytes);
			m_usage.preparing += _bytes;
			return result;
		}
		Usage usage() const { std::lock_guard<std::mutex> lock(m_mutex); return m_usage; }
	private:
		mutable std::mutex m_mutex;
		Usage m_usage;
		std::size_t m_limit;
	};
	inline std::shared_ptr<ImageStreamMemory> sharedImageStreamMemory() {
		static const auto accounting = std::make_shared<ImageStreamMemory>(); return accounting;
	}
}
#endif
