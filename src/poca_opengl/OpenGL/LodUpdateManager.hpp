/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      LodUpdateManager.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*/

#ifndef LodUpdateManager_h__
#define LodUpdateManager_h__

#include <cstdint>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

namespace poca::opengl {

	class Camera;

	enum class LodRequestStatus : uint8_t {
		Idle = 0,
		Queued,
		Preparing,
		Ready,
		Uploading
	};

	struct ImageLodReady;

	struct ImageLodRequest {
		uint64_t imageId{ 0 };
		uint32_t requestedLevel{ 0 };
		uint32_t requestVersion{ 0 };
		float priority{ 0.f };
		glm::uvec3 targetDims{ 1u, 1u, 1u };
		bool visible{ true };
		std::function<bool(const ImageLodRequest&, ImageLodReady&)> prepareCallback;

		bool operator<(const ImageLodRequest& _other) const
		{
			return priority < _other.priority;
		}
	};

	struct ImageLodReady {
		uint64_t imageId{ 0 };
		uint32_t requestedLevel{ 0 };
		uint32_t requestVersion{ 0 };
		glm::uvec3 preparedDims{ 1u, 1u, 1u };
		bool visible{ true };
		bool obsolete{ false };
		std::shared_ptr<void> payload;
	};

	struct ImageLodState {
		uint32_t currentDisplayedLevel{ 0 };
		uint32_t requestedLevel{ 0 };
		uint32_t latestVersion{ 0 };
		float priority{ 0.f };
		LodRequestStatus status{ LodRequestStatus::Idle };
		uint64_t lastVisibleFrame{ 0 };
		glm::uvec3 targetDims{ 1u, 1u, 1u };
	};

	class LodUpdateManager {
	public:
		explicit LodUpdateManager(Camera* = nullptr);
		~LodUpdateManager();

		void setCamera(Camera*);
		Camera* camera() const { return m_camera; }

		uint32_t request(uint64_t imageId, uint32_t requestedLevel, float priority, const glm::uvec3& targetDims, uint64_t frameIndex, bool visible = true);
		void cancel(uint64_t imageId);
		void clear();

		bool hasQueuedRequests() const;
		bool hasReadyUploads() const;

		bool popNextQueuedRequest(ImageLodRequest&);
		std::vector<ImageLodRequest> drainQueuedRequests();
		std::vector<ImageLodReady> drainReadyUploads(std::size_t maxUploads = 0);

		void markPreparing(uint64_t imageId, uint32_t requestVersion);
		void markReady(const ImageLodReady&);
		void markUploaded(uint64_t imageId, uint32_t displayedLevel);

		bool state(uint64_t imageId, ImageLodState& outState) const;

		friend std::ostream& operator<<(std::ostream&, const LodUpdateManager&);

	private:
		void workerLoop();
		bool popNextQueuedRequestUnsafe(ImageLodRequest&);

		Camera* m_camera{ nullptr };
		std::priority_queue<ImageLodRequest> m_requests;
		std::vector<ImageLodReady> m_ready;
		std::unordered_map<uint64_t, ImageLodState> m_states;
		mutable std::mutex m_mutex;
		std::condition_variable m_condition;
		std::thread m_worker;
		bool m_stopWorker{ false };
	};
}

#endif // LodUpdateManager_h__
