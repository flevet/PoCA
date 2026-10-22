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
#include <chrono>
#include <ostream>
#include <string>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <General/Region3D.hpp>
#include "ImageStreamMemory.hpp"
#include "ImageVolumeResidency.hpp"

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
		uint64_t viewGeneration{ 0 };
		float priority{ 0.f };
		glm::vec2 effectiveTarget{ 1 };
		glm::uvec3 targetDims{ 1u, 1u, 1u };
		glm::uvec3 downsampleFactors{ 1u, 1u, 1u };
		// Empty sourceRegion retains the bounded legacy test/whole-level contract.
		poca::core::Region3D sourceRegion;
		glm::uvec3 sourceDims{ 1u };
		glm::vec3 residentBottom{ 0.f }, residentTop{ 1.f }, imageBottom{ 0.f }, imageTop{ 1.f };
		std::size_t estimatedPreparedBytes{ 0 }, textureBytes{ 0 }, readerScratchBytes{ 0 };
		bool regional{ false }, residentSource{ false }, forceLevelZero{ false }, incremental{ false };
		uint64_t chunkIndex{ 0 };
		std::shared_ptr<ImageStreamMemory::Reservation> readerReservation;
		poca::core::Region3D outputRegion; // Internal bounded destination brick; empty means whole target.
		std::size_t textureBudgetBytes{ 0 };
		std::string qualityReason;
		std::string planFailure; // Planning refusal only; never a reduced forced Detail.
		ImageStreamRole role{ ImageStreamRole::Detail };
		uint32_t navigationClass{ 0 };
		int currentFrame{ -1 };
		std::string reductionMode{ "MIP" };
		std::function<bool()> canceled;
		bool visible{ true };
		std::function<bool(const ImageLodRequest&, ImageLodReady&)> prepareCallback;
		std::function<bool(const ImageLodReady&)> uploadCallback;

		bool operator<(const ImageLodRequest& _other) const
		{
			return priority < _other.priority;
		}
	
		friend std::ostream& operator<<(std::ostream&, const ImageLodRequest&);
	};

	struct ImageLodReady {
		uint64_t imageId{ 0 };
		uint32_t requestedLevel{ 0 };
		uint32_t requestVersion{ 0 };
		uint64_t viewGeneration{ 0 };
		glm::uvec3 preparedDims{ 1u, 1u, 1u };
		bool visible{ true };
		bool obsolete{ false };
		ImageStreamRole role{ ImageStreamRole::Detail };
		uint32_t navigationClass{ 0 };
		std::size_t preparedBytes{ 0 };
		bool moreChunks{ false };
		std::shared_ptr<ImageLodRequest> continuation; // Same immutable version, next brick; no payload cycle.
		std::shared_ptr<ImageStreamMemory::Reservation> memoryReservation, readerReservation;
		std::shared_ptr<void> payload;
		std::function<bool(const ImageLodReady&)> uploadCallback;
	
		friend std::ostream& operator<<(std::ostream&, const ImageLodReady&);
	};

	struct ImageLodState {
		uint32_t currentDisplayedLevel{ 0 };
		uint32_t requestedLevel{ 0 };
		uint32_t latestVersion{ 0 };
		uint64_t viewGeneration{ 0 };
		float priority{ 0.f };
		glm::vec2 effectiveTarget{ 1 };
		LodRequestStatus status{ LodRequestStatus::Idle };
		uint64_t lastVisibleFrame{ 0 };
		glm::uvec3 targetDims{ 1u, 1u, 1u };
		glm::uvec3 downsampleFactors{ 1u, 1u, 1u };
		poca::core::Region3D sourceRegion;
		bool residentSource{ false }, failed{ false }, forceLevelZero{ false }, incremental{ false };
		std::size_t textureBytes{ 0 };
		ImageStreamRole role{ ImageStreamRole::Detail };
		uint32_t navigationClass{ 0 };
		int currentFrame{ -1 };
		glm::uvec3 sourceDims{ 1u };
		glm::vec3 residentBottom{ 0.f }, residentTop{ 1.f }, imageBottom{ 0.f }, imageTop{ 1.f };
		std::string reductionMode;
		std::string failureReason;
		bool visible{ true };
	
		friend std::ostream& operator<<(std::ostream&, const ImageLodState&);
	};

	class LodUpdateManager {
	public:
		explicit LodUpdateManager(Camera* = nullptr, std::shared_ptr<ImageStreamMemory> = sharedImageStreamMemory());
		~LodUpdateManager();

		void setCamera(Camera*);
		Camera* camera() const { return m_camera; }

		uint32_t request(const ImageLodRequest& request, uint64_t frameIndex);
		void cancel(uint64_t imageId);
		void invalidateDetail(uint64_t imageId);
		void forget(uint64_t imageId);
		void cancelInvisible();
		bool isCurrent(const ImageLodReady&) const;
		ImageVolumeResidency& residency() { return m_residency; }
		void refreshGpuBudget(); // Current rendering context only; throttled outside allocation admission.
		void refreshGpuBudgetForAllocation();
		const std::string& gpuBudgetSource() const { return m_gpuBudgetSource; }
		std::size_t availableGpuBytes() const { return m_availableGpuBytes; }
		bool gpuMemoryHintAvailable() const { return m_gpuMemoryHintAvailable; }
		void reportFailure(uint64_t, uint32_t, const std::string&);
		ImageStreamMemory::Usage memoryUsage() const { return m_memory->usage(); }
		void clear();

		bool hasQueuedRequests() const;
		bool hasReadyUploads() const;
		bool uploadReadyForFrame();
		void acknowledgeRedraw();

		bool popNextQueuedRequest(ImageLodRequest&);
		std::vector<ImageLodRequest> drainQueuedRequests();
		std::vector<ImageLodReady> drainReadyUploads(std::size_t maxUploads = 0, std::size_t maxPreparedBytes = 0);

		void markPreparing(uint64_t imageId, uint32_t requestVersion);
		void markReady(const ImageLodReady&);
		void markUploaded(uint64_t imageId, uint32_t displayedLevel, uint32_t version = 0);

		bool state(uint64_t imageId, ImageLodState& outState) const;

		friend std::ostream& operator<<(std::ostream&, const LodUpdateManager&);

	private:
		void workerLoop();
		void requestRedrawUnsafe(); // m_mutex held; receiver lifetime is Camera-owned.
		bool takeAdmittedRequestUnsafe(ImageLodRequest&, std::shared_ptr<ImageStreamMemory::Reservation>&);
		bool popNextQueuedRequestUnsafe(ImageLodRequest&);
		void removeQueuedRequestsForImageUnsafe(uint64_t imageId);
		void removeReadyUploadsForImageUnsafe(uint64_t imageId);

		std::shared_ptr<ImageStreamMemory> m_memory;
		ImageVolumeResidency m_residency;
		std::string m_gpuBudgetSource{ "unqueried" };
		bool m_gpuMemoryHintAvailable{ false };
		std::size_t m_availableGpuBytes{ 0 }, m_gpuCapacityCeiling{ 0 };
		std::chrono::steady_clock::time_point m_lastGpuQuery{};
		std::unordered_map<uint64_t, unsigned int> m_inFlight;
		uint32_t m_nextVersion{ 0 };
		Camera* m_camera{ nullptr };
		std::priority_queue<ImageLodRequest> m_requests;
		std::vector<ImageLodReady> m_ready;
		std::unordered_map<uint64_t, ImageLodState> m_states;
		mutable std::mutex m_mutex;
		std::condition_variable m_condition;
		std::vector<std::thread> m_workers;
		bool m_stopWorker{ false };
		bool m_redrawPending{ false };
	};
}

#endif // LodUpdateManager_h__
