/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      LodUpdateManager.cpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*/

#include "LodUpdateManager.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <QtCore/QMetaObject>

#include <General/Engine.hpp>
#include <General/PerformanceProfiler.hpp>

#include "Camera.hpp"

namespace poca::opengl {
	namespace {
		bool lodDebugEnabled()
		{
			poca::core::Engine* engine = poca::core::Engine::instance();
			if (!engine->verboseEnabled())
				return false;

			/*
			 * LOD/raycasting diagnostics are noisy and should only be printed when
			 * their explicit verbose category is selected. Do not use Engine::verbose()
			 * here because an empty verbose-type list currently means "all verbose",
			 * which made image raycasting logs appear even after all debug categories
			 * were unchecked.
			 */
			return engine->hasVerboseType("debugPyramidalRendering") || engine->hasVerboseType("lodDebug");
		}
	}

	LodUpdateManager::LodUpdateManager(Camera* _camera, std::shared_ptr<ImageStreamMemory> _memory)
		: m_memory(std::move(_memory)), m_camera(_camera)
	{
		if (!m_memory) throw std::invalid_argument("Image stream memory accounting is required");
		unsigned int workerCount = std::thread::hardware_concurrency();
		workerCount = workerCount > 1u ? workerCount - 1u : 1u;
		workerCount = std::max(1u, std::min(4u, workerCount));
		m_workers.reserve(workerCount);
		for (unsigned int n = 0; n < workerCount; ++n)
			m_workers.emplace_back(&LodUpdateManager::workerLoop, this);
	}

	LodUpdateManager::~LodUpdateManager()
	{
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			m_stopWorker = true;
		}
		m_condition.notify_all();
		for (std::thread& worker : m_workers)
			if (worker.joinable())
				worker.join();
	}

	void LodUpdateManager::setCamera(Camera* _camera)
	{
		m_camera = _camera;
	}

	uint32_t LodUpdateManager::request(const ImageLodRequest& _request, uint64_t _frameIndex)
	{
		poca::core::PerformanceProfiler::ScopedTimer timer("LOD queue", "Queue image LOD request");
		std::lock_guard<std::mutex> lock(m_mutex);
		ImageLodState& state = m_states[_request.imageId];

		// Camera generation is diagnostic; versions identify immutable storage targets.
		const bool sameTarget =
			state.requestedLevel == _request.requestedLevel &&
			state.targetDims == _request.targetDims &&
			state.downsampleFactors == _request.downsampleFactors &&
			state.residentSource == _request.residentSource && state.visible == _request.visible && sameRegion(state.sourceRegion, _request.sourceRegion) && state.preview == _request.preview && state.reductionMode == _request.reductionMode && state.currentFrame == _request.currentFrame && state.sourceDims == _request.sourceDims && state.residentBottom == _request.residentBottom && state.residentTop == _request.residentTop && state.imageBottom == _request.imageBottom && state.imageTop == _request.imageTop;

		if (sameTarget && (state.status == LodRequestStatus::Queued || state.status == LodRequestStatus::Preparing || state.status == LodRequestStatus::Ready)) {
			const bool queuedPriorityIncreased = state.status == LodRequestStatus::Queued && _request.priority > state.priority;
			state.priority = std::max(state.priority, _request.priority);
			state.lastVisibleFrame = _frameIndex;
			if (queuedPriorityIncreased) {
				removeQueuedRequestsForImageUnsafe(_request.imageId);
				ImageLodRequest queued = _request;
				queued.requestVersion = state.latestVersion;
				queued.priority = state.priority;
				m_requests.push(std::move(queued));
				m_condition.notify_all();
			}
			if (lodDebugEnabled())
				std::cout << "[PoCA][ImageLOD][queue-skip-same] " << _request << " queueSize=" << m_requests.size() << " readySize=" << m_ready.size() << std::endl;
			return state.latestVersion;
		}

		state.requestedLevel = _request.requestedLevel; state.viewGeneration = _request.viewGeneration;
		state.priority = _request.priority; state.effectiveTarget = _request.effectiveTarget;
		state.targetDims = _request.targetDims;
		state.downsampleFactors = _request.downsampleFactors;
		state.visible = _request.visible;
		state.sourceRegion = _request.sourceRegion; state.preview = _request.preview; state.residentSource = _request.residentSource; state.reductionMode = _request.reductionMode;
		state.currentFrame = _request.currentFrame; state.sourceDims = _request.sourceDims;
		state.residentBottom = _request.residentBottom; state.residentTop = _request.residentTop; state.imageBottom = _request.imageBottom; state.imageTop = _request.imageTop;
		state.lastVisibleFrame = _frameIndex;
		state.latestVersion = ++m_nextVersion;
		state.status = LodRequestStatus::Queued;

		removeQueuedRequestsForImageUnsafe(_request.imageId);
		removeReadyUploadsForImageUnsafe(_request.imageId);

		ImageLodRequest queued = _request;
		queued.requestVersion = state.latestVersion;
		m_requests.push(std::move(queued));
		if (lodDebugEnabled())
			std::cout << "[PoCA][ImageLOD][queue-push] imageID=" << _request.imageId
				<< " level=" << _request.requestedLevel
				<< " version=" << state.latestVersion
				<< " priority=" << _request.priority
				<< " target=" << _request.targetDims.x << "x" << _request.targetDims.y << "x" << _request.targetDims.z
				<< " factors=" << _request.downsampleFactors.x << "x" << _request.downsampleFactors.y << "x" << _request.downsampleFactors.z
				<< " visible=" << (_request.visible ? 1 : 0)
				<< " queueSize=" << m_requests.size()
				<< " readySize=" << m_ready.size()
				<< std::endl;
		m_condition.notify_all();
		//std::cout << "request = " << queued << std::endl;
		return state.latestVersion;
	}

	void LodUpdateManager::invalidateDetail(uint64_t _imageId)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		auto it = m_states.find(_imageId);
		// Whole-image first pixels are camera-independent; do not restart their I/O.
		if (it == m_states.end() || it->second.preview || it->second.status == LodRequestStatus::Idle) return;
		it->second.latestVersion = ++m_nextVersion;
		it->second.status = LodRequestStatus::Idle;
		removeQueuedRequestsForImageUnsafe(_imageId);
		removeReadyUploadsForImageUnsafe(_imageId);
		if (lodDebugEnabled()) std::cout << "[PoCA][ImageStream][schedule] action=discarded image=" << _imageId << std::endl;
	}

	void LodUpdateManager::cancel(uint64_t _imageId)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		auto it = m_states.find(_imageId);
		if (it == m_states.end())
			return;
		if (it->second.status == LodRequestStatus::Idle)
			return;

		it->second.latestVersion = ++m_nextVersion;
		it->second.status = LodRequestStatus::Idle;
		removeQueuedRequestsForImageUnsafe(_imageId);
		removeReadyUploadsForImageUnsafe(_imageId);
	}

	void LodUpdateManager::clear()
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		while (!m_requests.empty())
			m_requests.pop();
		m_ready.clear();
		m_states.clear();
	}

	bool LodUpdateManager::hasQueuedRequests() const
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		return !m_requests.empty();
	}

	bool LodUpdateManager::hasReadyUploads() const
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		return !m_ready.empty();
	}

	bool LodUpdateManager::popNextQueuedRequest(ImageLodRequest& _request)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		return popNextQueuedRequestUnsafe(_request);
	}

	bool LodUpdateManager::popNextQueuedRequestUnsafe(ImageLodRequest& _request)
	{
		while (!m_requests.empty()) {
			const ImageLodRequest request = m_requests.top();
			m_requests.pop();

			auto it = m_states.find(request.imageId);
			if (it == m_states.end())
				continue;
			if (request.requestVersion != it->second.latestVersion)
				continue;

			_request = request;
			if (lodDebugEnabled())
				std::cout << "[PoCA][ImageLOD][queue-pop] " << _request << " remainingQueue=" << m_requests.size() << std::endl;
			return true;
		}

		return false;
	}

	std::vector<ImageLodRequest> LodUpdateManager::drainQueuedRequests()
	{
		std::vector<ImageLodRequest> requests;
		ImageLodRequest request;
		while (popNextQueuedRequest(request)) {
			requests.push_back(request);
		}
		return requests;
	}

	std::vector<ImageLodReady> LodUpdateManager::drainReadyUploads(std::size_t _maxUploads, std::size_t _maxPreparedBytes)
	{
		poca::core::PerformanceProfiler::ScopedTimer timer("LOD queue", "Drain ready LOD uploads");
		std::lock_guard<std::mutex> lock(m_mutex);
		m_ready.erase(std::remove_if(m_ready.begin(), m_ready.end(),
			[this](const ImageLodReady& _ready) {
				auto it = m_states.find(_ready.imageId);
				return it == m_states.end() || _ready.requestVersion != it->second.latestVersion;
			}),
			m_ready.end());
		std::stable_sort(m_ready.begin(), m_ready.end(),
			[this](const ImageLodReady& _a, const ImageLodReady& _b) {
				const auto stateA = m_states.find(_a.imageId);
				const auto stateB = m_states.find(_b.imageId);
				const float priorityA = stateA != m_states.end() ? stateA->second.priority : 0.f;
				const float priorityB = stateB != m_states.end() ? stateB->second.priority : 0.f;
				const uint64_t frameA = stateA != m_states.end() ? stateA->second.lastVisibleFrame : 0u;
				const uint64_t frameB = stateB != m_states.end() ? stateB->second.lastVisibleFrame : 0u;
				if (_a.visible != _b.visible)
					return _a.visible && !_b.visible;
				if (priorityA != priorityB)
					return priorityA > priorityB;
				return frameA > frameB;
			});
		std::vector<ImageLodReady> ready;
		if ((_maxUploads == 0 || _maxUploads >= m_ready.size()) && _maxPreparedBytes == 0) {
			ready.swap(m_ready);
			if (lodDebugEnabled() && !ready.empty())
				std::cout << "[PoCA][ImageLOD][ready-drain] count=" << ready.size() << " remainingReady=" << m_ready.size() << std::endl;
			return ready;
		}

		auto preparedBytes = [](const ImageLodReady& _ready) -> std::size_t {
			return _ready.preparedBytes;
		};

		std::size_t drainCount = 0;
		std::size_t drainedBytes = 0;
		while (drainCount < m_ready.size()) {
			if (_maxUploads != 0 && drainCount >= _maxUploads)
				break;
			const std::size_t nextBytes = preparedBytes(m_ready[drainCount]);
			if (_maxPreparedBytes != 0 && drainCount > 0 && nextBytes > _maxPreparedBytes - std::min(drainedBytes, _maxPreparedBytes))
				break;
			drainedBytes = streamAdd(drainedBytes, nextBytes);
			++drainCount;
		}
		if (drainCount == 0 && !m_ready.empty())
			drainCount = 1;

		ready.insert(ready.end(), m_ready.begin(), m_ready.begin() + drainCount);
		m_ready.erase(m_ready.begin(), m_ready.begin() + drainCount);
		if (lodDebugEnabled() && !ready.empty())
			std::cout << "[PoCA][ImageLOD][ready-drain] count=" << ready.size() << " remainingReady=" << m_ready.size() << std::endl;
		return ready;
	}

	void LodUpdateManager::markPreparing(uint64_t _imageId, uint32_t _requestVersion)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		auto it = m_states.find(_imageId);
		if (it == m_states.end())
			return;
		if (_requestVersion != it->second.latestVersion)
			return;

		it->second.status = LodRequestStatus::Preparing;
	}

	void LodUpdateManager::markReady(const ImageLodReady& _ready)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		auto it = m_states.find(_ready.imageId);
		if (it == m_states.end())
			return;
		if (_ready.requestVersion != it->second.latestVersion)
			return;

		if (_ready.payload && !_ready.memoryReservation) throw std::logic_error("Ready image stream payload must own its CPU reservation");
		it->second.status = LodRequestStatus::Ready;
		m_ready.push_back(_ready);
		if (lodDebugEnabled())
			std::cout << "[PoCA][ImageLOD][ready-push] " << _ready << " readySize=" << m_ready.size() << std::endl;
	}

	void LodUpdateManager::markUploaded(uint64_t _imageId, uint32_t _displayedLevel, uint32_t _version)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		auto it = m_states.find(_imageId);
		if (it == m_states.end())
			return;

		if (_version && it->second.latestVersion != _version) return;
		it->second.currentDisplayedLevel = _displayedLevel;
		it->second.status = LodRequestStatus::Idle;

		m_ready.erase(std::remove_if(m_ready.begin(), m_ready.end(),
			[_imageId](const ImageLodReady& _ready) { return _ready.imageId == _imageId; }),
			m_ready.end());
	}

	bool LodUpdateManager::state(uint64_t _imageId, ImageLodState& _outState) const
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		auto it = m_states.find(_imageId);
		if (it == m_states.end())
			return false;
		_outState = it->second;
		return true;
	}

	void LodUpdateManager::removeQueuedRequestsForImageUnsafe(uint64_t _imageId)
	{
		if (m_requests.empty())
			return;

		std::priority_queue<ImageLodRequest> compacted;
		while (!m_requests.empty()) {
			ImageLodRequest request = m_requests.top();
			m_requests.pop();
			if (request.imageId == _imageId)
				continue;

			auto it = m_states.find(request.imageId);
			if (it == m_states.end())
				continue;
			if (request.requestVersion != it->second.latestVersion)
				continue;
			compacted.push(std::move(request));
		}
		m_requests.swap(compacted);
	}

	void LodUpdateManager::removeReadyUploadsForImageUnsafe(uint64_t _imageId)
	{
		m_ready.erase(std::remove_if(m_ready.begin(), m_ready.end(),
			[_imageId](const ImageLodReady& _ready) { return _ready.imageId == _imageId; }),
			m_ready.end());
	}

	void LodUpdateManager::cancelInvisible()
	{
		std::vector<uint64_t> ids;
		{ std::lock_guard<std::mutex> lock(m_mutex); for (const auto& state : m_states)
			if (!m_residency.visible(state.first)) ids.push_back(state.first); }
		for (auto id : ids) cancel(id);
	}
	void LodUpdateManager::forget(uint64_t _imageId)
	{
		cancel(_imageId);
		std::unique_lock<std::mutex> lock(m_mutex);
		m_condition.wait(lock, [this, _imageId]() { return m_inFlight.find(_imageId) == m_inFlight.end(); });
		m_states.erase(_imageId);
	}

	bool LodUpdateManager::isCurrent(const ImageLodReady& _ready) const
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		auto it = m_states.find(_ready.imageId);
		return !_ready.obsolete && it != m_states.end() && it->second.latestVersion == _ready.requestVersion;
	}

	bool LodUpdateManager::uploadReadyForFrame()
	{
		bool uploaded = false;
		const auto readyUploads = drainReadyUploads(4, imageStreamPolicy().uploadBytes);
		for (const auto& ready : readyUploads) {
			if (!isCurrent(ready) || !m_residency.visible(ready.imageId)) continue;
			if (!ready.uploadCallback) throw std::logic_error("Image stream has no render-thread upload callback");
			const bool changed = ready.uploadCallback(ready);
			ImageLodState current;
			if (state(ready.imageId, current))
				markUploaded(ready.imageId, changed ? ready.requestedLevel : current.currentDisplayedLevel, ready.requestVersion);
			uploaded = uploaded || changed;
		}
		return uploaded;
	}

	bool LodUpdateManager::takeAdmittedRequestUnsafe(ImageLodRequest& _request, std::shared_ptr<ImageStreamMemory::Reservation>& _reservation)
	{
		std::vector<ImageLodRequest> deferred;
		bool found = false;
		while (!m_requests.empty()) {
			auto next = m_requests.top(); m_requests.pop();
			auto state = m_states.find(next.imageId);
			if (state == m_states.end() || state->second.latestVersion != next.requestVersion) continue;
			if (!next.estimatedPreparedBytes || next.estimatedPreparedBytes > imageStreamPolicy().cpuBytes) {
				state->second.status = LodRequestStatus::Idle;
				std::cerr << "[PoCA][ImageStream] Rejected unbounded CPU preparation" << std::endl;
				continue;
			}
			// A canceled reader can still be inside an uninterruptible call. Keep only
			// its latest queued successor, and let other images obtain first pixels.
			if (m_inFlight.find(next.imageId) == m_inFlight.end()) _reservation = m_memory->reserve(next.estimatedPreparedBytes);
			if (_reservation) { _request = std::move(next); found = true; break; }
			deferred.push_back(std::move(next));
		}
		for (auto& queued : deferred) m_requests.push(std::move(queued));
		return found;
	}

	void LodUpdateManager::workerLoop()
	{
		for (;;) {
			ImageLodRequest request;
			std::shared_ptr<ImageStreamMemory::Reservation> reservation;
			{
				std::unique_lock<std::mutex> lock(m_mutex);
				// A timed wake also observes reservations released by drained payload owners.
				m_condition.wait_for(lock, std::chrono::milliseconds(25), [this]() { return m_stopWorker || !m_requests.empty(); });
				if (m_stopWorker) return;
				if (m_requests.empty()) continue;
				if (!takeAdmittedRequestUnsafe(request, reservation)) { m_condition.wait_for(lock, std::chrono::milliseconds(25)); continue; }
				m_states.at(request.imageId).status = LodRequestStatus::Preparing;
				++m_inFlight[request.imageId];
			}
			request.canceled = [this, id = request.imageId, version = request.requestVersion]() {
				std::lock_guard<std::mutex> lock(m_mutex);
				auto it = m_states.find(id); return m_stopWorker || it == m_states.end() || it->second.latestVersion != version;
			};
			ImageLodReady ready;
			ready.imageId = request.imageId; ready.requestedLevel = request.requestedLevel;
			ready.requestVersion = request.requestVersion; ready.viewGeneration = request.viewGeneration; ready.visible = request.visible;
			ready.uploadCallback = request.uploadCallback;
			bool success = false;
			try {
				if (request.prepareCallback) success = request.prepareCallback(request, ready);
				if (success) {
					reservation->ready(ready.preparedBytes);
					ready.memoryReservation = std::move(reservation);
				}
			}
			catch (const ImageStreamCanceled&) { ready.obsolete = true; }
			catch (const std::exception& error) {
				std::cerr << "[PoCA][ImageStream][read] image=" << request.imageId << " level=" << request.requestedLevel << ": " << error.what() << std::endl;
			}
			{
				std::lock_guard<std::mutex> lock(m_mutex);
				auto state = m_states.find(request.imageId);
				const bool current = state != m_states.end() && state->second.latestVersion == request.requestVersion;
				if (!current || !success || ready.obsolete) {
					// Release obsolete payload/scratch BEFORE declaring this reader finished.
					// forget() and its queued successor then see balanced CPU accounting.
					ready.payload.reset(); ready.memoryReservation.reset(); reservation.reset(); success = false;
				}
				if (current) {
					state->second.status = success ? LodRequestStatus::Ready : LodRequestStatus::Idle;
					if (success) m_ready.push_back(std::move(ready));
				}
				auto flight = m_inFlight.find(request.imageId);
				if (--flight->second == 0) m_inFlight.erase(flight);
				m_condition.notify_all();
				if (m_camera != nullptr) QMetaObject::invokeMethod(m_camera, "update", Qt::QueuedConnection);
			}
		}
	}
	std::ostream& operator<<(std::ostream& _os, const ImageLodRequest& _ilr)
	{
		return _os << "ImageLodRequest, imageID=" <<_ilr.imageId << ", requestedLevel=" << _ilr.requestedLevel
			<< ", requestVersion=" << _ilr.requestVersion << ", priority=" << _ilr.priority << ", targetDim=" << glm::to_string(_ilr.targetDims)
			<< ", downsampleFactors=" << glm::to_string(_ilr.downsampleFactors);
	}

	std::ostream& operator<<(std::ostream& _os, const ImageLodReady& _ilr)
	{
		return _os << "ImageLodReady, imageID=" << _ilr.imageId << ", requestedLevel=" << _ilr.requestedLevel
			<< ", requestVersion=" << _ilr.requestVersion << ", obsolete=" << _ilr.obsolete << ", preparedDims=" << glm::to_string(_ilr.preparedDims);
	}

	std::ostream& operator<<(std::ostream& _os, const ImageLodState& _ils)
	{
		return _os << "ImageLodState, currentDisplayedLevel=" << _ils.currentDisplayedLevel << ", requestedLevel=" << _ils.requestedLevel
			<< ", latestVersion=" << _ils.latestVersion << ", priority=" << _ils.priority << ", targetDims=" << glm::to_string(_ils.targetDims)
			<< ", downsampleFactors=" << glm::to_string(_ils.downsampleFactors)
			<< ", visible=" << _ils.visible;
	}

	std::ostream& operator<<(std::ostream& _os, const LodUpdateManager& _lum)
	{
		return _os << _lum.m_requests.size();
	}
}
