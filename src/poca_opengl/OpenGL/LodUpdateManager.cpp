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
#include <iostream>
#include <QtCore/QMetaObject>

#include <General/Engine.hpp>

#include "Camera.hpp"

namespace poca::opengl {
	namespace {
		bool lodDebugEnabled()
		{
			poca::core::Engine* engine = poca::core::Engine::instance();
			return engine->verbose("debugPyramidalRendering") || engine->verbose("lodDebug");
		}
	}

	LodUpdateManager::LodUpdateManager(Camera* _camera)
		: m_camera(_camera)
	{
		m_worker = std::thread(&LodUpdateManager::workerLoop, this);
	}

	LodUpdateManager::~LodUpdateManager()
	{
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			m_stopWorker = true;
		}
		m_condition.notify_all();
		if (m_worker.joinable())
			m_worker.join();
	}

	void LodUpdateManager::setCamera(Camera* _camera)
	{
		m_camera = _camera;
	}

	uint32_t LodUpdateManager::request(const ImageLodRequest& _request, uint64_t _frameIndex)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		ImageLodState& state = m_states[_request.imageId];

		const bool sameTarget =
			state.requestedLevel == _request.requestedLevel &&
			state.targetDims == _request.targetDims &&
			state.visible == _request.visible;

		if (sameTarget && (state.status == LodRequestStatus::Queued || state.status == LodRequestStatus::Preparing || state.status == LodRequestStatus::Ready)) {
			state.priority = std::max(state.priority, _request.priority);
			state.lastVisibleFrame = _frameIndex;
			if (lodDebugEnabled())
				std::cout << "[PoCA][ImageLOD][queue-skip-same] " << _request << " queueSize=" << m_requests.size() << " readySize=" << m_ready.size() << std::endl;
			return state.latestVersion;
		}

		state.requestedLevel = _request.requestedLevel;
		state.priority = _request.priority;
		state.targetDims = _request.targetDims;
		state.visible = _request.visible;
		state.lastVisibleFrame = _frameIndex;
		state.latestVersion++;
		state.status = LodRequestStatus::Queued;

		ImageLodRequest queued = _request;
		queued.requestVersion = state.latestVersion;
		m_requests.push(std::move(queued));
		if (lodDebugEnabled())
			std::cout << "[PoCA][ImageLOD][queue-push] imageID=" << _request.imageId
				<< " level=" << _request.requestedLevel
				<< " version=" << state.latestVersion
				<< " priority=" << _request.priority
				<< " target=" << _request.targetDims.x << "x" << _request.targetDims.y << "x" << _request.targetDims.z
				<< " visible=" << (_request.visible ? 1 : 0)
				<< " queueSize=" << m_requests.size()
				<< " readySize=" << m_ready.size()
				<< std::endl;
		m_condition.notify_one();
		//std::cout << "request = " << queued << std::endl;
		return state.latestVersion;
	}

	void LodUpdateManager::cancel(uint64_t _imageId)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		auto it = m_states.find(_imageId);
		if (it == m_states.end())
			return;

		it->second.latestVersion++;
		it->second.status = LodRequestStatus::Idle;
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

	std::vector<ImageLodReady> LodUpdateManager::drainReadyUploads(std::size_t _maxUploads)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		std::vector<ImageLodReady> ready;
		if (_maxUploads == 0 || _maxUploads >= m_ready.size()) {
			ready.swap(m_ready);
			if (lodDebugEnabled() && !ready.empty())
				std::cout << "[PoCA][ImageLOD][ready-drain] count=" << ready.size() << " remainingReady=" << m_ready.size() << std::endl;
			return ready;
		}

		ready.insert(ready.end(), m_ready.begin(), m_ready.begin() + _maxUploads);
		m_ready.erase(m_ready.begin(), m_ready.begin() + _maxUploads);
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

		it->second.status = LodRequestStatus::Ready;
		m_ready.push_back(_ready);
		if (lodDebugEnabled())
			std::cout << "[PoCA][ImageLOD][ready-push] " << _ready << " readySize=" << m_ready.size() << std::endl;
	}

	void LodUpdateManager::markUploaded(uint64_t _imageId, uint32_t _displayedLevel)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		auto it = m_states.find(_imageId);
		if (it == m_states.end())
			return;

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

	void LodUpdateManager::workerLoop()
	{
		for (;;) {
			ImageLodRequest request;
			{
				std::unique_lock<std::mutex> lock(m_mutex);
				m_condition.wait(lock, [this]() { return m_stopWorker || !m_requests.empty(); });
				if (m_stopWorker)
					return;
				if (!popNextQueuedRequestUnsafe(request))
					continue;

				auto it = m_states.find(request.imageId);
				if (it == m_states.end())
					continue;
				if (request.requestVersion != it->second.latestVersion)
					continue;
				it->second.status = LodRequestStatus::Preparing;
			}

			ImageLodReady ready;
			ready.imageId = request.imageId;
			ready.requestedLevel = request.requestedLevel;
			ready.requestVersion = request.requestVersion;
			ready.preparedDims = request.targetDims;
			ready.visible = request.visible;
			if (request.prepareCallback && !request.prepareCallback(request, ready))
				continue;

			{
				std::lock_guard<std::mutex> lock(m_mutex);
				auto it = m_states.find(ready.imageId);
				if (it == m_states.end())
					continue;
				if (ready.requestVersion != it->second.latestVersion)
					continue;

				it->second.status = LodRequestStatus::Ready;
				m_ready.push_back(ready);
				if (lodDebugEnabled())
					std::cout << "[PoCA][ImageLOD][worker-ready] " << ready << " readySize=" << m_ready.size() << std::endl;
			}

			if (m_camera != nullptr)
				QMetaObject::invokeMethod(m_camera, "update", Qt::QueuedConnection);
			//std::cout << "LodUpdateManager::workerLoop - " << m_requests.size() << std::endl;
		}
	}


	std::ostream& operator<<(std::ostream& _os, const ImageLodRequest& _ilr)
	{
		return _os << "ImageLodRequest, imageID=" <<_ilr.imageId << ", requestedLevel=" << _ilr.requestedLevel
			<< ", requestVersion=" << _ilr.requestVersion << ", priority=" << _ilr.priority << ", targetDim=" << glm::to_string(_ilr.targetDims);
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
			<< ", visible=" << _ils.visible;
	}

	std::ostream& operator<<(std::ostream& _os, const LodUpdateManager& _lum)
	{
		return _os << _lum.m_requests.size();
	}
}
