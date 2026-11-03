/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "ProgressiveLoadProfile.hpp"
#include <Objects/MyMultipleObject.hpp>
#include <iostream>
#include <algorithm>
#include <thread>

namespace poca::core {
	std::shared_ptr<ProgressiveLoadProfile> ProgressiveLoadProfile::forObject(MyObjectInterface* object)
	{
		auto* multiple=dynamic_cast<MyMultipleObject*>(object);
		auto profile=multiple ? multiple->loadingProfile() : nullptr;
		return profile && profile->recording() ? profile : nullptr;
	}
	double ProgressiveLoadProfile::elapsedMs() const { return std::chrono::duration<double,std::milli>(Clock::now()-m_start).count(); }
	bool ProgressiveLoadProfile::recording() const { std::lock_guard<std::mutex> lock(m_mutex); return !m_reported; }
	void ProgressiveLoadProfile::writeEvent(const char* event, size_t index) const
	{
		std::cout << "[PoCA][Zarr][Pipeline] job=" << this << " t_ms=" << elapsedMs()
			<< " thread=" << std::this_thread::get_id() << " event=" << event;
		if (index != size_t(-1)) std::cout << " index=" << index;
		std::cout << std::endl;
	}
	void ProgressiveLoadProfile::event(const char* name, size_t index) { std::lock_guard<std::mutex> lock(m_mutex); writeEvent(name,index); }
	void ProgressiveLoadProfile::workerBusy(double ms) { std::lock_guard<std::mutex> lock(m_mutex); m_stats.workerBusyMs+=ms; }
	void ProgressiveLoadProfile::queueWait(double ms) { std::lock_guard<std::mutex> lock(m_mutex); m_stats.queueWaitMs+=ms; }
	void ProgressiveLoadProfile::guiCommit(double ms) { std::lock_guard<std::mutex> lock(m_mutex); m_stats.guiCommitMs+=ms; }
	void ProgressiveLoadProfile::beginPipeline() { std::lock_guard<std::mutex> lock(m_mutex); m_pipelineStart=Clock::now(); writeEvent("pipeline_begin",size_t(-1)); }
	void ProgressiveLoadProfile::queueChanged(bool enqueue, size_t index, size_t depth, size_t bytes)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		m_stats.queueDepth=depth; m_stats.queuedBytes=bytes;
		m_stats.peakQueueDepth=(std::max)(m_stats.peakQueueDepth,depth);
		m_stats.peakQueuedBytes=(std::max)(m_stats.peakQueuedBytes,bytes);
		writeEvent(enqueue ? "enqueue" : "dequeue",index);
		std::cout << "[PoCA][Zarr][Pipeline] job=" << this << " queue_depth=" << depth << " accounted_queue_bytes=" << bytes << std::endl;
	}
	void ProgressiveLoadProfile::gpuInitialization(const char* label, double ms, bool trace)
	{
		std::lock_guard<std::mutex> lock(m_mutex); m_stats.gpuInitializationMs+=ms;
		if (trace) {
			writeEvent("gpu_initialization_end",size_t(-1));
			std::cout << "[PoCA][Zarr][Pipeline] job=" << this << " resource=" << label << " gpu_cpu_ms=" << ms << std::endl;
		}
	}
	void ProgressiveLoadProfile::redrawScheduled(const char* source)
	{
		std::lock_guard<std::mutex> lock(m_mutex); ++m_stats.redrawSchedules;
		// Count every schedule; log the first to keep streaming/camera frames compact.
		if (m_stats.redrawSchedules == 1) { writeEvent("redraw_scheduled",size_t(-1)); std::cout << "[PoCA][Zarr][Pipeline] source=" << source << std::endl; }
	}
	void ProgressiveLoadProfile::frameSubmitted()
	{
		std::lock_guard<std::mutex> lock(m_mutex); ++m_stats.frames;
		if (m_stats.frames == 1) writeEvent("first_submitted_frame_not_visibility",size_t(-1));
	}
	void ProgressiveLoadProfile::sceneDrawn(bool complete, double ms)
	{
		std::lock_guard<std::mutex> lock(m_mutex); ++m_stats.sceneDraws;
		m_stats.sceneCpuMs+=ms; m_stats.maxSceneCpuMs=(std::max)(m_stats.maxSceneCpuMs,ms);
		m_completedScene=m_completedScene || complete;
	}
	bool ProgressiveLoadProfile::hasCompletedScene() const { std::lock_guard<std::mutex> lock(m_mutex); return m_completedScene; }
	ProgressiveLoadProfile::Snapshot ProgressiveLoadProfile::snapshot() const { std::lock_guard<std::mutex> lock(m_mutex); return m_stats; }
	void ProgressiveLoadProfile::report(const char* state, bool final)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		if (m_reported) return;
		m_reported=final; writeEvent(state,size_t(-1));
		std::cout << "[PoCA][Zarr][Pipeline] job=" << this << " wall_ms=" << elapsedMs() << " pipeline_ms=";
		if (m_pipelineStart) std::cout << std::chrono::duration<double,std::milli>(Clock::now()-*m_pipelineStart).count();
		else std::cout << "not_started";
		std::cout << " worker_busy_ms=" << m_stats.workerBusyMs << " queue_slot_wait_ms=" << m_stats.queueWaitMs
			<< " gui_commit_ms=" << m_stats.guiCommitMs << " gpu_cpu_ms=" << m_stats.gpuInitializationMs
			<< " peak_queue_depth=" << m_stats.peakQueueDepth << " peak_accounted_queue_bytes=" << m_stats.peakQueuedBytes
			<< " redraw_schedules=" << m_stats.redrawSchedules << " scene_redraws=" << m_stats.sceneDraws << " submitted_frames=" << m_stats.frames
			<< " scene_cpu_ms=" << m_stats.sceneCpuMs << " max_scene_cpu_ms=" << m_stats.maxSceneCpuMs << std::endl;
	}
	ProgressiveLoadProfile::GpuScope::GpuScope(MyObjectInterface* object, const char* label, bool trace)
		: m_profile(forObject(object)), m_label(label), m_trace(trace), m_start(Clock::now())
	{
		if (m_profile && m_trace) m_profile->event("gpu_initialization_begin");
	}
	ProgressiveLoadProfile::GpuScope::~GpuScope()
	{
		if (m_profile) m_profile->gpuInitialization(m_label,std::chrono::duration<double,std::milli>(Clock::now()-m_start).count(),m_trace);
	}
}
