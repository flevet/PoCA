/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#pragma once
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <cstdint>

namespace poca::core {
	class MyObjectInterface;
	// One opening, shared by the producer, GUI and existing renderers. No GL ownership.
	class ProgressiveLoadProfile {
	public:
		using Clock = std::chrono::steady_clock;
		struct Snapshot {
			double workerBusyMs{}, queueWaitMs{}, guiCommitMs{}, gpuInitializationMs{};
			double sceneCpuMs{}, maxSceneCpuMs{};
			size_t queueDepth{}, peakQueueDepth{}, queuedBytes{}, peakQueuedBytes{};
			uint64_t redrawSchedules{}, frames{}, sceneDraws{};
		};
		class GpuScope {
		public:
			GpuScope(MyObjectInterface*, const char*, bool = true);
			~GpuScope();
			GpuScope(const GpuScope&) = delete;
			GpuScope& operator=(const GpuScope&) = delete;
		private:
			std::shared_ptr<ProgressiveLoadProfile> m_profile;
			const char* m_label;
			bool m_trace;
			Clock::time_point m_start;
		};
		static std::shared_ptr<ProgressiveLoadProfile> forObject(MyObjectInterface*);
		void event(const char*, size_t = size_t(-1));
		void workerBusy(double);
		void queueWait(double);
		void queueChanged(bool, size_t, size_t, size_t);
		void guiCommit(double);
		void gpuInitialization(const char*, double, bool);
		void redrawScheduled(const char*);
		void frameSubmitted();
		void sceneDrawn(bool, double);
		bool hasCompletedScene() const;
		void beginPipeline();
		void report(const char*, bool = true);
		Snapshot snapshot() const;
		double elapsedMs() const;
		bool recording() const;
	private:
		void writeEvent(const char*, size_t) const;
		const Clock::time_point m_start{Clock::now()};
		std::optional<Clock::time_point> m_pipelineStart;
		mutable std::mutex m_mutex;
		Snapshot m_stats;
		bool m_reported{false}, m_completedScene{false};
	};
}
