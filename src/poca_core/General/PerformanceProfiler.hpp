/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PerformanceProfiler.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef PerformanceProfiler_hpp__
#define PerformanceProfiler_hpp__

#include <chrono>
#include <array>
#include <atomic>
#include <thread>
#include <cstdint>
#include <deque>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace poca::core {

	class PerformanceProfiler {
	public:
		struct Sample {
			uint64_t sequence{ 0 };
			std::string category;
			std::string name;
			double milliseconds{ 0.0 };
		};

		class ScopedTimer {
		public:
			ScopedTimer(const std::string&, const std::string&);
			~ScopedTimer();

		private:
			std::string m_category;
			std::string m_name;
			std::chrono::high_resolution_clock::time_point m_start;
		};

		// Bounded, optional image diagnostics shared by the executable and plugins.
		enum class ImageMetric : size_t {
			Displays, DisplayUs, Batches, GeometryAllocations, DescriptorAllocations, DescriptorUploads,
			VisibilityHits, VisibilityMisses, VisibilityUs, PickingPasses, PickingUs,
			UploadFrames, Uploads, UploadUs, PreviewPreparations, NavigationPreparations, DetailPreparations,
			PreparationUs, Completions, Failures, RefinementsAvoided, DetailRetirements,
			RedrawRequests, RedrawScheduled, StreamingFrames,
			TiffPreviewSessions, TiffNavigationSessions, TiffDetailSessions, TiffPlanes, TiffStackEquivalents, Count
		};
		class ImageTimer {
		public:
			explicit ImageTimer(ImageMetric, int preparationRole = -1);
			~ImageTimer();
			ImageTimer(const ImageTimer&) = delete;
			ImageTimer& operator=(const ImageTimer&) = delete;
		private:
			PerformanceProfiler* m_profiler{ nullptr };
			ImageMetric m_metric;
			int m_role;
			std::chrono::steady_clock::time_point m_start;
		};
		void setImageMetricsEnabled(bool);
		bool imageMetricsEnabled() const { return m_imageMetricsEnabled.load(std::memory_order_relaxed); }
		void addImageMetric(ImageMetric, uint64_t value = 1);
		void recordImageTiffRead(uint64_t planes, uint64_t stackEquivalents, bool firstRead);
		void reportImageMetrics(size_t queued, size_t ready, size_t preparing);

		// Render-thread diagnostics. Records/scopes are shared through instance(), never DLL-local TLS.
		enum class RenderMetric : size_t {
			PaintGLCalls, DrawElementsOnscreen, DrawElementsOffscreen, RecomputeFrameCalls, ChildFrameRecomputes,
			GridFaceDraws, GridLineDraws, BboxDraws, RawVolumeSingle, RawVolumeMulti, NonrawVolumeDraws,
			OtherAuditedDraws, PickingDraws, DescriptorDataCalls, DescriptorDataBytes, DescriptorSubDataCalls, DescriptorSubDataBytes,
			FrameDataCalls, FrameDataBytes, OverlapPairTests, OverlapCacheHits, OverlapCacheMisses, BatchCount, BatchMaxImages,
			VisibleImages, CulledImages, ChildDispatches, PickingFboCreates, PickingFboClears,
			VisibilityCalls, VisibilityHits, VisibilityMisses, ImageShareCalls, ResidencyEntriesExamined,
			LodRequestAttempts, LodRequestsEnqueued, LodRequestsDeduplicated, LodPriorityRequeues, LodContinuationsEnqueued,
			ReadyUploadsProcessed, Texture3DAllocations, Texture3DChunks, Texture3DBytes, StreamRedrawRequests, StreamRedrawScheduled,
			DepthClears, FrameCacheHits, FrameCacheMisses, FrameCacheEntries, FrameCacheBytes,
			MultiFallbackIncompatible, MultiFallbackOverlays, MultiFallbackLimit,
			DrawElementsUs, GridPreparationUs, GridDrawUs, ChildDispatchUs, VisibilityUs, LodPlanningUs,
			ImagesListUs, MultiDisplayUs, OverlapUs, TextureUploadUs, PickingUs, ScenePreparationUs, ObjectDisplayUs, LodRequestsDeferred, Texture3DSubmittedBytes, FrameStackSpills, Count
		};
		enum class RenderWorkerMetric : size_t { Preparations, PreparationUs, AdmissionDeferred, AdmissionRejected, RedrawRequests, Count };
		struct RenderRecord {
			uint64_t frameId{ 0 }, cameraId{ 0 }, children{ 0 };
			int width{ 0 }, height{ 0 };
			bool paint{ false }, offscreen{ false }, multiple{ false }, interactive{ false };
			double cpuMs{ 0. };
			int64_t queued{ -1 }, ready{ -1 }, preparing{ -1 };
			std::array<uint64_t, size_t(RenderMetric::Count)> values{};
			std::array<bool, size_t(RenderMetric::Count)> activeTimers{}; // Same-metric nested scopes count only their outer interval.
		};
		class RenderFrame {
		public:
			RenderFrame(uint64_t camera, bool paint, bool offscreen, bool multiple, uint64_t children, int width, int height, bool interactive);
			~RenderFrame();
			RenderFrame(const RenderFrame&) = delete;
			RenderFrame& operator=(const RenderFrame&) = delete;
		private:
			PerformanceProfiler* m_profiler{ nullptr };
			RenderRecord* m_record{ nullptr };
			RenderRecord* m_parent{ nullptr };
			std::unique_ptr<RenderRecord> m_overflowRecord; // Only for >16 simultaneous nested frames.
			std::chrono::steady_clock::time_point m_start;
		};
		class RenderTimer {
		public:
			explicit RenderTimer(RenderMetric, bool active = true);
			~RenderTimer();
			void stop();
			RenderTimer(const RenderTimer&) = delete;
			RenderTimer& operator=(const RenderTimer&) = delete;
		private:
			PerformanceProfiler* m_betweenProfiler{ nullptr };
			RenderRecord* m_record{ nullptr };
			RenderMetric m_metric;
			std::chrono::steady_clock::time_point m_start;
		};
		void setRenderMetricsEnabled(bool); // GUI thread, before starting a frame.
		bool renderMetricsEnabled() const { return m_renderEnabled.load(std::memory_order_acquire); }
		static PerformanceProfiler& renderInstance(); // Cached pointer to the existing process-wide instance.
		uint64_t allocateRenderIdentity() { return ++m_renderIdentitySequence; }
		static void renderCount(RenderMetric, uint64_t value = 1);
		static void renderMaximum(RenderMetric, uint64_t);
		static void renderWorkerCount(RenderWorkerMetric, uint64_t value = 1);
		void renderQueues(size_t queued, size_t ready, size_t preparing);
		std::vector<RenderRecord> renderRecords() const; // GUI-thread snapshot, bounded to 512.
		static std::string renderCsvHeader();
		static std::string renderCsvRow(const RenderRecord&);
		static const char* renderMetricName(RenderMetric);

		static PerformanceProfiler& instance();

		void setEnabled(bool);
		bool enabled() const;
		void record(const std::string&, const std::string&, double);
		std::vector<Sample> samplesSince(uint64_t) const;
		std::map<std::string, double> latestByCategory() const;
		void clear();

	private:
		PerformanceProfiler() = default;
		~PerformanceProfiler();
		void finishRenderRecord(const RenderRecord&);
		void reportRenderMetrics();
		void stopRenderCsv();
		void renderCsvLoop(const std::string&);
		std::atomic<bool> m_renderEnabled{ false };
		std::thread::id m_renderThread;
		RenderRecord* m_activeRender{ nullptr };
		std::array<RenderRecord, 16> m_renderStack{};
		size_t m_renderDepth{ 0 };
		uint64_t m_renderSequence{ 0 };
		std::atomic<uint64_t> m_renderIdentitySequence{ 0 };
		std::array<RenderRecord, 512> m_renderRing{};
		size_t m_renderRingNext{ 0 }, m_renderRingSize{ 0 }, m_renderWindowFrames{ 0 };
		std::array<uint64_t, size_t(RenderMetric::Count)> m_renderTotals{}, m_renderBetweenFrames{};
		std::array<bool, size_t(RenderMetric::Count)> m_betweenTimers{};
		std::array<std::atomic<uint64_t>, size_t(RenderWorkerMetric::Count)> m_renderWorkers{};
		std::chrono::steady_clock::time_point m_renderReportStart;
		std::mutex m_renderCsvMutex;
		std::condition_variable m_renderCsvCondition;
		std::thread m_renderCsvThread;
		std::array<RenderRecord, 256> m_renderCsvQueue{};
		size_t m_renderCsvHead{ 0 }, m_renderCsvSize{ 0 };
		bool m_renderCsvStop{ false }, m_renderCsvActive{ false };
		std::atomic<uint64_t> m_renderCsvDropped{ 0 };

	private:
		mutable std::mutex m_mutex;
		std::deque<Sample> m_samples;
		uint64_t m_sequence{ 0 };
		bool m_enabled{ true };
		std::atomic<bool> m_imageMetricsEnabled{ false };
		std::array<std::atomic<uint64_t>, size_t(ImageMetric::Count)> m_imageMetrics{};
		std::map<std::thread::id, int> m_imagePreparationRoles; // Only used while diagnostics are enabled.
		std::chrono::steady_clock::time_point m_imageReportStart;
		inline static constexpr size_t m_maxSamples = 4096;
	};
}

#endif
