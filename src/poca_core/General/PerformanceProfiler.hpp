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
#include <map>
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

		static PerformanceProfiler& instance();

		void setEnabled(bool);
		bool enabled() const;
		void record(const std::string&, const std::string&, double);
		std::vector<Sample> samplesSince(uint64_t) const;
		std::map<std::string, double> latestByCategory() const;
		void clear();

	private:
		PerformanceProfiler() = default;

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
