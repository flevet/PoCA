/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PerformanceProfiler.cpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#include "PerformanceProfiler.hpp"

#include <iostream>
#include <sstream>
#include <QtCore/QCoreApplication>
#include <QtCore/QVariant>

namespace poca::core {

	PerformanceProfiler::ScopedTimer::ScopedTimer(const std::string& _category, const std::string& _name)
		: m_category(_category), m_name(_name), m_start(std::chrono::high_resolution_clock::now())
	{
	}

	PerformanceProfiler::ScopedTimer::~ScopedTimer()
	{
		auto end = std::chrono::high_resolution_clock::now();
		double ms = std::chrono::duration<double, std::milli>(end - m_start).count();
		PerformanceProfiler::instance().record(m_category, m_name, ms);
	}

	PerformanceProfiler::ImageTimer::ImageTimer(ImageMetric metric, int preparationRole) : m_metric(metric), m_role(preparationRole)
	{
		auto& profiler = PerformanceProfiler::instance();
		if (!profiler.imageMetricsEnabled()) return;
		m_profiler = &profiler; m_start = std::chrono::steady_clock::now();
		if (m_role >= 0) {
			std::lock_guard<std::mutex> lock(profiler.m_mutex);
			profiler.m_imagePreparationRoles[std::this_thread::get_id()] = m_role;
		}
	}

	PerformanceProfiler::ImageTimer::~ImageTimer()
	{
		if (!m_profiler) return;
		m_profiler->addImageMetric(m_metric, uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_start).count()));
		if (m_role >= 0) {
			std::lock_guard<std::mutex> lock(m_profiler->m_mutex);
			m_profiler->m_imagePreparationRoles.erase(std::this_thread::get_id());
		}
	}

	void PerformanceProfiler::setImageMetricsEnabled(bool enabled)
	{
		if (m_imageMetricsEnabled.exchange(enabled, std::memory_order_relaxed) == enabled) return;
		for (auto& value : m_imageMetrics) value.store(0, std::memory_order_relaxed);
		m_imageReportStart = std::chrono::steady_clock::now();
	}

	void PerformanceProfiler::addImageMetric(ImageMetric metric, uint64_t value)
	{
		if (imageMetricsEnabled()) m_imageMetrics[size_t(metric)].fetch_add(value, std::memory_order_relaxed);
	}

	void PerformanceProfiler::recordImageTiffRead(uint64_t planes, uint64_t stackEquivalents, bool firstRead)
	{
		if (!imageMetricsEnabled()) return;
		addImageMetric(ImageMetric::TiffPlanes, planes); addImageMetric(ImageMetric::TiffStackEquivalents, stackEquivalents);
		if (!firstRead) return;
		std::lock_guard<std::mutex> lock(m_mutex);
		const auto role = m_imagePreparationRoles.find(std::this_thread::get_id());
		if (role == m_imagePreparationRoles.end()) return; // Scientific/opening reads have no display role.
		if (role->second == 0) addImageMetric(ImageMetric::TiffPreviewSessions);
		else if (role->second == 1) addImageMetric(ImageMetric::TiffNavigationSessions);
		else if (role->second == 2) addImageMetric(ImageMetric::TiffDetailSessions);
	}

	void PerformanceProfiler::reportImageMetrics(size_t queued, size_t ready, size_t preparing)
	{
		if (!imageMetricsEnabled()) return;
		const auto now = std::chrono::steady_clock::now();
		const double seconds = std::chrono::duration<double>(now - m_imageReportStart).count();
		if (seconds < 1.) return;
		m_imageReportStart = now;
		static const std::array<const char*, size_t(ImageMetric::Count)> names{
			"displays", "display_us", "batches", "geometry_allocations", "descriptor_allocations", "descriptor_uploads",
			"visibility_hits", "visibility_misses", "visibility_us", "picking_passes", "picking_us",
			"upload_frames", "uploads", "upload_us", "preview_preparations", "navigation_preparations", "detail_preparations",
			"preparation_us", "completions", "failures", "refinements_avoided", "detail_retirements",
			"redraw_requests", "redraw_scheduled", "streaming_frames",
			"tiff_preview_sessions", "tiff_navigation_sessions", "tiff_detail_sessions", "tiff_planes", "tiff_stack_equivalents"
		};
		std::ostringstream line;
		line << "[PoCA][ImagePerformance] seconds=" << seconds;
		for (size_t i = 0; i < names.size(); ++i) line << ' ' << names[i] << '=' << m_imageMetrics[i].exchange(0, std::memory_order_relaxed);
		line << " queued=" << queued << " ready=" << ready << " preparing=" << preparing;
		std::cout << line.str() << std::endl;
	}

	PerformanceProfiler& PerformanceProfiler::instance()
	{
		/*
		 * poca_core is currently built as a static library and is linked both by the
		 * main application and by plugins. A function-local static singleton would
		 * therefore be duplicated once per module, so samples recorded from plugins
		 * would not be visible from the PerformanceWidget living in the application.
		 *
		 * Store the profiler object pointer on the unique QCoreApplication instance,
		 * which is shared process-wide through QtCore, and let all modules retrieve
		 * the same profiler. The object intentionally lives until process shutdown.
		 */
		static const char* propertyName = "poca.core.PerformanceProfiler.instance";

		QCoreApplication* app = QCoreApplication::instance();
		if (app != nullptr) {
			QVariant value = app->property(propertyName);
			if (value.isValid()) {
				PerformanceProfiler* profiler = reinterpret_cast<PerformanceProfiler*>(value.value<quintptr>());
				if (profiler != nullptr)
					return *profiler;
			}

			PerformanceProfiler* profiler = new PerformanceProfiler();
			app->setProperty(propertyName, QVariant::fromValue<quintptr>(reinterpret_cast<quintptr>(profiler)));
			return *profiler;
		}

		static PerformanceProfiler fallbackProfiler;
		return fallbackProfiler;
	}

	void PerformanceProfiler::setEnabled(bool _enabled)
	{
		std::lock_guard<std::mutex> guard(m_mutex);
		m_enabled = _enabled;
	}

	bool PerformanceProfiler::enabled() const
	{
		std::lock_guard<std::mutex> guard(m_mutex);
		return m_enabled;
	}

	void PerformanceProfiler::record(const std::string& _category, const std::string& _name, double _milliseconds)
	{
		std::lock_guard<std::mutex> guard(m_mutex);
		if (!m_enabled)
			return;

		m_samples.push_back(Sample{ ++m_sequence, _category, _name, _milliseconds });
		while (m_samples.size() > m_maxSamples)
			m_samples.pop_front();
	}

	std::vector<PerformanceProfiler::Sample> PerformanceProfiler::samplesSince(uint64_t _sequence) const
	{
		std::lock_guard<std::mutex> guard(m_mutex);
		std::vector<Sample> samples;
		for (const Sample& sample : m_samples)
			if (sample.sequence > _sequence)
				samples.push_back(sample);
		return samples;
	}

	std::map<std::string, double> PerformanceProfiler::latestByCategory() const
	{
		std::lock_guard<std::mutex> guard(m_mutex);
		std::map<std::string, double> values;
		for (const Sample& sample : m_samples)
			values[sample.category] = sample.milliseconds;
		return values;
	}

	void PerformanceProfiler::clear()
	{
		std::lock_guard<std::mutex> guard(m_mutex);
		m_samples.clear();
	}
}
