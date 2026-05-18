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
