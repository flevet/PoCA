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
		inline static constexpr size_t m_maxSamples = 4096;
	};
}

#endif
