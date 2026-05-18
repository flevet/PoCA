/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PerformanceWidget.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef PerformanceWidget_hpp__
#define PerformanceWidget_hpp__

#include <QtCore/QTimer>
#include <QtWidgets/QWidget>

#include <General/PerformanceProfiler.hpp>

#include <deque>
#include <map>
#include <string>
#include <vector>

namespace poca::qt {

	class PerformanceWidget : public QWidget {
	public:
		PerformanceWidget(QWidget* = nullptr);

	protected:
		void paintEvent(QPaintEvent*) override;
		QSize sizeHint() const override;

	private:
		void refreshSamples();

	private:
		struct Series {
			std::deque<double> values;
			double latest{ 0.0 };
		};

		QTimer m_timer;
		uint64_t m_lastSequence{ 0 };
		std::map<std::string, Series> m_series;
		size_t m_maxValues{ 180 };
	};
}

#endif
