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

#include <QtCore/QString>
#include <QtCore/QTimer>
#include <QtGui/QColor>
#include <QtWidgets/QWidget>

#include <General/PerformanceProfiler.hpp>

#include <deque>
#include <map>
#include <string>
#include <vector>

class QCheckBox;
class QSplitter;
class QMouseEvent;

namespace poca::qt {

	class PerformancePlotWidget;
	class PerformanceLegendWidget;

	class PerformanceWidget : public QWidget {
	public:
		PerformanceWidget(QWidget* = nullptr);

		QSize sizeHint() const override;

		const std::deque<std::map<std::string, double>>& history() const { return m_history; }
		const std::vector<std::string>& categories() const { return m_categories; }
		double latestValue(const std::string&) const;
		double latestTotal() const;
		double displayedValue(const std::string&) const;
		double displayedTotal() const;
		QString displayedFrameLabel() const;
		int selectedFrameIndex() const { return m_selectedFrameIndex; }
		void setSelectedFrameIndex(int);
		bool useLogScale() const { return m_useLogScale; }
		QColor colorForCategory(const std::string&) const;

	private:
		void refreshSamples();
		void refreshViews();

	private:
		QTimer m_timer;
		uint64_t m_lastSequence{ 0 };
		std::deque<std::map<std::string, double>> m_history;
		std::map<std::string, double> m_latest;
		std::vector<std::string> m_categories;
		size_t m_maxValues{ 180 };
		bool m_useLogScale{ false };
		int m_selectedFrameIndex{ -1 };

		QSplitter* m_splitter{ nullptr };
		PerformancePlotWidget* m_plotWidget{ nullptr };
		PerformanceLegendWidget* m_legendWidget{ nullptr };
		QCheckBox* m_logScaleCheckBox{ nullptr };
	};
}

#endif
