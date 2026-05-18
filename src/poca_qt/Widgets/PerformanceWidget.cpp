/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PerformanceWidget.cpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#include "PerformanceWidget.hpp"

#include <QtGui/QPainter>
#include <QtGui/QPainterPath>
#include <QtWidgets/QSizePolicy>
#include <QtWidgets/QStyle>
#include <QtWidgets/QStyleOption>

#include <algorithm>
#include <cmath>

namespace poca::qt {

	PerformanceWidget::PerformanceWidget(QWidget* _parent) : QWidget(_parent)
	{
		setMinimumHeight(180);
		setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
		m_timer.setInterval(500);
		connect(&m_timer, &QTimer::timeout, this, [this]() {
			refreshSamples();
			update();
		});
		m_timer.start();
	}

	QSize PerformanceWidget::sizeHint() const
	{
		return QSize(320, 220);
	}

	void PerformanceWidget::refreshSamples()
	{
		std::vector<poca::core::PerformanceProfiler::Sample> samples = poca::core::PerformanceProfiler::instance().samplesSince(m_lastSequence);
		std::map<std::string, double> totals;
		for (const auto& sample : samples) {
			m_lastSequence = std::max(m_lastSequence, sample.sequence);
			totals[sample.category] += sample.milliseconds;
		}
		for (const auto& item : totals) {
			Series& series = m_series[item.first];
			series.latest = item.second;
			series.values.push_back(item.second);
			while (series.values.size() > m_maxValues)
				series.values.pop_front();
		}
	}

	void PerformanceWidget::paintEvent(QPaintEvent*)
	{
		QStyleOption opt;
		opt.initFrom(this);
		QPainter painter(this);
		style()->drawPrimitive(QStyle::PE_Widget, &opt, &painter, this);
		painter.setRenderHint(QPainter::Antialiasing, true);

		const QRect r = rect().adjusted(10, 10, -10, -10);
		painter.fillRect(r, QColor(248, 249, 251));
		painter.setPen(QColor(190, 196, 205));
		painter.drawRect(r);

		if (m_series.empty()) {
			painter.setPen(QColor(90, 96, 106));
			painter.drawText(r, Qt::AlignCenter, QStringLiteral("No performance samples yet"));
			return;
		}

		QRect titleRect = r.adjusted(8, 4, -8, 0);
		titleRect.setHeight(22);
		painter.setPen(QColor(35, 39, 46));
		painter.drawText(titleRect, Qt::AlignLeft | Qt::AlignVCenter, QStringLiteral("Performance"));

		QRect plot = r.adjusted(48, 34, -10, -30);
		double maxValue = 1.0;
		for (const auto& item : m_series)
			for (double v : item.second.values)
				maxValue = std::max(maxValue, v);
		maxValue = std::ceil(maxValue / 5.0) * 5.0;

		painter.setPen(QColor(222, 226, 232));
		for (int n = 0; n <= 4; n++) {
			const int y = plot.bottom() - (plot.height() * n) / 4;
			painter.drawLine(plot.left(), y, plot.right(), y);
		}
		painter.setPen(QColor(95, 101, 112));
		painter.drawText(QRect(r.left() + 4, plot.top() - 8, 42, 18), Qt::AlignRight | Qt::AlignVCenter, QString::number(maxValue, 'f', 0));
		painter.drawText(QRect(r.left() + 4, plot.bottom() - 10, 42, 18), Qt::AlignRight | Qt::AlignVCenter, QStringLiteral("0"));

		const QColor colors[] = {
			QColor(53, 112, 192),
			QColor(213, 103, 37),
			QColor(54, 145, 102),
			QColor(150, 84, 166),
			QColor(190, 70, 94)
		};

		int index = 0;
		QRect legend = r.adjusted(8, r.height() - 24, -8, -4);
		for (const auto& item : m_series) {
			const QColor color = colors[index % (sizeof(colors) / sizeof(colors[0]))];
			const Series& series = item.second;
			if (series.values.size() > 1) {
				QPainterPath path;
				for (size_t n = 0; n < series.values.size(); n++) {
					const double x = plot.left() + (double)n * (double)plot.width() / (double)(m_maxValues - 1);
					const double normalized = std::min(1.0, std::max(0.0, series.values[n] / maxValue));
					const double y = plot.bottom() - normalized * plot.height();
					if (n == 0)
						path.moveTo(x, y);
					else
						path.lineTo(x, y);
				}
				painter.setPen(QPen(color, 2.0));
				painter.drawPath(path);
			}

			const int xLegend = legend.left() + index * 145;
			painter.fillRect(QRect(xLegend, legend.top() + 5, 12, 8), color);
			painter.setPen(QColor(45, 50, 58));
			painter.drawText(QRect(xLegend + 16, legend.top(), 125, legend.height()),
				Qt::AlignLeft | Qt::AlignVCenter,
				QString("%1 %2 ms").arg(QString::fromStdString(item.first)).arg(series.latest, 0, 'f', 1));
			index++;
			if (xLegend + 145 > legend.right())
				break;
		}
	}
}
