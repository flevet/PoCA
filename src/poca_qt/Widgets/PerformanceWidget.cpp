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

#include <QtCore/QHash>
#include <QtGui/QMouseEvent>
#include <QtGui/QPainter>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSizePolicy>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStyle>
#include <QtWidgets/QStyleOption>
#include <QtWidgets/QVBoxLayout>

#include <algorithm>
#include <cmath>

namespace poca::qt {

	class PerformancePlotWidget : public QWidget {
	public:
		PerformancePlotWidget(PerformanceWidget* _owner, QWidget* _parent = nullptr) : QWidget(_parent), m_owner(_owner)
		{
			setMinimumSize(240, 160);
			setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
		}

	protected:
		void mousePressEvent(QMouseEvent* _event) override
		{
			const auto& history = m_owner->history();
			if (history.empty())
				return;

			const QRect r = rect().adjusted(8, 8, -8, -8);
			const QRect plot = r.adjusted(54, 34, -12, -28);
			if (!plot.contains(_event->pos())) {
				m_owner->setSelectedFrameIndex(-1);
				return;
			}

			const int count = (int)history.size();
			const double step = std::max(1.0, (double)plot.width() / (double)std::max(1, count));
			int index = (int)std::floor((double)(_event->pos().x() - plot.left()) / step);
			index = std::max(0, std::min(count - 1, index));
			m_owner->setSelectedFrameIndex(index);
		}

		void paintEvent(QPaintEvent*) override
		{
			QStyleOption opt;
			opt.initFrom(this);
			QPainter painter(this);
			style()->drawPrimitive(QStyle::PE_Widget, &opt, &painter, this);
			painter.setRenderHint(QPainter::Antialiasing, true);

			const QRect r = rect().adjusted(8, 8, -8, -8);
			painter.fillRect(r, QColor(248, 249, 251));
			painter.setPen(QColor(190, 196, 205));
			painter.drawRect(r);

			const auto& history = m_owner->history();
			if (history.empty()) {
				painter.setPen(QColor(90, 96, 106));
				painter.drawText(r, Qt::AlignCenter, QStringLiteral("No performance samples yet"));
				return;
			}

			QRect titleRect = r.adjusted(8, 4, -8, 0);
			titleRect.setHeight(22);
			painter.setPen(QColor(35, 39, 46));
			painter.drawText(titleRect, Qt::AlignLeft | Qt::AlignVCenter, QStringLiteral("Performance per refresh loop (ms)"));

			QRect plot = r.adjusted(54, 34, -12, -28);
			if (!plot.isValid())
				return;

			double maxTotal = 1.0;
			for (const auto& frame : history) {
				double total = 0.0;
				for (const auto& item : frame)
					total += std::max(0.0, item.second);
				maxTotal = std::max(maxTotal, total);
			}

			const bool logScale = m_owner->useLogScale();
			double scaleMax = logScale ? std::log10(maxTotal + 1.0) : maxTotal;
			if (!logScale)
				scaleMax = std::ceil(scaleMax / 5.0) * 5.0;
			scaleMax = std::max(1.0, scaleMax);

			painter.setPen(QColor(222, 226, 232));
			for (int n = 0; n <= 4; n++) {
				const int y = plot.bottom() - (plot.height() * n) / 4;
				painter.drawLine(plot.left(), y, plot.right(), y);
			}

			painter.setPen(QColor(95, 101, 112));
			const QString maxLabel = logScale ? QStringLiteral("log %1").arg(maxTotal, 0, 'f', 0) : QString::number(maxTotal, 'f', 0);
			painter.drawText(QRect(r.left() + 4, plot.top() - 8, 48, 18), Qt::AlignRight | Qt::AlignVCenter, maxLabel);
			painter.drawText(QRect(r.left() + 4, plot.bottom() - 10, 48, 18), Qt::AlignRight | Qt::AlignVCenter, QStringLiteral("0"));
			painter.drawText(QRect(plot.left(), plot.bottom() + 6, plot.width(), 18), Qt::AlignCenter, logScale ? QStringLiteral("Log scale") : QStringLiteral("Linear scale"));

			const int count = (int)history.size();
			const double step = std::max(1.0, (double)plot.width() / (double)std::max(1, count));
			const int barWidth = std::max(2, (int)std::floor(step * 0.82));
			const auto& categories = m_owner->categories();

			for (int i = 0; i < count; i++) {
				const auto& frame = history[(size_t)i];
				double accumulatedDisplay = 0.0;
				const int x = plot.left() + (int)std::round((double)i * step);
				for (const std::string& category : categories) {
					auto it = frame.find(category);
					if (it == frame.end() || it->second <= 0.0)
						continue;

					const double nextTotal = [&]() {
						double total = 0.0;
						for (const std::string& c : categories) {
							auto jt = frame.find(c);
							if (jt != frame.end() && jt->second > 0.0)
								total += jt->second;
							if (c == category)
								break;
						}
						return total;
					}();

					const double bottomValue = logScale ? accumulatedDisplay : (nextTotal - it->second);
					const double topValue = logScale ? std::log10(nextTotal + 1.0) : nextTotal;
					const double bottomNormalized = std::min(1.0, std::max(0.0, bottomValue / scaleMax));
					const double topNormalized = std::min(1.0, std::max(0.0, topValue / scaleMax));
					const int yTop = plot.bottom() - (int)std::round(topNormalized * plot.height());
					const int yBottom = plot.bottom() - (int)std::round(bottomNormalized * plot.height());
					QRect bar(x, yTop, barWidth, std::max(1, yBottom - yTop));
					painter.fillRect(bar, m_owner->colorForCategory(category));
					accumulatedDisplay = topValue;
				}
			}

			const int selected = m_owner->selectedFrameIndex();
			if (selected >= 0 && selected < count) {
				const int x = plot.left() + (int)std::round((double)selected * step);
				QPen pen(QColor(30, 30, 30));
				pen.setWidth(2);
				painter.setPen(pen);
				painter.drawRect(QRect(x - 2, plot.top(), barWidth + 4, plot.height()));
			}
		}

	private:
		PerformanceWidget* m_owner{ nullptr };
	};

	class PerformanceLegendWidget : public QWidget {
	public:
		PerformanceLegendWidget(PerformanceWidget* _owner, QWidget* _parent = nullptr) : QWidget(_parent), m_owner(_owner)
		{
			setMinimumWidth(190);
			setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
		}

		QSize sizeHint() const override
		{
			return QSize(250, 240);
		}

	protected:
		void paintEvent(QPaintEvent*) override
		{
			QStyleOption opt;
			opt.initFrom(this);
			QPainter painter(this);
			style()->drawPrimitive(QStyle::PE_Widget, &opt, &painter, this);
			painter.setRenderHint(QPainter::Antialiasing, true);

			QRect r = rect().adjusted(4, 4, -4, -4);
			painter.fillRect(r, QColor(248, 249, 251));
			painter.setPen(QColor(190, 196, 205));
			painter.drawRect(r);

			QRect title = r.adjusted(8, 6, -8, 0);
			title.setHeight(22);
			painter.setPen(QColor(35, 39, 46));
			painter.drawText(title, Qt::AlignLeft | Qt::AlignVCenter, QStringLiteral("Categories"));

			int y = title.bottom() + 10;
			const int rowHeight = 24;
			const auto& categories = m_owner->categories();
			if (categories.empty()) {
				painter.setPen(QColor(90, 96, 106));
				painter.drawText(r.adjusted(8, 34, -8, -8), Qt::AlignTop | Qt::AlignLeft, QStringLiteral("No categories yet"));
				return;
			}

			for (const std::string& category : categories) {
				if (y + rowHeight > r.bottom())
					break;
				const QColor color = m_owner->colorForCategory(category);
				painter.fillRect(QRect(r.left() + 10, y + 6, 14, 12), color);
				painter.setPen(QColor(45, 50, 58));
				const QString text = QStringLiteral("%1  %2 ms").arg(QString::fromStdString(category)).arg(m_owner->displayedValue(category), 0, 'f', 1);
				painter.drawText(QRect(r.left() + 32, y, r.width() - 40, rowHeight), Qt::AlignLeft | Qt::AlignVCenter, text);
				y += rowHeight;
			}

			painter.setPen(QColor(60, 65, 74));
			painter.drawText(QRect(r.left() + 8, r.bottom() - 24, r.width() - 16, 18), Qt::AlignLeft | Qt::AlignVCenter,
				QStringLiteral("%1  Total: %2 ms").arg(m_owner->displayedFrameLabel()).arg(m_owner->displayedTotal(), 0, 'f', 1));
		}

	private:
		PerformanceWidget* m_owner{ nullptr };
	};

	PerformanceWidget::PerformanceWidget(QWidget* _parent) : QWidget(_parent)
	{
		setMinimumHeight(220);
		setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

		m_plotWidget = new PerformancePlotWidget(this, this);
		m_legendWidget = new PerformanceLegendWidget(this, this);
		m_logScaleCheckBox = new QCheckBox(QStringLiteral("Log display"), this);

		QWidget* sidePanel = new QWidget(this);
		QVBoxLayout* sideLayout = new QVBoxLayout(sidePanel);
		sideLayout->setContentsMargins(0, 0, 0, 0);
		sideLayout->setSpacing(6);
		sideLayout->addWidget(m_logScaleCheckBox);
		sideLayout->addWidget(m_legendWidget, 1);

		m_splitter = new QSplitter(Qt::Horizontal, this);
		m_splitter->setChildrenCollapsible(false);
		m_splitter->addWidget(m_plotWidget);
		m_splitter->addWidget(sidePanel);
		m_splitter->setStretchFactor(0, 1);
		m_splitter->setStretchFactor(1, 0);
		m_splitter->setSizes({ 900, 260 });

		QHBoxLayout* layout = new QHBoxLayout(this);
		layout->setContentsMargins(4, 4, 4, 4);
		layout->addWidget(m_splitter);

		connect(m_logScaleCheckBox, &QCheckBox::toggled, this, [this](bool _checked) {
			m_useLogScale = _checked;
			refreshViews();
		});

		m_timer.setInterval(500);
		connect(&m_timer, &QTimer::timeout, this, [this]() {
			refreshSamples();
			refreshViews();
		});
		m_timer.start();
	}

	QSize PerformanceWidget::sizeHint() const
	{
		return QSize(760, 280);
	}

	double PerformanceWidget::latestValue(const std::string& _category) const
	{
		auto it = m_latest.find(_category);
		return it == m_latest.end() ? 0.0 : it->second;
	}

	double PerformanceWidget::latestTotal() const
	{
		double total = 0.0;
		for (const auto& item : m_latest)
			total += item.second;
		return total;
	}

	double PerformanceWidget::displayedValue(const std::string& _category) const
	{
		if (m_selectedFrameIndex >= 0 && m_selectedFrameIndex < (int)m_history.size()) {
			const auto& frame = m_history[(size_t)m_selectedFrameIndex];
			auto it = frame.find(_category);
			return it == frame.end() ? 0.0 : it->second;
		}
		return latestValue(_category);
	}

	double PerformanceWidget::displayedTotal() const
	{
		if (m_selectedFrameIndex >= 0 && m_selectedFrameIndex < (int)m_history.size()) {
			double total = 0.0;
			for (const auto& item : m_history[(size_t)m_selectedFrameIndex])
				total += item.second;
			return total;
		}
		return latestTotal();
	}

	QString PerformanceWidget::displayedFrameLabel() const
	{
		if (m_selectedFrameIndex >= 0 && m_selectedFrameIndex < (int)m_history.size())
			return QStringLiteral("Selected #%1").arg(m_selectedFrameIndex + 1);
		return QStringLiteral("Latest");
	}

	void PerformanceWidget::setSelectedFrameIndex(int _index)
	{
		if (_index < 0 || _index >= (int)m_history.size())
			m_selectedFrameIndex = -1;
		else
			m_selectedFrameIndex = _index;
		refreshViews();
	}

	QColor PerformanceWidget::colorForCategory(const std::string& _category) const
	{
		static const QColor fixedColors[] = {
			QColor(53, 112, 192), QColor(213, 103, 37), QColor(54, 145, 102), QColor(150, 84, 166),
			QColor(190, 70, 94), QColor(46, 160, 185), QColor(228, 153, 54), QColor(116, 189, 80),
			QColor(92, 130, 210), QColor(173, 95, 45), QColor(37, 145, 117), QColor(182, 80, 156),
			QColor(117, 104, 180), QColor(196, 115, 60), QColor(72, 155, 80), QColor(207, 80, 80)
		};

		for (size_t i = 0; i < m_categories.size(); i++)
			if (m_categories[i] == _category)
				return fixedColors[i % (sizeof(fixedColors) / sizeof(fixedColors[0]))];

		const uint hue = qHash(QString::fromStdString(_category)) % 360;
		return QColor::fromHsv((int)hue, 170, 205);
	}

	void PerformanceWidget::refreshSamples()
	{
		std::vector<poca::core::PerformanceProfiler::Sample> samples = poca::core::PerformanceProfiler::instance().samplesSince(m_lastSequence);
		std::map<std::string, double> totals;
		for (const auto& sample : samples) {
			m_lastSequence = std::max(m_lastSequence, sample.sequence);
			totals[sample.category] += sample.milliseconds;
		}
		if (totals.empty())
			return;

		for (const auto& item : totals) {
			m_latest[item.first] = item.second;
			if (std::find(m_categories.begin(), m_categories.end(), item.first) == m_categories.end())
				m_categories.push_back(item.first);
		}

		m_history.push_back(totals);
		while (m_history.size() > m_maxValues) {
			m_history.pop_front();
			if (m_selectedFrameIndex > 0)
				m_selectedFrameIndex--;
			else if (m_selectedFrameIndex == 0)
				m_selectedFrameIndex = -1;
		}
		if (m_selectedFrameIndex >= (int)m_history.size())
			m_selectedFrameIndex = -1;
	}

	void PerformanceWidget::refreshViews()
	{
		if (m_plotWidget != nullptr)
			m_plotWidget->update();
		if (m_legendWidget != nullptr)
			m_legendWidget->update();
	}
}
