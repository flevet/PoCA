/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      FilterHistogramWidget.cpp
*
* Copyright: Florian Levet (2020-2022)
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*
* PoCA is a free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 3 of the License, or (at your option) any later version.
*
* The algorithms that underlie PoCA have required considerable
* development. They are described in the original SR-Tesseler paper,
* doi:10.1038/nmeth.3579. If you use PoCA as part of work (visualization, 
* manipulation, quantification) towards a scientific publication, please include 
* a citation to the original paper.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program; if not, write to the Free Software Foundation,
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#include "FilterHistogramWidget.hpp"
#include "Icons.hpp"

namespace poca::plot {

	FilterHistogramWidget::FilterHistogramWidget(poca::core::MediatorWObjectFWidgetInterface* _mediator, const std::string& _nameComponent, QWidget* _parent /*= 0*/, Qt::WindowFlags _flags /*= 0 */) : QWidget(_parent, _flags), m_nameComponent(_nameComponent), m_redraw(false)
	{
		m_mediator = _mediator;
		m_parent = dynamic_cast <ObserverForMediator*>(_parent);

		this->setObjectName("FilterHistogramWidget");
		this->addActionToObserve("LoadObjCharacteristicsAllWidgets");

		m_customPlot = new QCPHistogram(this);
		m_customPlot->setMaximumHeight(60);
		m_customPlot->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
		m_customPlot->legend->setTextColor(Qt::black);
		QFont fontLegend("Helvetica", 9);
		fontLegend.setBold(true);
		m_customPlot->legend->setFont(fontLegend);
		m_customPlot->legend->setBrush(Qt::NoBrush);
		m_customPlot->legend->setBorderPen(Qt::NoPen);
		QColor background = QWidget::palette().color(QWidget::backgroundRole());
		m_customPlot->setBackground(background);

		int maxSize = 20;
		m_buttonDisplay = new QPushButton();
		m_buttonDisplay->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		m_buttonDisplay->setMaximumSize(QSize(maxSize, maxSize));
		m_buttonDisplay->setIcon(QIcon(QPixmap(poca::plot::bullseyeIcon)));
		m_buttonDisplay->setToolTip("Select histogram for display");

		m_buttonSave = new QPushButton();
		m_buttonSave->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		m_buttonSave->setMaximumSize(QSize(maxSize, maxSize));
		m_buttonSave->setIcon(QIcon(QPixmap(poca::plot::saveIcon)));
		m_buttonSave->setToolTip("Save histogram values");

		m_cboxLog = new QCheckBox;
		m_cboxLog->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		m_cboxLog->setMaximumSize(QSize(maxSize, maxSize));

		m_minLEdit = new QLineEdit;
		m_minLEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		m_minLEdit->setMaximumSize(QSize(maxSize * 2, maxSize));

		m_maxLEdit = new QLineEdit;
		m_maxLEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		m_maxLEdit->setMaximumSize(QSize(maxSize * 2, maxSize));

		m_buttonDelete = new QPushButton();
		m_buttonDelete->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		m_buttonDelete->setMaximumSize(QSize(maxSize, maxSize));
		m_buttonDelete->setIcon(QIcon(QPixmap(poca::plot::deleteIcon)));
		m_buttonDelete->setToolTip("Delete feature");

		QHBoxLayout* layout = new QHBoxLayout;
		layout->setContentsMargins(5, 0, 5, 0);
		layout->setSpacing(5);
		layout->addWidget(m_buttonDisplay);
		layout->addWidget(m_buttonSave);
		layout->addWidget(m_cboxLog);
		layout->addWidget(m_minLEdit);
		layout->addWidget(m_customPlot);
		layout->addWidget(m_maxLEdit);
		layout->addWidget(m_buttonDelete);

		this->setLayout(layout);

		QObject::connect(m_buttonDisplay, SIGNAL(pressed()), this, SLOT(actionNeeded()));
		QObject::connect(m_buttonSave, SIGNAL(pressed()), this, SLOT(actionNeeded()));
		QObject::connect(m_buttonDelete, SIGNAL(pressed()), this, SLOT(actionNeeded()));
		QObject::connect(m_cboxLog, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
		QObject::connect(m_customPlot, SIGNAL(actionNeededSignal(const QString&)), this, SLOT(actionNeeded(const QString&)));

		QObject::connect(m_minLEdit, SIGNAL(returnPressed()), SLOT(actionNeeded()));
		QObject::connect(m_maxLEdit, SIGNAL(returnPressed()), SLOT(actionNeeded()));
	}

	FilterHistogramWidget::~FilterHistogramWidget()
	{

	}

	void FilterHistogramWidget::setInfos(const QString& _name, poca::core::HistogramInterface* _hist, const bool _log, poca::core::PaletteInterface* _palette)
	{
		m_name = _name.toLatin1().data();
		m_buttonDelete->setEnabled(m_name != "x" && m_name != "y" && m_name != "z");
		m_customPlot->setInfos(_name, _hist, _palette);
		m_customPlot->setEnabled(_hist->hasInteraction());
		m_customPlot->update();
		double minV = m_customPlot->getCurrentMin();
		double maxV = m_customPlot->getCurrentMax();
		int precision = (minV < -10. || minV > 10.) ? 0 : 3;
		m_minLEdit->setText(QString::number(minV, 'f', precision));
		precision = (maxV < -10. || maxV > 10.) ? 0 : 3;
		m_maxLEdit->setText(QString::number(maxV, 'f', precision));
		m_cboxLog->blockSignals(true);
		m_cboxLog->setChecked(_log);
		m_cboxLog->blockSignals(false);
	}

	void FilterHistogramWidget::actionNeeded()
	{
		QObject* sender = QObject::sender();
		if (sender == m_buttonDisplay) {
			m_redraw = true;
			poca::core::CommandInfo ci(true, "histogram",
				"feature", m_name,
				"action", std::string("displayWithLUT"));
			m_mediator->actionAsked(m_parent, &ci);
		}
		else if (sender == m_buttonSave) {
			m_redraw = true;
			poca::core::CommandInfo ci(true, "histogram",
				"feature", m_name,
				"action", std::string("save"));
			m_mediator->actionAsked(m_parent, &ci);
		}
		else if (sender == m_minLEdit) {
			bool ok;
			double val = m_minLEdit->text().toDouble(&ok);
			if (ok) {
				m_customPlot->setCurrentMin(val);
				m_redraw = true;
				poca::core::CommandInfo ci(true, "histogram",
					"feature", m_name,
					"action", std::string("changeBoundsCustom"),
					"min", m_customPlot->getCurrentMin(),
					"max", m_customPlot->getCurrentMax());
				m_mediator->actionAsked(m_parent, &ci);
			}
		}
		else if (sender == m_maxLEdit) {
			bool ok;
			double val = m_maxLEdit->text().toDouble(&ok);
			if (ok) {
				m_customPlot->setCurrentMax(val);
				m_redraw = true;
				poca::core::CommandInfo ci(true, "histogram",
					"feature", m_name,
					"action", std::string("changeBoundsCustom"),
					"min", m_customPlot->getCurrentMin(),
					"max", m_customPlot->getCurrentMax());
				m_mediator->actionAsked(m_parent, &ci);
			}
		}
		else if (sender == m_buttonDelete) {
			m_redraw = true;
			poca::core::CommandInfo ci(true, "histogram",
				"feature", m_name,
				"action", std::string("delete"));
			m_mediator->actionAsked(m_parent, &ci);
		}
	}

	void FilterHistogramWidget::actionNeeded(bool _val)
	{
		QObject* sender = QObject::sender();
		if (sender == m_cboxLog) {
			m_redraw = true;
			poca::core::CommandInfo ci(true, "histogram",
				"feature", m_name,
				"action", std::string("log"),
				"value", _val);
			m_mediator->actionAsked(m_parent, &ci);
		}
	}

	void FilterHistogramWidget::actionNeeded(int _val)
	{
	}

	void FilterHistogramWidget::actionNeeded(const QString& _action)
	{
		QObject* sender = QObject::sender();
		if (sender == m_customPlot) {
			if (_action == "changeBoundsCustom") {
				m_redraw = true;
				poca::core::CommandInfo ci(true, "histogram",
					"feature", m_name,
					"action", std::string("changeBoundsCustom"),
					"min", m_customPlot->getCurrentMin(),
					"max", m_customPlot->getCurrentMax());
				m_mediator->actionAsked(m_parent, &ci);
			}
		}
	}

	void FilterHistogramWidget::actionNeeded(poca::core::CommandInfo* _ci)
	{
		QObject* sender = QObject::sender();
		if (sender == m_customPlot)
			m_mediator->actionAsked(m_parent, _ci);
	}

	void FilterHistogramWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
	{
	}

	void FilterHistogramWidget::redraw()
	{
		m_customPlot->update();
		m_redraw = false;
	}
}

