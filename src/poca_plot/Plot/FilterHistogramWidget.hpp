/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      FilterHistogramWidget.hpp
*
* Copyright: Florian Levet (2020-2025)
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

#ifndef FilterHistogramWidget_h__
#define FilterHistogramWidget_h__

#include <QtWidgets/QWidget>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QPushButton>

#include <DesignPatterns/Observer.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <Interfaces/PaletteInterface.hpp>
#include <General/Command.hpp>

#include "QCPHistogram.hpp"

namespace poca::plot {

	class FilterHistogramWidget : public QWidget, public poca::core::ObserverForMediator {
		Q_OBJECT

	public:
		FilterHistogramWidget(poca::core::MediatorWObjectFWidgetInterface*, const std::string&, QWidget* = 0, Qt::WindowFlags = Qt::WindowFlags());
		virtual ~FilterHistogramWidget();

		void setInfos(const QString&, poca::core::HistogramInterface*, const bool, poca::core::PaletteInterface*);
		void setPalette(poca::core::PaletteInterface* _pal) { m_customPlot->setPalette(_pal); }
		void performAction(poca::core::MyObjectInterface*, poca::core::CommandInfo*);
		void update(poca::core::SubjectInterface*, const poca::core::CommandInfo&) {}
		void executeMacro(poca::core::MyObjectInterface*, poca::core::CommandInfo*) {}

		void redraw();

		inline const std::string& name() const { return m_name; }

	signals:
		void actionNeededSignal(poca::core::CommandInfo*);

	protected slots:
		virtual void actionNeeded();
		virtual void actionNeeded(bool);
		virtual void actionNeeded(int);
		virtual void actionNeeded(const QString&);
		virtual void actionNeeded(poca::core::CommandInfo*);

	protected:
		poca::core::MediatorWObjectFWidgetInterface* m_mediator;
		std::string m_nameComponent;//Name of the component on which commands are applied

		QPushButton* m_buttonDisplay, * m_buttonSave, * m_buttonDelete, * m_buttonScaleLUT;
		QCheckBox* m_cboxLog;
		QLineEdit* m_minLEdit, * m_maxLEdit;

		std::string m_name;
		poca::core::ObserverForMediator* m_parent;

		bool m_redraw;

		QCPHistogram* m_customPlot;
	};
}

#endif

