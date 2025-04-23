/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MediatorWObjectFWidget.cpp
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

#include "../DesignPatterns/MediatorWObjectFWidget.hpp"
#include "../General/Command.hpp"

namespace poca::core {

	MediatorWObjectFWidget* MediatorWObjectFWidget::m_instance = 0;

	MediatorWObjectFWidget* MediatorWObjectFWidget::instance()
	{
		if (m_instance == 0)
			m_instance = new MediatorWObjectFWidget;
		return m_instance;
	}

	void MediatorWObjectFWidget::setMediatorWObjectFWidgetSingleron(poca::core::MediatorWObjectFWidget* _med)
	{
		m_instance = _med;
	}

	void MediatorWObjectFWidget::deleteInstance()
	{
		if (m_instance != 0)
			delete m_instance;
		m_instance = 0;
	}

	MediatorWObjectFWidget::MediatorWObjectFWidget()
	{
		m_currentObj = NULL;
	}

	MediatorWObjectFWidget::~MediatorWObjectFWidget()
	{
		m_widgets.clear();
	}

	void MediatorWObjectFWidget::addWidget(ObserverForMediator* _dw)
	{
		std::set < ObserverForMediator* >::iterator it = m_widgets.find(_dw);
		if (it == m_widgets.end())
			m_widgets.insert(_dw);
	}

	void MediatorWObjectFWidget::actionAsked(ObserverForMediator* _widget, CommandInfo* _actionText)
	{
		std::set < ObserverForMediator* >::iterator it = m_widgets.find(_widget);
		if(it != m_widgets.end())
			(*it)->performAction(m_currentObj, _actionText);
	}

	void MediatorWObjectFWidget::actionAskedAllObservers(CommandInfo* _actionText)
	{
		for (std::set < ObserverForMediator* >::iterator it = m_widgets.begin(); it != m_widgets.end(); it++) {
			ObserverForMediator* ofm = *it;
			ofm->performAction(m_currentObj, _actionText);
		}
	}

	void MediatorWObjectFWidget::addObserversToSubject(SubjectInterface* _subject, const CommandInfo& _aspect)
	{
		for (std::set < ObserverForMediator* >::iterator it = m_widgets.begin(); it != m_widgets.end(); it++) {
			ObserverForMediator* ofm = *it;
			if (ofm->hasActionToObserve(_aspect))
				_subject->attach(ofm, _aspect);
		}
	}

	void MediatorWObjectFWidget::setCurrentObject(MyObjectInterface* _obj)
	{
		m_currentObj = _obj;
	}
}

