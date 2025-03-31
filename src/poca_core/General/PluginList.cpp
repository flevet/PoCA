/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PluginList.cpp
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

#include <QtCore/QThread>
#include <QtCore/QDebug>

#include "PluginList.hpp"

namespace poca::core {

	PluginList::PluginList()
	{

	}

	PluginList::~PluginList()
	{

	}

	void PluginList::addGUI(poca::core::MediatorWObjectFWidgetInterface* _mediator, QTabWidget* _tabW)
	{
		for(PluginInterface * plugin : m_plugins)
			plugin->addGUI(_mediator, _tabW);
	}

	poca::core::MyObjectInterface* PluginList::actionTriggered(QObject* _sender, poca::core::MyObjectInterface* _obj)
	{
		poca::core::MyObjectInterface* newObj = NULL;
		for (PluginInterface* plugin : m_plugins) {
			newObj = plugin->actionTriggered(_sender, _obj);
			if (newObj)
				return newObj;
		}
		return newObj;
	}

	void PluginList::addCommands(poca::core::CommandableObject* _bci)
	{
		for (PluginInterface* plugin : m_plugins)
			plugin->addCommands(_bci);
	}

	void PluginList::setSingletons(poca::core::Engine* _engine)
	{
		for (PluginInterface* plugin : m_plugins)
			plugin->setSingletons(_engine);
	}

	void PluginList::execute(poca::core::CommandInfo* _ci)
	{
		for (PluginInterface* plugin : m_plugins)
			plugin->execute(_ci);
	}
}

