/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DetectionSetPlugin.cpp
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

#include <QtWidgets/QTabWidget>

#include <General/Misc.h>
#include <OpenGL/Helper.h>
#include <Geometry/DetectionSet.hpp>
#include <General/Engine.hpp>

#include "DetectionSetPlugin.hpp"
#include "DetectionSetWidget.hpp"
#include "DetectionSetDisplayCommand.hpp"
#include "HeatMapDisplayCommand.hpp"
#include "CleanerCommand.hpp"
#include "DetectionSetBasicCommands.hpp"
#include "GaussianDisplayCommand.hpp"

poca::core::PluginList* DetectionSetPlugin::m_plugins = NULL;

void DetectionSetPlugin::addGUI(poca::core::MediatorWObjectFWidgetInterface* _mediator, QTabWidget* _parent)
{
	int pos = -1;
	for (int n = 0; n < _parent->count(); n++)
		if (_parent->tabText(n) == "Localizations")
			pos = n;
	if (pos != -1)
		m_parent = static_cast <QTabWidget*>(_parent->widget(pos));
	else
		m_parent = _parent;
	DetectionSetWidget* dsw = new DetectionSetWidget(_mediator, m_parent);
	_mediator->addWidget(dsw);
	int index2 = m_parent->addTab(dsw->getCleanerWidget(), QObject::tr("Cleaning"));
	int index = m_parent->insertTab(0, dsw, QObject::tr("Filtering/Display"));
	m_parent->setCurrentIndex(0);
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	m_parent->setTabVisible(index2, false);
#endif
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	m_parent->setTabVisible(index, false);
#endif
	QObject::connect(dsw, SIGNAL(transferNewObjectCreated(poca::core::MyObjectInterface*)), _parent->parentWidget(), SLOT(createWidget(poca::core::MyObjectInterface*)));
}

std::vector <std::pair<QAction*, QString>> DetectionSetPlugin::getActions()
{
	return m_actions;
}

poca::core::MyObjectInterface* DetectionSetPlugin::actionTriggered(QObject* _sender, poca::core::MyObjectInterface* _obj)
{
	return NULL;
}

void DetectionSetPlugin::addCommands(poca::core::CommandableObject* _bc)
{
	poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(_bc);
	if (dset) {
		dset->addCommand(new DetectionSetDisplayCommand(dset));
		dset->addCommand(new DetectionSetBasicCommands(dset));
		dset->addCommand(new HeatMapDisplayCommand(dset));
		dset->addCommand(new CleanerCommand(dset));
		dset->addCommand(new GaussianDisplayCommand(dset));
	}
}

void DetectionSetPlugin::setSingletons(poca::core::Engine* _engine)
{
	poca::core::Engine::instance()->setEngineSingleton(_engine); poca::core::Engine::instance()->setAllSingletons();
}