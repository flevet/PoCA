/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      NearestLocsMultiColorPlugin.cpp
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
#include <General/Engine.hpp>
#include <Geometry/ObjectLists.hpp>
#include <General/PluginList.hpp>
#include <Objects/MyObject.hpp>
#include <General/Engine.hpp>

#include "NearestLocsMultiColorPlugin.hpp"
#include "NearestLocsMultiColorCommands.hpp"
#include "NearestLocsMultiColorWidget.hpp"

nlohmann::json NearestLocsMultiColorPlugin::m_parameters;
poca::core::PluginList* NearestLocsMultiColorPlugin::m_plugins = NULL;

void NearestLocsMultiColorPlugin::addGUI(poca::core::MediatorWObjectFWidgetInterface* _mediator, QTabWidget* _parent)
{
	NearestLocsMultiColorWidget* w = new NearestLocsMultiColorWidget(_mediator, _parent);
	_mediator->addWidget(w);

	QTabWidget* tabW = poca::core::utils::addSingleTabWidget(_parent, QString("Colocalization"), QString("Quantifs"), w);
	w->setParentTab(tabW);
}

std::vector <std::pair<QAction*, QString>> NearestLocsMultiColorPlugin::getActions()
{
	return m_actions;
}

poca::core::MyObjectInterface* NearestLocsMultiColorPlugin::actionTriggered(QObject* _sender, poca::core::MyObjectInterface* _obj)
{
	return NULL;
}

void NearestLocsMultiColorPlugin::addCommands(poca::core::CommandableObject* _bc)
{
	poca::core::MyObject* obj = dynamic_cast <poca::core::MyObject*>(_bc);
	if (obj && obj->nbColors() > 1) {
		obj->addCommand(new NearestLocsMultiColorCommands(obj));
	}
}

void NearestLocsMultiColorPlugin::setSingletons(poca::core::Engine* _engine)
{
	poca::core::Engine::instance()->setEngineSingleton(_engine); poca::core::Engine::instance()->setAllSingletons();
}



