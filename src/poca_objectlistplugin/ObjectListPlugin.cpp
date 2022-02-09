/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListPlugin.cpp
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

#include <QtWidgets/QTabWidget>

#include <General/Misc.h>
#include <OpenGL/Helper.h>
#include <DesignPatterns/StateSoftwareSingleton.hpp>
#include <DesignPatterns/MacroRecorderSingleton.hpp>
#include <DesignPatterns/GlobalParametersSingleton.hpp>
#include <Geometry/VoronoiDiagram.hpp>
#include <Geometry/DetectionSet.hpp>
#include <Interfaces/ObjectListFactoryInterface.hpp>

#include "ObjectListPlugin.hpp"
#include "ObjectListBasicCommands.hpp"
#include "ObjectListDisplayCommand.hpp"
#include "ObjectListWidget.hpp"
#include "ObjectListParamDialog.hpp"

nlohmann::json ObjectListPlugin::m_parameters;

void ObjectListPlugin::addGUI(poca::core::MediatorWObjectFWidgetInterface* _mediator, QTabWidget* _parent)
{
	std::string nameStr = name().toLatin1().data();
	std::string type = "triangulation";

	poca::core::StateSoftwareSingleton* sss = poca::core::StateSoftwareSingleton::instance();
	const nlohmann::json& parameters = sss->getParameters();
	if (parameters.contains(nameStr)) {
		nlohmann::json param = parameters[nameStr];
		if (param.contains("typeObject"))
			type = param["typeObject"].get<std::string>();
	}
	m_parameters[nameStr]["typeObject"] = type;
	poca::core::GlobalParametersSingleton::instance()->m_parameters["typeObject"] = poca::geometry::ObjectListFactoryInterface::getTypeId(type);

	m_parent = _parent->findChild <QTabWidget*>("ObjectList");
	if (m_parent == NULL) {
		int pos = _parent->addTab(new QTabWidget, QObject::tr("ObjectList"));
		m_parent = static_cast <QTabWidget*>(_parent->widget(pos));
	}
	m_widget = new ObjectListWidget(_mediator, m_parent);
	_mediator->addWidget(m_widget);
	QObject::connect(m_widget, SIGNAL(transferNewObjectCreated(poca::core::MyObjectInterface*)), _parent->parentWidget(), SLOT(createWidget(poca::core::MyObjectInterface*)));
	int index = m_parent->addTab(m_widget, QObject::tr("Filtering/Display"));
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	m_parent->setTabVisible(index, false);
#endif
}

std::vector <std::pair<QAction*, QString>> ObjectListPlugin::getActions()
{
	if (m_actions.empty()) {
		QAction* action2 = new QAction(tr("Parameters"), this);
		action2->setStatusTip(tr("Set parameters"));

		m_actions.push_back(std::make_pair(action2, "Plugins/Objects"));
	}
	return m_actions;
}

poca::core::MyObjectInterface* ObjectListPlugin::actionTriggered(QObject* _sender, poca::core::MyObjectInterface* _obj)
{
	QAction* action = static_cast <QAction*>(_sender);
	std::string nameStr = name().toLatin1().data();
	if (action == m_actions[0].first) {
		ObjectListParamDialog* dial = new ObjectListParamDialog(m_parameters[nameStr]["typeObject"].get<std::string>());
		dial->setModal(true);
		if (dial->exec() == QDialog::Accepted) {
			std::string type = dial->typeObject();
			m_parameters[nameStr]["typeObject"] = type;
			poca::core::GlobalParametersSingleton::instance()->m_parameters["typeObject"] = poca::geometry::ObjectListFactoryInterface::getTypeId(type);
		}
		delete dial;
	}
	return NULL;
}

void ObjectListPlugin::addCommands(poca::core::CommandableObject* _bc)
{
	poca::geometry::ObjectList* objs = dynamic_cast <poca::geometry::ObjectList*>(_bc);
	if (objs) {
		objs->addCommand(new ObjectListDisplayCommand(objs));
		objs->addCommand(new ObjectListBasicCommands(objs));
	}
}

void ObjectListPlugin::setSingletons(const std::map <std::string, std::any>& _list)
{
	poca::core::setAllSingletons(_list);
	if (_list.find("HelperSingleton") != _list.end()) {
		poca::opengl::HelperSingleton::setHelperSingleton(std::any_cast <poca::opengl::HelperSingleton*>(_list.at("HelperSingleton")));
	}
}

void ObjectListPlugin::execute(poca::core::CommandInfo* _com)
{
	if (_com->isRecordable())
		poca::core::MacroRecorderSingleton::instance()->addCommand("VoronoiDiagramPlugin", _com);
	if (_com->nameCommand == "saveParameters") {
		if (!_com->hasParameter("file")) return;
		nlohmann::json* json = _com->getParameterPtr<nlohmann::json>("file");

		std::string nameStr = name().toLatin1().data();
		(*json)[nameStr] = m_parameters[nameStr];
	}
}

