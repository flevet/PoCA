/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagramPlugin.cpp
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
#include <Geometry/VoronoiDiagram.hpp>
#include <General/Engine.hpp>
#include <General/Engine.hpp>
#include <DesignPatterns/MacroRecorderSingleton.hpp>
#include <General/PluginList.hpp>
#include <Plot/Icons.hpp>
#include <Interfaces/VoronoiDiagramFactoryInterface.hpp>
#include <General/Engine.hpp>

#include "VoronoiDiagramPlugin.hpp"
#include "VoronoiDiagramDisplayCommand.hpp"
#include "VoronoiDiagramBasicCommands.hpp"
#include "VoronoiDiagramParamDialog.hpp"
#include "VoronoiDiagramCharacteristicsCommands.hpp"

VoronoiDiagramConstructionCommand::VoronoiDiagramConstructionCommand(poca::geometry::DetectionSet* _dset) :poca::core::Command("VoronoiDiagramConstructionCommand")
{
	m_dset = _dset;
}

VoronoiDiagramConstructionCommand::VoronoiDiagramConstructionCommand(const VoronoiDiagramConstructionCommand& _o) : Command(_o)
{
	m_dset = _o.m_dset;
}

VoronoiDiagramConstructionCommand::~VoronoiDiagramConstructionCommand()
{

}

void VoronoiDiagramConstructionCommand::execute(poca::core::CommandInfo* _ci)
{
	if (m_dset == NULL) return;
	if (_ci->nameCommand == "computeVoronoi") {
		poca::core::MyObjectInterface* obj = poca::core::Engine::instance()->getObject(m_dset);
		poca::core::MyObjectInterface* oneColorObj = obj->currentObject();
		poca::core::BasicComponentInterface* bci = oneColorObj->getBasicComponent("VoronoiDiagram");
		bool onSphere = _ci->hasParameter("onSphere") && _ci->getParameter<bool>("onSphere");
		if (bci != NULL && !onSphere) {
			if (bci->nbCommands() == 0)
				VoronoiDiagramPlugin::m_plugins->addCommands(bci);
			obj->notifyAll("updateDisplay");
			return;
		}
		poca::geometry::VoronoiDiagramFactoryInterface* factory = poca::geometry::createVoronoiDiagramFactory();
		poca::geometry::VoronoiDiagram* voro = NULL;
		if (!onSphere)
			voro = factory->createVoronoiDiagram(oneColorObj, !VoronoiDiagramPlugin::m_parameters["VoronoiDiagramPlugin"]["createCells"].get<bool>(), VoronoiDiagramPlugin::m_plugins);
		else
			voro = factory->createVoronoiDiagramOnSphere(oneColorObj, !VoronoiDiagramPlugin::m_parameters["VoronoiDiagramPlugin"]["createCells"].get<bool>(), VoronoiDiagramPlugin::m_plugins);
		delete factory;
		obj->notifyAll("updateDisplay");
	}
	
}

poca::core::Command* VoronoiDiagramConstructionCommand::copy()
{
	return NULL;
}

poca::core::CommandInfo VoronoiDiagramConstructionCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "computeVoronoi") {
		poca::core::CommandInfo ci(false, "computeVoronoi");
		if (_parameters.contains("onSphere"))
			ci.addParameter("onSphere", _parameters["onSphere"].get<bool>());
		return ci;
	}
	return poca::core::CommandInfo();
}

nlohmann::json VoronoiDiagramPlugin::m_parameters;
poca::core::PluginList* VoronoiDiagramPlugin::m_plugins = NULL;

void VoronoiDiagramPlugin::addGUI(poca::core::MediatorWObjectFWidgetInterface* _mediator, QTabWidget* _parent)
{
	std::string nameStr = name().toLatin1().data();
	m_parameters[nameStr]["createCells"] = false;
	m_parameters[nameStr]["showTab"] = true;
	m_parameters[nameStr]["voronoiOnSphere"] = false;

	
	const nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
	if(parameters.contains(nameStr)) {
		nlohmann::json param = parameters[nameStr];
		if(param.contains("createCells"))
			m_parameters[nameStr]["createCells"] = param["createCells"].get<bool>();
		if (param.contains("showTab"))
			m_parameters[nameStr]["showTab"] = param["showTab"].get<bool>();
#ifndef NO_PYTHON
		if (param.contains("voronoiOnSphere"))
			m_parameters[nameStr]["voronoiOnSphere"] = param["voronoiOnSphere"].get<bool>();
#endif
	}

	int pos = -1;
	for (int n = 0; n < _parent->count(); n++) {
		std::cout << n << " -> " << _parent->tabText(n).toStdString() << std::endl;
		if (_parent->tabText(n) == "Voronoi")
			pos = n;
	}
	if (pos != -1)
		m_parent = static_cast <QTabWidget*>(_parent->widget(pos));
	else {
		pos = _parent->addTab(new QTabWidget, QObject::tr("Voronoi"));
		m_parent = static_cast <QTabWidget*>(_parent->widget(pos));
	}
	m_widget = new VoronoiDiagramWidget(_mediator, m_parent);
	_mediator->addWidget(m_widget);
	int index = m_parent->insertTab(0, m_widget, QObject::tr("Filtering/Display"));
	m_parent->setCurrentIndex(0);
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	m_parent->setTabVisible(index, false);
#endif
}

std::vector <std::pair<QAction*, QString>> VoronoiDiagramPlugin::getActions()
{
	if (m_actions.empty()) {
		QPixmap pixmap(poca::plot::voronoiIcon);
		QAction* action = new QAction(QIcon(pixmap), tr("Voronoi diagram"), this);
		action->setStatusTip(tr("Compute Voronoi diagram"));

		QAction* action2 = new QAction(tr("Parameters"), this);
		action2->setStatusTip(tr("Set parameters"));

		m_actions.push_back(std::make_pair(action, "Toolbar/1Color"));
		m_actions.push_back(std::make_pair(action2, "Plugins/Voronoi diagram"));
	}
	return m_actions;
}

poca::core::MyObjectInterface* VoronoiDiagramPlugin::actionTriggered(QObject* _sender, poca::core::MyObjectInterface* _obj)
{
	QAction* action = static_cast <QAction*>(_sender);
	std::string nameStr = name().toLatin1().data();
	if (action == m_actions[0].first) {
		poca::core::CommandInfo command(true, "computeVoronoi");
#ifndef NO_PYTHON
		if (_obj->dimension() == 3)
			command.addParameter("onSphere", m_parameters[nameStr]["voronoiOnSphere"].get<bool>());
#endif
		_obj->executeCommandOnSpecificComponent("DetectionSet", &command);
	}
	else if (action == m_actions[1].first) {
		VoronoiDiagramParamDialog* dial = new VoronoiDiagramParamDialog(m_parameters[nameStr]["showTab"].get<bool>(), m_parameters[nameStr]["createCells"].get<bool>(), m_parameters[nameStr]["voronoiOnSphere"].get<bool>());
		dial->setModal(true);
		if (dial->exec() == QDialog::Accepted) {
			m_parameters[nameStr]["createCells"] = dial->isCreateCells();
			m_parameters[nameStr]["showTab"] = dial->isTabShown();
			m_parameters[nameStr]["voronoiOnSphere"] = dial->isVoronoiOnSphere();

#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
			m_parent->setTabVisible(m_parent->indexOf(m_widget), m_parameters[nameStr]["showTab"]);
#endif
		}
		delete dial;
	}
	return NULL;
}

void VoronoiDiagramPlugin::addCommands(poca::core::CommandableObject* _bc)
{
	poca::geometry::VoronoiDiagram* voro = dynamic_cast <poca::geometry::VoronoiDiagram*>(_bc);
	if (voro) {
		voro->addCommand(new VoronoiDiagramDisplayCommand(voro));
		voro->addCommand(new VoronoiDiagramBasicCommands(voro));
		voro->addCommand(new VoronoiDiagramCharacteristicsCommands(voro));
	}

	poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(_bc);
	if (dset)
		dset->addCommand(new VoronoiDiagramConstructionCommand(dset));
}

void VoronoiDiagramPlugin::setSingletons(poca::core::Engine* _engine)
{
	poca::core::Engine::instance()->setEngineSingleton(_engine); poca::core::Engine::instance()->setAllSingletons();
}

void VoronoiDiagramPlugin::execute(poca::core::CommandInfo* _com)
{
	bool done = false;
	if (_com->nameCommand == "saveParameters") {
		if (!_com->hasParameter("file")) return;
		nlohmann::json* json = _com->getParameterPtr<nlohmann::json>("file");

		std::string nameStr = name().toLatin1().data();
		(*json)[nameStr] = m_parameters[nameStr];
		done = true;
	}
	if (done && _com->isRecordable())
		poca::core::MacroRecorderSingleton::instance()->addCommand("VoronoiDiagramPlugin", _com);
}

void VoronoiDiagramPlugin::computeVoronoi(poca::core::MyObjectInterface* _obj)
{
	if (_obj == NULL) return;
	std::string nameStr = name().toLatin1().data();
	poca::core::MyObjectInterface* oneColorObj = _obj->currentObject();
	poca::core::BasicComponentInterface* bci = oneColorObj->getBasicComponent("VoronoiDiagram");
	if (bci != NULL) {
		if (bci->nbCommands() == 0)
			m_plugins->addCommands(bci);
		_obj->notifyAll("updateDisplay");
		return;
	}
	poca::geometry::VoronoiDiagramFactoryInterface* factory = poca::geometry::createVoronoiDiagramFactory();
	poca::geometry::VoronoiDiagram* voro = factory->createVoronoiDiagram(oneColorObj, !m_parameters[nameStr]["createCells"].get<bool>(), m_plugins);
	delete factory;
	_obj->notifyAll("updateDisplay");
}

