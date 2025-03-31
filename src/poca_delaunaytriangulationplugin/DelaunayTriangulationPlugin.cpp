/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DelaunayTriangulationPlugin.cpp
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
#include <Interfaces/DelaunayTriangulationInterface.hpp>
#include <General/Engine.hpp>
#include <DesignPatterns/MacroRecorderSingleton.hpp>
#include <General/PluginList.hpp>
#include <Plot/Icons.hpp>
#include <Interfaces/DelaunayTriangulationFactoryInterface.hpp>
#include <General/PythonInterpreter.hpp>
#include <General/Engine.hpp>
#include <General/MyData.hpp>

#include "DelaunayTriangulationPlugin.hpp"
#include "DelaunayTriangulationDisplayCommand.hpp"
#include "DelaunayTriangulationBasicCommands.hpp"
#include "DelaunayTriangulationWidget.hpp"
#include "DelaunayTriangulationParamDialog.hpp"

DelaunayTriangulationConstructionCommand::DelaunayTriangulationConstructionCommand(poca::geometry::DetectionSet* _dset) :poca::core::Command("DelaunayTriangulationConstructionCommand")
{
	m_dset = _dset;
}

DelaunayTriangulationConstructionCommand::DelaunayTriangulationConstructionCommand(const DelaunayTriangulationConstructionCommand& _o) : Command(_o)
{
	m_dset = _o.m_dset;
}

DelaunayTriangulationConstructionCommand::~DelaunayTriangulationConstructionCommand()
{

}

void DelaunayTriangulationConstructionCommand::execute(poca::core::CommandInfo* _ci)
{
	if (m_dset == NULL) return;
	if (_ci->nameCommand == "computeDelaunay") {
		poca::core::MyObjectInterface* obj = poca::core::Engine::instance()->getObject(m_dset);
		poca::core::MyObjectInterface* oneColorObj = obj->currentObject();
		poca::core::BasicComponentInterface* bci = oneColorObj->getBasicComponent("DelaunayTriangulation");
		bool onSphere = _ci->hasParameter("onSphere") && _ci->getParameter<bool>("onSphere");
		if (bci != NULL && !onSphere){
			if (bci->nbCommands() == 0)
				DelaunayTriangulationPlugin::m_plugins->addCommands(bci);
			obj->notifyAll("updateDisplay");
			obj->notify("LoadObjCharacteristicsDelaunayTriangulationWidget");
			return;
		}
		poca::geometry::DelaunayTriangulationFactoryInterface* factory = poca::geometry::createDelaunayTriangulationFactory();
		poca::geometry::DelaunayTriangulationInterface* delau = NULL;
		if(!onSphere)
			delau = factory->createDelaunayTriangulation(oneColorObj, DelaunayTriangulationPlugin::m_plugins);
		else {
#ifndef NO_PYTHON
			const std::vector <float>& xs = m_dset->getMyData("x")->getData<float>();
			const std::vector <float>& ys = m_dset->getMyData("y")->getData<float>();
			const std::vector <float>& zs = m_dset->getMyData("z")->getData<float>();

			QVector <QVector <double>> coordinates;
			coordinates.resize(3);
			for (size_t n = 0; n < 3; n++)
				coordinates[n].resize(xs.size());
			for (size_t n = 0; n < xs.size(); n++) {
				coordinates[0][n] = xs[n];
				coordinates[1][n] = ys[n];
				coordinates[2][n] = zs[n];
			}
			QVector <double> coeffs;
			poca::core::PythonInterpreter* py = poca::core::PythonInterpreter::instance();
			bool res = py->applyFunctionWithNArraysParameterAnd1ArrayReturned(coeffs, coordinates, "sphereFit", "sphereFit");
			if (res == EXIT_FAILURE) {
				_ci->errorMessage("python script was not run.");
			}
			else {
				poca::core::Vec3mf centroid(coeffs[0], coeffs[1], coeffs[2]);
				float radius = coeffs[3];

				std::cout << "center: " << centroid << std::endl;
				std::cout << "radius: " << radius << std::endl;

				delau = factory->createDelaunayTriangulationOnSphere(oneColorObj, DelaunayTriangulationPlugin::m_plugins, centroid, radius);
			}
#else
			return;
#endif
		}
		delete factory;
		obj->notifyAll("updateDisplay");
		obj->notify("LoadObjCharacteristicsDelaunayTriangulationWidget");
	}

}

poca::core::Command* DelaunayTriangulationConstructionCommand::copy()
{
	return NULL;
}

poca::core::CommandInfo DelaunayTriangulationConstructionCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "computeDelaunay") {
		poca::core::CommandInfo ci(false, "computeDelaunay");
		if (_parameters.contains("onSphere"))
			ci.addParameter("onSphere", _parameters["onSphere"].get<bool>());
		return ci;
	}
	return poca::core::CommandInfo();
}

poca::core::PluginList* DelaunayTriangulationPlugin::m_plugins = NULL;
nlohmann::json DelaunayTriangulationPlugin::m_parameters;

void DelaunayTriangulationPlugin::addGUI(poca::core::MediatorWObjectFWidgetInterface* _mediator, QTabWidget* _parent)
{
	std::string nameStr = name().toLatin1().data();
	m_parameters[nameStr]["delaunay3D"] = true;
	m_parameters[nameStr]["delaunayOnSphere"] = false;

#ifndef NO_PYTHON
	
	const nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
	if (parameters.contains(nameStr)) {
		nlohmann::json param = parameters[nameStr];
		if (param.contains("delaunay3D"))
			m_parameters[nameStr]["delaunay3D"] = param["delaunay3D"].get<bool>();
		if (param.contains("delaunayOnSphere"))
			m_parameters[nameStr]["delaunayOnSphere"] = param["delaunayOnSphere"].get<bool>();
	}
#endif

	m_parent = _parent->findChild <QTabWidget*>("Delaunay");
	if (m_parent == NULL) {
		int pos = _parent->addTab(new QTabWidget, QObject::tr("Delaunay"));
		m_parent = static_cast <QTabWidget*>(_parent->widget(pos));
	}
	DelaunayTriangulationWidget* dtw = new DelaunayTriangulationWidget(_mediator, m_parent);
	_mediator->addWidget(dtw);
	int index = m_parent->addTab(dtw, QObject::tr("Filtering/Display"));
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	m_parent->setTabVisible(index, false);
#endif
}

std::vector <std::pair<QAction*, QString>> DelaunayTriangulationPlugin::getActions()
{
	if (m_actions.empty()) {
		QPixmap pixmap(poca::plot::delaunayIcon);
		QAction* action = new QAction(QIcon(pixmap), tr("Delaunay triangulation"), this);
		action->setStatusTip(tr("Compute Delaunay triangulation"));
		m_actions.push_back(std::make_pair(action, "Toolbar/1Color"));

#ifndef NO_PYTHON
		QAction* action2 = new QAction(tr("Parameters"), this);
		action2->setStatusTip(tr("Set parameters"));
		m_actions.push_back(std::make_pair(action2, "Plugins/Delaunay Triangulation"));
#endif
	}
	return m_actions;
}

poca::core::MyObjectInterface* DelaunayTriangulationPlugin::actionTriggered(QObject* _sender, poca::core::MyObjectInterface* _obj)
{
	QAction* action = static_cast <QAction*>(_sender);
	std::string nameStr = name().toLatin1().data();
	if (action == m_actions[0].first) {
		poca::core::CommandInfo command(true, "computeDelaunay");
#ifndef NO_PYTHON
		if (_obj->dimension() == 3)
			command.addParameter("onSphere", m_parameters[nameStr]["delaunayOnSphere"].get<bool>());
#endif
		_obj->executeCommandOnSpecificComponent("DetectionSet", &command);
	}
	else if (m_actions.size() > 1 && action == m_actions[1].first) {
		DelaunayTriangulationParamDialog* dial = new DelaunayTriangulationParamDialog(m_parameters[nameStr]["delaunay3D"].get<bool>(), m_parameters[nameStr]["delaunayOnSphere"].get<bool>());
		dial->setModal(true);
		if (dial->exec() == QDialog::Accepted) {
			m_parameters[nameStr]["delaunay3D"] = dial->isDelaunay3D();
			m_parameters[nameStr]["delaunayOnSphere"] = dial->isDelaunayOnSphere();
		}
		delete dial;
	}
	return NULL;
}

void DelaunayTriangulationPlugin::addCommands(poca::core::CommandableObject* _bc)
{
	poca::geometry::DelaunayTriangulationInterface* delau = dynamic_cast <poca::geometry::DelaunayTriangulationInterface*>(_bc);
	if (delau) {
		delau->addCommand(new DelaunayTriangulationDisplayCommand(delau));
		delau->addCommand(new DelaunayTriangulationBasicCommands(delau));
	}

	poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(_bc);
	if (dset)
		dset->addCommand(new DelaunayTriangulationConstructionCommand(dset));
}

void DelaunayTriangulationPlugin::setSingletons(poca::core::Engine* _engine)
{
	poca::core::Engine::instance()->setEngineSingleton(_engine); poca::core::Engine::instance()->setAllSingletons();
#ifndef NO_PYTHON
	poca::core::PythonInterpreter* python = std::any_cast <poca::core::PythonInterpreter*>(_engine->getSingleton("PythonInterpreter"));
	poca::core::PythonInterpreter::setPythonInterpreterSingleton(python);
#endif
}

