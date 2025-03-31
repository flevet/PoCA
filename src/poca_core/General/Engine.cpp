/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Engine.cpp
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

#include <fstream>
#include <QtCore/QDir>
#include <QtWidgets/QApplication>
#include <QtCore/QPluginLoader>

#include <OpenGL/Helper.h>
#include <Interfaces/CameraInterface.hpp>

#include "../../include/LoaderInterface.hpp"
#include "../../include/GuiInterface.hpp"
#include "../../include/PluginInterface.hpp"

#include "../DesignPatterns/MacroRecorderSingleton.hpp"
#include "../DesignPatterns/MacroRecorderSingleton.hpp"
#include "../DesignPatterns/MediatorWObjectFWidget.hpp"
#include "../General/Misc.h"
#include "../Interfaces/BasicComponentInterface.hpp"
#include "../Objects/MyObject.hpp"
#include "../Objects/MyMultipleObject.hpp"
#include "../Objects/MyObjectDisplayCommand.hpp"
#include "../General/BasicComponentList.hpp"

#ifndef NO_PYTHON
#include "PythonInterpreter.hpp"
#endif

#include "PluginList.hpp"

#include "Engine.hpp"

namespace poca::core {
	Engine* Engine::m_instance = 0;

	Engine* Engine::instance()
	{
		if (m_instance == 0)
			m_instance = new Engine;
		return m_instance;
	}

	void Engine::setEngineSingleton(poca::core::Engine* _eng)
	{
		m_instance = _eng;
	}

	void Engine::deleteInstance()
	{
		if (m_instance != 0)
			delete m_instance;
		m_instance = 0;
	}

	Engine::Engine()
	{

	}

	Engine::~Engine()
	{
	}

	void Engine::initialize(const bool _withDisplay)
	{
		m_withMainWindow = _withDisplay;
		m_mediator = poca::core::MediatorWObjectFWidget::instance();
		loadPlugin();
		initializeAllSingletons();
		for (auto loader : m_loadersFile)
			loader->setSingletons(this);
		m_plugins->setSingletons(this);
	}

	void Engine::loadPlugin()
	{
		m_plugins = new poca::core::PluginList();
		QDir pluginsDir(QCoreApplication::applicationDirPath());
#if defined(Q_OS_WIN)
		if (pluginsDir.dirName().toLower() == "debug" || pluginsDir.dirName().toLower() == "release")
			pluginsDir.cdUp();
#elif defined(Q_OS_MAC)
		if (pluginsDir.dirName() == "MacOS") {
			pluginsDir.cdUp();
			pluginsDir.cdUp();
			pluginsDir.cdUp();
		}
#endif
		pluginsDir.cd("plugins");
		QString extension(".dll");
#if defined _DEBUG
		pluginsDir.cd("Debug");
		extension.push_front("d");
#endif
		//std::cout << pluginsDir.absolutePath().toLatin1().data() << std::endl;
		const QStringList entries = pluginsDir.entryList(QDir::Files);
		for (const QString& fileName : entries) {
			if (!fileName.endsWith(extension)) continue;
			std::cout << fileName.toStdString() << std::endl;
			QPluginLoader pluginLoader(pluginsDir.absoluteFilePath(fileName));
			QObject* plugin = pluginLoader.instance();
			LoaderInterface* llinterface = NULL;
			GuiInterface* ginterface = NULL;
			PluginInterface* pinterface = NULL;
			if (plugin) {
				llinterface = qobject_cast<LoaderInterface*>(plugin);
				if (llinterface)
					m_loadersFile.push_back(llinterface);
				ginterface = qobject_cast<GuiInterface*>(plugin);
				if (ginterface)
					m_GUIWidgets.push_back(ginterface);
				pinterface = qobject_cast<PluginInterface*>(plugin);
				if (pinterface)
					m_plugins->addPlugin(pinterface);
				if (llinterface == NULL && ginterface == NULL && pinterface == NULL)
					pluginLoader.unload();
			}
		}
		const std::vector <PluginInterface*>& plugs = m_plugins->getPlugins();
		for (PluginInterface* plugin : plugs)
			plugin->setPlugins(m_plugins);
	}

	void Engine::initializeAllSingletons()
	{
		std::ifstream fs("poca.ini");
		if (fs.good())
			fs >> m_globalParameters;
		fs.close();

		poca::core::MediatorWObjectFWidget* med = poca::core::MediatorWObjectFWidget::instance();
		poca::core::MacroRecorderSingleton* macroRecord = poca::core::MacroRecorderSingleton::instance();

		m_singletons["MediatorWObjectFWidget"] = med;
		m_singletons["MacroRecorderSingleton"] = macroRecord;

		if (m_withMainWindow) {
			poca::opengl::HelperSingleton* help = poca::opengl::HelperSingleton::instance();
			m_singletons["HelperSingleton"] = help;
		}

#ifndef NO_PYTHON
		poca::core::PythonInterpreter* python = poca::core::PythonInterpreter::instance();
		m_singletons["PythonInterpreter"] = python;
#endif
	}

	void Engine::setAllSingletons()
	{
		if (m_singletons.find("MediatorWObjectFWidget") != m_singletons.end()) {
			poca::core::MediatorWObjectFWidget::setMediatorWObjectFWidgetSingleron(std::any_cast <poca::core::MediatorWObjectFWidget*>(m_singletons.at("MediatorWObjectFWidget")));
		}
		if (m_singletons.find("MacroRecorderSingleton") != m_singletons.end()) {
			poca::core::MacroRecorderSingleton::setMacroRecorderSingleton(std::any_cast <poca::core::MacroRecorderSingleton*>(m_singletons.at("MacroRecorderSingleton")));
		}
		if (m_singletons.find("HelperSingleton") != m_singletons.end()) {
			poca::opengl::HelperSingleton::setHelperSingleton(std::any_cast <poca::opengl::HelperSingleton*>(m_singletons.at("HelperSingleton")));
		}
	}

	void Engine::addGUI(QTabWidget* _tab)
	{
		for (size_t n = 0; n < m_GUIWidgets.size(); n++)
			m_GUIWidgets[n]->addGUI(m_mediator, _tab);
		m_plugins->addGUI(m_mediator, _tab);
	}

	poca::core::MyObjectInterface* Engine::loadDataAndCreateObject(const QString& _filename, poca::core::CommandInfo* _command)
	{
		poca::core::BasicComponentInterface* bci = loadData(_filename, _command);
		if (bci == NULL)
			return NULL;

		QFileInfo finfo(_filename);
		poca::core::MyObject* wobj = new poca::core::MyObject();
		wobj->setDir(finfo.absolutePath().toStdString());
		wobj->setName(finfo.baseName().toStdString());
		wobj->addBasicComponent(bci);
		wobj->setDimension(bci->dimension());
		wobj->addCommand(new MyObjectDisplayCommand(wobj));

		m_plugins->addCommands(bci);
		BasicComponentList* blist = dynamic_cast<BasicComponentList*>(bci);
		if(blist)
			for(auto bcomp : blist->components())
				m_plugins->addCommands(bcomp);
		m_plugins->addCommands(wobj);

		m_datasets.push_back(std::make_tuple(wobj, nullptr));
		m_currentDataset = &m_datasets.back();

		return wobj;
	}

	const bool Engine::loadDataAndAddToObject(const QString& _filename, poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _command)
	{
		poca::core::BasicComponentInterface* bci = loadData(_filename, _command);
		if (bci == NULL)
			return false;
		m_plugins->addCommands(bci);
		BasicComponentList* blist = dynamic_cast<BasicComponentList*>(bci);
		if (blist)
			for (auto bcomp : blist->components())
				m_plugins->addCommands(bcomp);
		_obj->addBasicComponent(bci);
		return true;
	}

	const bool Engine::addComponentToObject(MyObjectInterface* _obj, BasicComponentInterface* _comp)
	{
		if (_obj == NULL || _comp == NULL)
			return false;
		m_plugins->addCommands(_comp);
		BasicComponentList* blist = dynamic_cast<BasicComponentList*>(_comp);
		if (blist)
			for (auto bcomp : blist->components())
				m_plugins->addCommands(bcomp);
		_obj->addBasicComponent(_comp);
		return true;
	}

	void Engine::addCommands(BasicComponentInterface* _comp)
	{
		m_plugins->addCommands(_comp);
	}

	poca::core::BasicComponentInterface* Engine::loadData(const QString& _filename, poca::core::CommandInfo* _command, poca::core::MyObjectInterface* _obj)
	{
		QFileInfo finfo(_filename);
		if (!finfo.exists())
			return NULL;
		poca::core::BasicComponentInterface* bci = NULL;
		for (auto loader : m_loadersFile) {
			if (!poca::core::utils::isExtensionInList(finfo.suffix(), loader->extensions())) 
				continue;
			bci = loader->loadData(_filename, _command);
			if (bci != NULL)
				return bci;
		}
		return NULL;
	}

	MyObjectInterface* Engine::generateMultipleObject(const std::vector <MyObjectInterface*>& _objs)
	{
		for (poca::core::MyObjectInterface* obj : _objs)
			if (obj == NULL) return NULL;

		poca::core::ChangeManagerSingleton* singleton = poca::core::ChangeManagerSingleton::instance();
		poca::core::CommandInfo ciHeatmap(false, "DetectionSet", "displayHeatmap", false);
		poca::core::CommandInfo ci(false, "All", "freeGPU", true);
		for (poca::core::MyObjectInterface* obj : _objs) {
			removeDatasetFromList(obj);
			obj->executeCommandOnSpecificComponent("DetectionSet", &poca::core::CommandInfo(false, "displayHeatmap", false));
			obj->executeGlobalCommand(&poca::core::CommandInfo(false, "freeGPU"));
			poca::core::SubjectInterface* subject = dynamic_cast <poca::core::SubjectInterface*>(obj);
			if (subject)
				singleton->UnregisterFromAllObservers(subject);
		}

		MyMultipleObject* wobj = new MyMultipleObject(_objs);
		wobj->setDir(_objs[0]->getDir());
		QString name("Colocalization_[");
		for (poca::core::MyObjectInterface* obj : _objs)
			name.append(obj->getName().c_str()).append(",");
		name.append("]");
		wobj->setName(name.toStdString());

		m_plugins->addCommands(wobj);

		m_datasets.push_back(std::make_tuple(wobj, nullptr));
		m_currentDataset = &m_datasets.back();

		return wobj;
	}

	void Engine::addCameraToObject(poca::core::MyObjectInterface* _obj, poca::opengl::CameraInterface* _cam)
	{
		auto it = std::find_if(m_datasets.begin(), m_datasets.end(), [_obj](const std::tuple<poca::core::MyObjectInterface*, poca::opengl::CameraInterface*>& e) {return std::get<0>(e) == _obj; });
		if (it != m_datasets.end())
			std::get<1>(*it) = _cam;
	}

	void Engine::addData(poca::core::MyObjectInterface* _obj, poca::opengl::CameraInterface* _cam)
	{
		m_datasets.push_back(std::make_tuple(_obj, _cam));
	}

	void Engine::removeDatasetFromList(poca::core::MyObjectInterface* _obj)
	{
		auto it = std::find_if(m_datasets.begin(), m_datasets.end(), [_obj](const std::tuple<poca::core::MyObjectInterface*, poca::opengl::CameraInterface*>& e) {return std::get<0>(e) == _obj; });
		if (it != m_datasets.end())
			m_datasets.erase(it);
	}
	
	void Engine::removeDatasetFromList(poca::opengl::CameraInterface* _cam)
	{
		auto it = std::find_if(m_datasets.begin(), m_datasets.end(), [_cam](const std::tuple<poca::core::MyObjectInterface*, poca::opengl::CameraInterface*>& e) {return std::get<1>(e) == _cam; });
		if (it != m_datasets.end())
			m_datasets.erase(it);
	}

	void Engine::removeObject(poca::core::MyObjectInterface* _obj, const bool _removeFromList)
	{
		auto it = std::find_if(m_datasets.begin(), m_datasets.end(), [_obj](const std::tuple<poca::core::MyObjectInterface*, poca::opengl::CameraInterface*>& e) {return std::get<0>(e) == _obj; });
		if (it != m_datasets.end()) {
			delete std::get<0>(*it);
			std::get<0>(*it) = nullptr;
			if(_removeFromList)
				m_datasets.erase(it);
		}
	}

	void Engine::removeObject(poca::opengl::CameraInterface* _cam, const bool _removeFromList)
	{
		auto it = std::find_if(m_datasets.begin(), m_datasets.end(), [_cam](const std::tuple<poca::core::MyObjectInterface*, poca::opengl::CameraInterface*>& e) {return std::get<1>(e) == _cam; });
		if (it != m_datasets.end()) {
			delete std::get<0>(*it);
			std::get<0>(*it) = nullptr;
			if (_removeFromList)
				m_datasets.erase(it);
		}
	}

	void Engine::removeCamera(poca::core::MyObjectInterface* _obj, const bool _removeFromList)
	{
		auto it = std::find_if(m_datasets.begin(), m_datasets.end(), [_obj](const std::tuple<poca::core::MyObjectInterface*, poca::opengl::CameraInterface*>& e) {return std::get<0>(e) == _obj; });
		if (it != m_datasets.end()) {
			delete std::get<1>(*it);
			std::get<1>(*it) = nullptr;
			if (_removeFromList)
				m_datasets.erase(it);
		}
	}

	void Engine::removeCamera(poca::opengl::CameraInterface* _cam, const bool _removeFromList)
	{
		auto it = std::find_if(m_datasets.begin(), m_datasets.end(), [_cam](const std::tuple<poca::core::MyObjectInterface*, poca::opengl::CameraInterface*>& e) {return std::get<1>(e) == _cam; });
		if (it != m_datasets.end()) {
			delete std::get<1>(*it);
			std::get<1>(*it) = nullptr;
			if (_removeFromList)
				m_datasets.erase(it);
		}
	}

	void Engine::removeObjectAndCamera(poca::core::MyObjectInterface* _obj)
	{
		auto it = std::find_if(m_datasets.begin(), m_datasets.end(), [_obj](const std::tuple<poca::core::MyObjectInterface*, poca::opengl::CameraInterface*>& e) {return std::get<0>(e) == _obj; });
		if (it != m_datasets.end()) {
			delete std::get<1>(*it);
			delete std::get<0>(*it);
			m_datasets.erase(it);
		}
	}

	void Engine::removeObjectAndCamera(poca::opengl::CameraInterface* _cam)
	{
		auto it = std::find_if(m_datasets.begin(), m_datasets.end(), [_cam](const std::tuple<poca::core::MyObjectInterface*, poca::opengl::CameraInterface*>& e) {return std::get<1>(e) == _cam; });
		if (it != m_datasets.end()) {
			delete std::get<1>(*it);
			delete std::get<0>(*it);
			m_datasets.erase(it);
		}
	}

	MyObjectInterface* Engine::getObject(BasicComponentInterface* _bci)
	{
		for (auto data : m_datasets) {
			auto obj = std::get<0>(data);
			for (auto n = 0; n < obj->nbColors(); n++) {
				auto curObj = obj->getObject(n);
				BasicComponentInterface* bci = curObj->getBasicComponent(_bci->getName());
				if (bci == _bci)
					return obj;
				for (auto comp : curObj->getComponents()) {
					if (comp->hasComponent(_bci))
						return obj;
				}
			}
		}
		return NULL;
	}

	MyObjectInterface* Engine::getObject(MyObjectInterface* _obj)
	{
		MyObjectInterface* obj = _obj;
		for (auto data : m_datasets) {
			auto obj = std::get<0>(data);
			if (obj->nbColors() == 1) continue;
			for (size_t n = 0; n < obj->nbColors(); n++) {
				MyObjectInterface* obj2 = obj->getObject(n);
				if (obj2 == _obj)
					return obj;
			}
		}
		return obj;
	}

	poca::opengl::CameraInterface* Engine::getCamera(BasicComponentInterface* _bci)
	{
		for (auto data : m_datasets) {
			auto obj = std::get<0>(data);
			auto cam = std::get<1>(data);
			if (!obj->hasBasicComponent(_bci->getName())) continue;
			BasicComponentInterface* bci = obj->getBasicComponent(_bci->getName());
			if (bci == _bci)
				return cam;
		}
		for (auto data : m_datasets) {
			auto obj = std::get<0>(data);
			auto cam = std::get<1>(data);
			for (auto comp : obj->getComponents()) {
				if (comp->hasComponent(_bci))
					return cam;
			}
		}
		return NULL;
	}

	poca::opengl::CameraInterface* Engine::getCamera(MyObjectInterface* _obj)
	{
		MyObjectInterface* obj = _obj;
		for (auto data : m_datasets) {
			auto obj = std::get<0>(data);
			auto cam = std::get<1>(data);
			if (obj->nbColors() == 1) continue;
			for (size_t n = 0; n < obj->nbColors(); n++) {
				MyObjectInterface* obj2 = obj->getObject(n);
				if (obj2 == _obj)
					return cam;
			}
		}
		return NULL;
	}
}