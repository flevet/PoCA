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
#include <algorithm>
#include <cctype>
#include <QtCore/QDir>
#include <QtWidgets/QApplication>
#include <QtCore/QPluginLoader>
#include <QtCore/QVariant>

#include <OpenGL/Helper.h>
#include <Interfaces/CameraInterface.hpp>

#include "../../include/LoaderInterface.hpp"
#include "../../include/GuiInterface.hpp"
#include "../../include/PluginInterface.hpp"

#include "../DesignPatterns/MacroRecorderSingleton.hpp"
#include "../DesignPatterns/MacroRecorderSingleton.hpp"
#include "../DesignPatterns/MediatorWObjectFWidget.hpp"
#include "../General/Command.hpp"
#include "../General/Misc.h"
#include "../Interfaces/BasicComponentInterface.hpp"
#include "../Objects/MyObject.hpp"
#include "../Objects/MyMultipleObject.hpp"
#include "../Objects/MyObjectDisplayCommand.hpp"
#include "../General/BasicComponentList.hpp"
#include "../General/Engine.hpp"
#include "../General/ImagesList.hpp"

#ifndef NO_PYTHON
#include "PythonInterpreter.hpp"
#endif

#include "PluginList.hpp"

#include "Engine.hpp"

namespace poca::core {
	Engine* Engine::m_instance = 0;

	namespace {
		nlohmann::json coerceTypedCommandParameterValue(const nlohmann::json& _typedValue)
		{
			nlohmann::json rawValue = poca::core::commandParameterJsonValue(_typedValue);
			const std::string type = _typedValue["type"].get<std::string>();
			try {
				if (type == "boolean") {
					if (rawValue.is_boolean())
						return rawValue;
					if (rawValue.is_number_integer())
						return rawValue.get<int>() != 0;
					if (rawValue.is_string()) {
						std::string value = rawValue.get<std::string>();
						std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return (char)std::tolower(c); });
						if (value == "true" || value == "1")
							return true;
						if (value == "false" || value == "0")
							return false;
					}
				}
				else if (type == "number") {
					if (rawValue.is_number())
						return rawValue;
					if (rawValue.is_string())
						return std::stof(rawValue.get<std::string>());
				}
				else if (type == "integer") {
					if (rawValue.is_number_integer())
						return rawValue;
					if (rawValue.is_string())
						return std::stoi(rawValue.get<std::string>());
				}
				else if (type == "unsignedInteger") {
					if (rawValue.is_number_unsigned() || (rawValue.is_number_integer() && rawValue.get<long long>() >= 0))
						return rawValue;
					if (rawValue.is_string()) {
						const long value = std::stol(rawValue.get<std::string>());
						if (value >= 0)
							return value;
					}
				}
				else if (type == "string") {
					if (rawValue.is_string())
						return rawValue;
					return rawValue.dump();
				}
			}
			catch (const std::exception&) {
			}
			return rawValue;
		}

		void normalizeLoadedCommandParameters(nlohmann::json& _value)
		{
			if (poca::core::isTypedCommandParameterJson(_value)) {
				nlohmann::json rawValue = coerceTypedCommandParameterValue(_value);
				_value = rawValue;
				normalizeLoadedCommandParameters(_value);
				return;
			}
			if (_value.is_object()) {
				for (auto& item : _value.items())
					normalizeLoadedCommandParameters(item.value());
			}
			else if (_value.is_array()) {
				for (auto& item : _value)
					normalizeLoadedCommandParameters(item);
			}
		}
	}

	namespace {
		const char* engineApplicationPropertyName()
		{
			return "poca.core.Engine.instance";
		}

		Engine* engineFromApplicationProperty()
		{
			QCoreApplication* app = QCoreApplication::instance();
			if (app == nullptr)
				return nullptr;

			QVariant value = app->property(engineApplicationPropertyName());
			if (!value.isValid())
				return nullptr;

			return reinterpret_cast<Engine*>(value.value<quintptr>());
		}

		void setEngineApplicationProperty(Engine* _engine)
		{
			QCoreApplication* app = QCoreApplication::instance();
			if (app != nullptr)
				app->setProperty(engineApplicationPropertyName(), QVariant::fromValue<quintptr>(reinterpret_cast<quintptr>(_engine)));
		}
	}

	Engine* Engine::instance()
	{
		/*
		 * poca_core is linked as a static library by the application and by plugins.
		 * A plain static Engine* is therefore duplicated per module. Store the real
		 * process-wide Engine pointer on QCoreApplication, exactly as the shared
		 * PerformanceProfiler does, so verbose state and verbose type filters are
		 * read consistently from plugins/opengl workers as well as from the main UI.
		 */
		Engine* sharedEngine = engineFromApplicationProperty();
		if (sharedEngine != nullptr) {
			m_instance = sharedEngine;
			return m_instance;
		}

		if (m_instance == 0) {
			m_instance = new Engine;
			setEngineApplicationProperty(m_instance);
		}
		return m_instance;
	}

	void Engine::setEngineSingleton(poca::core::Engine* _eng)
	{
		m_instance = _eng;
		setEngineApplicationProperty(m_instance);
	}

	void Engine::deleteInstance()
	{
		QCoreApplication* app = QCoreApplication::instance();
		Engine* sharedEngine = engineFromApplicationProperty();
		if (app != nullptr && sharedEngine == m_instance)
			app->setProperty(engineApplicationPropertyName(), QVariant());

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
		m_tests.clear();
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
				if (llinterface) {
					m_loadersFile.push_back(llinterface);
					m_fileExtensions << llinterface->extensions();
				}
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
		normalizeLoadedCommandParameters(m_globalParameters);
		initializePalettes();
		if (m_globalParameters.contains("Preferences")) {
			const nlohmann::json& preferences = m_globalParameters["Preferences"];
			if (preferences.contains("verbose"))
				m_verbose = preferences["verbose"].get<bool>();
			if (preferences.contains("verboseTypes") && preferences["verboseTypes"].is_array()) {
				m_verboseTypes.clear();
				for (const auto& type : preferences["verboseTypes"])
					if (type.is_string())
						addVerboseType(type.get<std::string>());
			}
		}
		/*
		 * Publish the loaded verbose state through QCoreApplication as soon as the
		 * ini file is read. This avoids static-library/plugin copies of Engine from
		 * falling back to their constructor defaults before the menu state changes.
		 */
		publishVerboseTypes();
		setVerbose(m_verbose);

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


	void Engine::initializePalettes()
	{
		m_palettes.clear();
		if (m_globalParameters.contains("Palettes") && m_globalParameters["Palettes"].is_object()) {
			for (const auto& item : m_globalParameters["Palettes"].items()) {
				const nlohmann::json& palJson = item.value();
				if (!palJson.contains("points") || !palJson["points"].is_array()) continue;
				poca::core::Palette palette;
				palette.setName(item.key());
				if (palJson.contains("hilow") && palJson["hilow"].is_boolean()) palette.setHiLow(palJson["hilow"].get<bool>());
				for (const auto& point : palJson["points"]) {
					if (!point.contains("position") || !point.contains("color") || !point["color"].is_array() || point["color"].size() < 3) continue;
					float position = point["position"].get<float>();
					unsigned char r = point["color"][0].get<unsigned char>();
					unsigned char g = point["color"][1].get<unsigned char>();
					unsigned char b = point["color"][2].get<unsigned char>();
					unsigned char a = point["color"].size() > 3 ? point["color"][3].get<unsigned char>() : 255;
					palette.setColor(position, Color4uc(r, g, b, a));
				}
				if (!palette.null()) m_palettes[item.key()] = palette;
			}
		}
		if (m_palettes.empty()) {
			for (const std::string& name : poca::core::Palette::getStaticLutNames()) {
				poca::core::Palette palette = poca::core::Palette::getStaticLut(name);
				if (!palette.null()) m_palettes[name] = palette;
			}
		}
	}

	poca::core::Palette* Engine::palette(const std::string& _name)
	{
		auto it = m_palettes.find(_name);
		return it == m_palettes.end() ? nullptr : &it->second;
	}

	void Engine::addOrReplacePalette(const std::string& _name, const poca::core::Palette& _palette)
	{
		poca::core::Palette palette(_palette);
		palette.setName(_name);
		m_palettes[_name] = palette;
	}

	void Engine::removePalette(const std::string& _name)
	{
		m_palettes.erase(_name);
	}

	void Engine::savePalettesToGlobalParameters()
	{
		nlohmann::json palettesJson = nlohmann::json::object();
		for (const auto& item : m_palettes) {
			std::vector<float> positions;
			std::vector<poca::core::Color4uc> colors;
			item.second.getGradientInfos(positions, colors);
			nlohmann::json palJson;
			palJson["hilow"] = item.second.isHiLow();
			palJson["points"] = nlohmann::json::array();
			for (size_t n = 0; n < positions.size(); n++) {
				nlohmann::json point;
				point["position"] = positions[n];
				point["color"] = { colors[n][0], colors[n][1], colors[n][2], colors[n][3] };
				palJson["points"].push_back(point);
			}
			palettesJson[item.first] = palJson;
		}
		m_globalParameters["Palettes"] = palettesJson;
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
		wobj->setName(finfo.completeBaseName().toStdString());
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

	MyObjectInterface* Engine::createObject(const std::string& _dir, const std::string& _name, BasicComponentInterface* _bci)
	{
		poca::core::MyObject* wobj = new poca::core::MyObject();
		wobj->setDir(_dir);
		wobj->setName(_name);
		wobj->addCommand(new MyObjectDisplayCommand(wobj));
		if (_bci != NULL) {
			addComponentToObject(wobj, _bci);
			wobj->setDimension(_bci->dimension());
		}
		m_plugins->addCommands(wobj);

		m_datasets.push_back(std::make_tuple(wobj, nullptr));
		m_currentDataset = &m_datasets.back();
		return wobj;
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

	bool Engine::addComponentToComponentList(MyObjectInterface* _obj, const std::string& _componentListName, BasicComponentInterface* _comp)
	{
		if (!_obj->hasBasicComponent(_componentListName))
			return false;

		BasicComponentList* blist = dynamic_cast<BasicComponentList*>(_obj->getBasicComponent(_componentListName));
		if (!blist)
			return false;

		BasicComponent* bc = dynamic_cast <BasicComponent*>(_comp);
		if (!bc)
			return false;

		m_plugins->addCommands(_comp);
		blist->addComponent(bc);

		return true;
	}

	bool Engine::addComponentToComponentList(BasicComponentList* _list, BasicComponent* _comp)
	{
		m_plugins->addCommands(_comp);
		_list->addComponent(_comp);
		return true;
	}

	void Engine::addCommands(BasicComponentInterface* _comp)
	{
		m_plugins->addCommands(_comp);
	}

	poca::core::BasicComponentInterface* Engine::loadData(const QString& _filename, poca::core::CommandInfo* _command, poca::core::MyObjectInterface* _obj)
	{
		poca::core::Engine* engine = poca::core::Engine::instance();
		
		QFileInfo finfo(_filename);
		if (!finfo.exists())
			return NULL;
		poca::core::BasicComponentInterface* bci = NULL;
		for (auto loader : m_loadersFile) {
			if (engine->verbose())
			{
				std::cout << finfo.suffix().toStdString();
				for (auto val : loader->extensions())
					std::cout << " - " << val.toStdString();
				std::cout << std::endl;
			}
			if (!poca::core::utils::isExtensionInList(finfo.suffix(), loader->extensions())) 
				continue;
			bci = loader->loadData(_filename, _command);
			if (bci != NULL)
				return bci;
		}
		return NULL;
	}

	BasicComponentList* Engine::mergeComponentLists(BasicComponentInterface* _l1, BasicComponentInterface* _l2)
	{
		BasicComponentList* l1 = dynamic_cast <BasicComponentList*>(_l1), *l2 = dynamic_cast <BasicComponentList*>(_l2);
		if (l1 != NULL && l2 != NULL && typeid(*_l1) == typeid(*_l2)) {
			l1->copyComponentsPtr(l2);
			l2->dontDeleteComponents();
			delete _l2;
			return l1;
		}
		else
			return NULL;
	}

	MyObjectInterface* Engine::generateMultipleObject(const std::vector <MyObjectInterface*>& _objs, const bool _batchComponentRendering)
	{
		for (poca::core::MyObjectInterface* obj : _objs)
			if (obj == NULL) return NULL;

		poca::core::ChangeManagerSingleton* singleton = poca::core::ChangeManagerSingleton::instance();
		poca::core::CommandInfo ciHeatmap(false, "DetectionSet", "displayHeatmap", false);
		poca::core::CommandInfo ci(false, "All", "freeGPU", true);
		for (poca::core::MyObjectInterface* obj : _objs) {
			auto cam = getCamera(obj);
			if (cam != nullptr)
				cam->makeCurrent();
			obj->executeCommandOnSpecificComponent("DetectionSet", &poca::core::CommandInfo(false, "displayHeatmap", false));
			obj->executeGlobalCommand(&poca::core::CommandInfo(false, "freeGPU"));
			poca::core::SubjectInterface* subject = dynamic_cast <poca::core::SubjectInterface*>(obj);
			if (subject)
				singleton->UnregisterFromAllObservers(subject);
			removeDatasetFromList(obj);
		}

		MyMultipleObject* wobj = new MyMultipleObject(_objs, _batchComponentRendering);
		wobj->setDir(_objs[0]->getDir());
		QString name("Colocalization_[");
		for (poca::core::MyObjectInterface* obj : _objs)
			name.append(obj->getName().c_str()).append(",");
		name.append("]");
		wobj->setName(name.toStdString());
		wobj->addCommand(new MyObjectDisplayCommand(wobj));

		m_plugins->addCommands(wobj);

		m_datasets.push_back(std::make_tuple(wobj, nullptr));
		m_currentDataset = &m_datasets.back();

		return wobj;
	}

	void Engine::setCurrentObject(MyObjectInterface* _obj)
	{
		auto it = std::find_if(m_datasets.begin(), m_datasets.end(), [_obj](const std::tuple<poca::core::MyObjectInterface*, poca::opengl::CameraInterface*>& e) {return std::get<0>(e) == _obj; });
		if (it != m_datasets.end())
			m_currentDataset = &(*it);
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

	MyObjectInterface* Engine::getTopObject(BasicComponentInterface* _bci)
	{
		for (auto data : m_datasets) {
			auto obj = std::get<0>(data);
			for (auto n = 0; n < obj->nbColors(); n++) {
				auto curObj = obj->getObject(n);
				for (auto n = 0; n < curObj->nbBasicComponents(); n++) {
					auto comp = curObj->getBasicComponent(n);
					if (_bci == comp)
						return obj;
				}
				for (auto n = 0; n < curObj->nbBasicComponents(); n++) {
					poca::core::BasicComponentList* blist = dynamic_cast <poca::core::BasicComponentList*>(curObj->getBasicComponent(n));
					if (blist) {
						for (auto i = 0; i < blist->nbComponents(); i++) {
							auto comp = blist->getComponent(i);
							if (_bci == comp)
								return obj;
						}
					}
				}
			}
		}
		return NULL;
	}

	MyObjectInterface* Engine::getObject(BasicComponentInterface* _bci)
	{
		for (auto data : m_datasets) {
			auto obj = std::get<0>(data);
			for (auto n = 0; n < obj->nbColors(); n++) {
				auto curObj = obj->getObject(n);
				for (auto n = 0; n < curObj->nbBasicComponents(); n++) {
					auto comp = curObj->getBasicComponent(n);
					if (_bci == comp)
						return curObj;
				}
				for (auto n = 0; n < curObj->nbBasicComponents(); n++) {
					poca::core::BasicComponentList* blist = dynamic_cast <poca::core::BasicComponentList*>(curObj->getBasicComponent(n));
					if (blist) {
						for (auto i = 0; i < blist->nbComponents(); i++) {
							auto comp = blist->getComponent(i);
							if (_bci == comp)
								return curObj;
						}
					}
				}
				/*BasicComponentInterface* bci = curObj->getBasicComponent(_bci->getName());
					if (bci == _bci)
						return obj;
					for (auto comp : curObj->getComponents()) {
						if (comp->hasComponent(_bci))
							return obj;
					}*/
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
		auto topObj = getTopObject(_bci);
		for (auto data : m_datasets) {
			auto obj = std::get<0>(data);
			auto cam = std::get<1>(data);
			if (obj == topObj)
				return cam;
		}
		/*for (auto data : m_datasets) {
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
		}*/
		return NULL;
	}

	poca::opengl::CameraInterface* Engine::getCamera(MyObjectInterface* _obj)
	{
		for (auto data : m_datasets) {
			auto obj = std::get<0>(data);
			auto cam = std::get<1>(data);
			if (obj == _obj)
				return cam;
			for (size_t n = 0; n < obj->nbColors(); n++) {
				MyObjectInterface* obj2 = obj->getObject(n);
				if (obj2 == _obj)
					return cam;
			}
		}
		return NULL;
	}

	void Engine::runMacro(std::vector<nlohmann::json> _macro)
	{
		for (auto json : _macro) {
			if (json.empty()) continue;

			const auto nameComp = json.begin().key();
			if (nameComp == "Engine")
				runMacro(json[nameComp]);
			else {
				poca::core::CommandableObject* comObj = NULL;
				poca::core::MyObjectInterface* obj = std::get<0>(*m_currentDataset);
				if (nameComp == "Object")
					comObj = dynamic_cast<poca::core::CommandableObject*>(obj);
				else {
					comObj = dynamic_cast<poca::core::CommandableObject*>(obj->getBasicComponent(nameComp));
				}
				if (comObj != NULL) {
					nlohmann::json jsonCommand = json[nameComp];
					for (auto& [nameCommand, value] : jsonCommand.items()) {
						nlohmann::json parameters;
						poca::core::CommandInfo command = comObj->createCommand(nameCommand, jsonCommand[nameCommand]);
						if (!command.empty()) {
							comObj->executeCommand(&command);
						}
						else
							std::cout << "Component [" << nameComp << "], command [" << nameCommand << "] does not exist, command " << jsonCommand.dump() << " was not executed." << std::endl;
					}
				}
				else
					std::cout << "Component [" << nameComp << "] does not exist, command " << json.dump() << " was not executed." << std::endl;
			}

		}
	}

	void Engine::runMacro(std::vector<nlohmann::json> _macro, QStringList _filenames)
	{
		for (auto filename : _filenames) {
			for (auto json : _macro) {
				if (json.empty()) continue;

				const auto nameComp = json.begin().key();
				if (nameComp == "MainWindow") {
					auto command = json[nameComp];
					if (command.contains("open")) {
						//if (command["open"].contains("path"))
						command["open"]["path"] = filename.toStdString();
					}
					runMacro(command);
				}
				else {
					poca::core::CommandableObject* comObj = NULL;
					if (nameComp == "Object")
						comObj = dynamic_cast<poca::core::CommandableObject*>(std::get<0>(*m_currentDataset));
					else {
						poca::core::MyObjectInterface* obj = std::get<0>(*m_currentDataset);
						comObj = dynamic_cast<poca::core::CommandableObject*>(obj->getBasicComponent(nameComp));
					}
					if (comObj != NULL) {
						nlohmann::json jsonCommand = json[nameComp];
						for (auto& [nameCommand, value] : jsonCommand.items()) {
							nlohmann::json parameters;
							poca::core::CommandInfo command = comObj->createCommand(nameCommand, jsonCommand[nameCommand]);
							if (!command.empty()) {
								comObj->executeCommand(&command);
							}
							else
								std::cout << "Component [" << nameComp << "], command [" << nameCommand << "] does not exist, command " << jsonCommand.dump() << " was not executed." << std::endl;
						}
					}
					else
						std::cout << "Component [" << nameComp << "] does not exist, command " << json.dump() << " was not executed." << std::endl;
				}

			}
		}
	}

	void Engine::runMacro(const nlohmann::json& _json)
	{
		if (_json.empty()) return;
		const auto tmp = _json.begin().key();
		if (tmp == "open") {
			poca::core::CommandInfo command(false, tmp);

			for (auto& [key, value] : _json[tmp].items()) {
				if (key == "path")
					command.addParameter(key, _json[tmp][key].get<std::string>());
				else if (key == "calibration_xy")
					command.addParameter(key, _json[tmp][key].get<float>());
				else if (key == "calibration_xy")
					command.addParameter(key, _json[tmp][key].get<float>());
				else if (key == "calibration_z")
					command.addParameter(key, _json[tmp][key].get<float>());
				else if (key == "calibration_t")
					command.addParameter(key, _json[tmp][key].get<float>());
				else if (key == "separator")
					command.addParameter(key, _json[tmp][key].get<char>());
				else
					command.addParameter(key, _json[tmp][key].get<size_t>());
			}

			//execute(&command);
		}
	}

	void Engine::executeCommand(BasicComponentInterface* _bci, const bool _record, const std::string& _nameCommand)
	{
		CommandInfo ci(_record, _nameCommand);
		executeCommand(_bci, &ci);
	}

	void Engine::executeCommand(BasicComponentInterface* _bci, CommandInfo* _com)
	{
		CommandExecutionContext context;
		CommandExecutionResult result;
		executeCommand(_bci, _com, context, result);
	}

	void Engine::executeCommand(BasicComponentInterface* _bci, CommandInfo* _com, const CommandExecutionContext& _context)
	{
		CommandExecutionResult result;
		executeCommand(_bci, _com, _context, result);
	}

	void Engine::executeCommand(BasicComponentInterface* _bci, CommandInfo* _com, const CommandExecutionContext& _context, CommandExecutionResult& _result)
	{
		if (_bci == NULL || _com == NULL)
			return;

		auto object = getTopObject(_bci);
		const bool refreshBatchRenderer = _com->nameCommand == "changeLUT" ||
			_com->nameCommand == "ellipsoidRendering" || _com->nameCommand == "histogram" ||
			_com->nameCommand == "updateFeature" || _com->nameCommand == "selected" ||
			_com->nameCommand == "updateTransform" || _com->nameCommand == "useVertexNormals" ||
			_com->nameCommand == "freeGPU" || _com->nameCommand == "rebuildDisplay";
		if (_com != nullptr && _com->nameCommand == "histogram" && _com->hasParameter("action")) {
			const std::string action = _com->getParameter<std::string>("action");
			if (action == "save" && !_com->hasParameter("dir") && object != NULL)
				_com->addParameter("dir", object->getDir());
		}

		if (object != NULL && object->nbColors() > 1 && m_globalCommands) {
			if (object->nbColors() > 1) {
				std::vector<size_t> indices;
				MyMultipleObject* multipleObject = dynamic_cast<MyMultipleObject*>(object);
				BasicComponentList* sourceList = dynamic_cast<BasicComponentList*>(_bci);
				const uint32_t sourceListIndex = sourceList != NULL ? sourceList->currentComponentIndex() : 0;
				if (multipleObject != NULL && multipleObject->hasSelectedObjectIndices())
					indices = multipleObject->selectedObjectIndices();
				else {
					indices.resize(object->nbColors());
					for (size_t n = 0; n < object->nbColors(); ++n)
						indices[n] = n;
				}
				for (const size_t n : indices) {
					if (n >= object->nbColors())
						continue;
					auto obj = object->getObject(n);
					if (obj->hasBasicComponent(_bci->getName())) {
						auto bc = obj->getBasicComponent(_bci->getName());
						BasicComponentList* targetList = dynamic_cast<BasicComponentList*>(bc);
						if (sourceList != NULL && (targetList == NULL || sourceListIndex >= targetList->nbComponents()))
							continue;
						if (targetList != NULL)
							targetList->setCurrentComponentIndex(sourceListIndex);
						CommandableObject* co = static_cast <CommandableObject*>(bc);
						co->executeCommand(_com, _context, _result);
					}
				}
				if (multipleObject != NULL && multipleObject->batchComponentRendering() && refreshBatchRenderer)
					multipleObject->poca::core::CommandableObject::executeCommand(_com, _context, _result);
			}
		}
		else {
			//auto obj = getObject(_bci);
			//if (obj == NULL) return;
			CommandableObject* co = static_cast <CommandableObject*>(_bci);
			co->executeCommand(_com, _context, _result);
			MyMultipleObject* multipleObject = dynamic_cast<MyMultipleObject*>(object);
			if (multipleObject != NULL && multipleObject->batchComponentRendering() && refreshBatchRenderer)
				multipleObject->poca::core::CommandableObject::executeCommand(_com, _context, _result);
		}
	}
	
	MyObjectInterface* Engine::createObjectFromImages(
		const std::string& _dir,
		const std::string& _name,
		const std::vector<std::pair<ImageInterface*, std::string>>& _images)
	{
		if (_images.empty())
			return nullptr;

		ImagesList* images = new ImagesList(_images.front().first, _images.front().second);
		for (size_t n = 1; n < _images.size(); n++)
			images->addImage(_images[n].first, _images[n].second);

		return createObject(_dir, _name, images);
	}
}
