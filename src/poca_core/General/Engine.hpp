/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Engine.hpp
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

#ifndef Engine_hpp__
#define Engine_hpp__

#include <QtWidgets/QTabWidget>
#include <vector>
#include <map>
#include <string>
#include <any>

#include <General/json.hpp>

class LoaderInterface;
class GuiInterface;
class PluginInterface;

namespace poca::core {
	class MediatorWObjectFWidget;
	class PluginList;
	class BasicComponentInterface;
	class CommandInfo;
	class MyObjectInterface;
}

namespace poca::opengl {
	class CameraInterface;
}

typedef std::tuple <poca::core::MyObjectInterface*, poca::opengl::CameraInterface*> Dataset;
typedef std::vector <Dataset> Datasets;

namespace poca::core {
	class Engine {
	public:
		static Engine* instance();
		static void deleteInstance();
		void setEngineSingleton(poca::core::Engine*);
		~Engine();

		void initialize(const bool = true);

		void loadPlugin();

		void initializeAllSingletons();
		
		//Used to set the singletons in the plugins
		void setAllSingletons();

		void addGUI(QTabWidget*);

		MyObjectInterface* loadDataAndCreateObject(const QString&, poca::core::CommandInfo* = NULL);
		const bool loadDataAndAddToObject(const QString&, MyObjectInterface*, CommandInfo* = NULL);
		MyObjectInterface* createObject(const std::string&, const std::string&, BasicComponentInterface* = NULL);
		const bool addComponentToObject(MyObjectInterface*, BasicComponentInterface*);
		void addCommands(BasicComponentInterface*);
		BasicComponentInterface* loadData(const QString&, CommandInfo* = NULL, MyObjectInterface* = NULL);
		MyObjectInterface* generateMultipleObject(const std::vector <MyObjectInterface*>&);

		void addCameraToObject(poca::core::MyObjectInterface*, poca::opengl::CameraInterface*);
		void removeDatasetFromList(poca::core::MyObjectInterface*);
		void removeDatasetFromList(poca::opengl::CameraInterface*);
		void removeObject(poca::core::MyObjectInterface*, const bool = true);
		void removeObject(poca::opengl::CameraInterface*, const bool = true);
		void removeCamera(poca::core::MyObjectInterface*, const bool = true);
		void removeCamera(poca::opengl::CameraInterface*, const bool = true);
		void removeObjectAndCamera(poca::core::MyObjectInterface*);
		void removeObjectAndCamera(poca::opengl::CameraInterface*);
		MyObjectInterface* getObject(BasicComponentInterface*);
		MyObjectInterface* getObject(MyObjectInterface*);
		poca::opengl::CameraInterface* getCamera(BasicComponentInterface*);
		poca::opengl::CameraInterface* getCamera(MyObjectInterface*);

		void addData(poca::core::MyObjectInterface*, poca::opengl::CameraInterface*);

		inline const std::any& getSingleton(const std::string& _name) const { return m_singletons.at(_name); }
		inline std::any& getSingleton(const std::string& _name) { return m_singletons.at(_name); }

		inline const std::map <std::string, std::any>& getSingletons() const { return m_singletons; }
		inline MediatorWObjectFWidget* getMediator() { return m_mediator; }
		inline const std::vector < LoaderInterface* >& getLoaders() const { return m_loadersFile; }
		inline PluginList* getPlugins() { return m_plugins; }

		inline void setStateParameters(const nlohmann::json& _param) { m_stateParameters = _param; }
		inline nlohmann::json getStateParameters() const { return m_stateParameters; }
		inline nlohmann::json& getStateParameters() { return m_stateParameters; }

		inline void setGlobalParameters(const nlohmann::json& _param) { m_globalParameters = _param; }
		inline nlohmann::json getGlobalParameters() const { return m_globalParameters; }
		inline nlohmann::json& getGlobalParameters() { return m_globalParameters; }

		inline bool headlessMode() const { return !m_withMainWindow; }
		inline void setMode(const bool _val) { m_withMainWindow = _val; }

	protected:
		Engine();

	private:
		static Engine* m_instance;

		std::vector < LoaderInterface* > m_loadersFile;
		std::vector < GuiInterface* > m_GUIWidgets;
		PluginList* m_plugins{ nullptr };

		std::map <std::string, std::any> m_singletons;

		MediatorWObjectFWidget* m_mediator;

		bool m_withMainWindow{ true };
		Datasets m_datasets;
		Dataset* m_currentDataset{ nullptr };

		//Replacing both StateSoftwareSingleton & GlobalParametersSingleton
		nlohmann::json m_stateParameters, m_globalParameters;
	};
}

#endif

