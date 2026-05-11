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
#include <algorithm>
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
	class BasicComponentList;
	class BasicComponent;
	class CommandExecutionContext;
	class CommandExecutionResult;
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
		bool addComponentToComponentList(MyObjectInterface*, const std::string&, BasicComponentInterface*);
		bool addComponentToComponentList(BasicComponentList*, BasicComponent*);
		void addCommands(BasicComponentInterface*);
		BasicComponentInterface* loadData(const QString&, CommandInfo* = NULL, MyObjectInterface* = NULL);
		MyObjectInterface* generateMultipleObject(const std::vector <MyObjectInterface*>&);

		BasicComponentList* mergeComponentLists(BasicComponentInterface*, BasicComponentInterface*);

		void addCameraToObject(poca::core::MyObjectInterface*, poca::opengl::CameraInterface*);
		void removeDatasetFromList(poca::core::MyObjectInterface*);
		void removeDatasetFromList(poca::opengl::CameraInterface*);
		void removeObject(poca::core::MyObjectInterface*, const bool = true);
		void removeObject(poca::opengl::CameraInterface*, const bool = true);
		void removeCamera(poca::core::MyObjectInterface*, const bool = true);
		void removeCamera(poca::opengl::CameraInterface*, const bool = true);
		void removeObjectAndCamera(poca::core::MyObjectInterface*);
		void removeObjectAndCamera(poca::opengl::CameraInterface*);
		MyObjectInterface* getTopObject(BasicComponentInterface*);
		MyObjectInterface* getObject(BasicComponentInterface*);
		MyObjectInterface* getObject(MyObjectInterface*);
		poca::opengl::CameraInterface* getCamera(BasicComponentInterface*);
		poca::opengl::CameraInterface* getCamera(MyObjectInterface*);
		void setCurrentObject(MyObjectInterface*);

		void addData(poca::core::MyObjectInterface*, poca::opengl::CameraInterface*);

		void runMacro(std::vector<nlohmann::json>);
		void runMacro(std::vector<nlohmann::json>, QStringList);
		void runMacro(const nlohmann::json&);

		void executeCommand(BasicComponentInterface*, const bool, const std::string&);
		void executeCommand(BasicComponentInterface*, CommandInfo*);
		void executeCommand(BasicComponentInterface*, CommandInfo*, const CommandExecutionContext&);
		void executeCommand(BasicComponentInterface*, CommandInfo*, const CommandExecutionContext&, CommandExecutionResult&);

		template<typename T>
		void executeCommand(BasicComponentInterface* _bci, const bool _record, const std::string& _name, const T& _param) {
			CommandInfo ci(_record, _name, _param);
			executeCommand(_bci, &ci);
		}

		template<typename T, typename... Args>
		void executeCommand(BasicComponentInterface* _bci, const bool _record, const std::string& _nameCommand, const std::string& _nameParameter, const T& _param, Args... more) {
			CommandInfo ci(_record, _nameCommand);
			ci.addParameters(_nameParameter, _param, more...);
			executeCommand(_bci, &ci);
		}

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

		inline const QStringList& extensions() const { return m_fileExtensions; }

		inline bool verbose(const std::string& _type = "") const {
			if (!m_verbose)
				return m_verbose;
			if (_type.empty())
				return m_verbose;
			if (m_verboseTypes.empty())
				return m_verbose;
			return std::find(m_verboseTypes.begin(), m_verboseTypes.end(), _type) != m_verboseTypes.end();
		}
		inline void setVerbose(const bool _val) { m_verbose = _val; }
		inline bool verboseEnabled() const { return m_verbose; }
		inline void addVerboseType(const std::string& _type) {
			if (std::find(m_verboseTypes.begin(), m_verboseTypes.end(), _type) == m_verboseTypes.end())
				m_verboseTypes.push_back(_type);
		}
		inline bool hasVerboseType(const std::string& _type) const {
			return std::find(m_verboseTypes.begin(), m_verboseTypes.end(), _type) != m_verboseTypes.end();
		}
		inline void removeVerboseType(const std::string& _type) {
			m_verboseTypes.erase(std::remove(m_verboseTypes.begin(), m_verboseTypes.end(), _type), m_verboseTypes.end());
		}
		inline void clearVerboseTypes() { m_verboseTypes.clear(); }
		inline const std::vector<std::string>& verboseTypes() const { return m_verboseTypes; }

		inline const bool globalCommands() const { return m_globalCommands; }
		inline void setGlobalCommands(const bool _val) { m_globalCommands = _val; }
		inline void toggleGlobalCommands() { m_globalCommands = !m_globalCommands; }

	protected:
		Engine();

	private:
		static Engine* m_instance;

		std::vector < LoaderInterface* > m_loadersFile;
		std::vector < GuiInterface* > m_GUIWidgets;
		PluginList* m_plugins{ nullptr };
		QStringList m_fileExtensions;

		std::map <std::string, std::any> m_singletons;

		MediatorWObjectFWidget* m_mediator;

		bool m_withMainWindow{ true };
		Datasets m_datasets;
		Dataset* m_currentDataset{ nullptr };

		//Replacing both StateSoftwareSingleton & GlobalParametersSingleton
		nlohmann::json m_stateParameters, m_globalParameters;

		bool m_verbose{ true }, m_globalCommands{ false };
		std::vector<std::string> m_verboseTypes;
	};
}

#endif
