/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Command.hpp
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

#ifndef Command_h__
#define Command_h__

#include <any>
#include <vector>
#include <string>
#include <map>
#include <iostream>

#include "json.hpp"

namespace poca::core {

	class CommandInfo
	{
	public:
		CommandInfo() : recordable(false) {

		}

		CommandInfo(const char* _name) : recordable(false), nameCommand(_name) {
		}

		CommandInfo(const std::string& _name) : recordable(false), nameCommand(_name) {
		}

		CommandInfo(const bool _record, const std::string& _name) : recordable(_record), nameCommand(_name){
		}

		template<typename T>
		CommandInfo(const bool _record, const std::string& _name, const T& _param) : recordable(_record), nameCommand(_name) {
			json[_name] = _param;
		}

		template<typename T>
		CommandInfo(const bool _record, const std::string& _name, T* _param) : recordable(_record), nameCommand(_name) {
			json[_name] = reinterpret_cast<std::uintptr_t>(_param);
		}

		template<typename T, typename... Args>
		CommandInfo(const bool _record, const std::string& _name, const std::string& _nameP, const T& _param, Args... more) : recordable(_record), nameCommand(_name)
		{
			addParameters(_nameP, _param, more...);
		}

		template<typename T, typename... Args>
		CommandInfo(const bool _record, const std::string& _name, const std::string& _nameP, T* _param, Args... more) : recordable(_record), nameCommand(_name)
		{
			addParameters(_nameP, _param, more...);
		}

		CommandInfo(const CommandInfo& _o) :recordable(_o.recordable), nameCommand(_o.nameCommand), json(_o.json) {
		}

		~CommandInfo() {
		}

		template<typename T, typename... Args>
		void addParameters(const std::string& _nameP, const T& _param, const Args& ... more) {
			addParameter(_nameP, _param);
			addParameters(more...);
		}

		template<typename T>
		void addParameter(const std::string& _nameP, const T& _param) {
			try{
				json[nameCommand][_nameP] = _param;
			}
			catch (nlohmann::json::exception& e) {
				std::cout << e.what() << std::endl;
			}
		}

		template<typename T, typename... Args>
		void addParameters(const std::string& _nameP, T* _param, const Args& ... more) {
			addParameter(_nameP, _param);
			addParameters(more...);
		}

		template<typename T>
		void addParameter(const std::string& _nameP, T* _param) {
			try {
				json[nameCommand][_nameP] = reinterpret_cast<std::uintptr_t>(_param);
			}
			catch (nlohmann::json::exception& e) {
				std::cout << e.what() << std::endl;
			}
		}

		//Do nothing, nedded by the variadic function
		void addParameters() {}

		const bool hasParameter(const std::string& _nameParameter) const {
			if (json.empty()) return false;
			if (nameCommand == _nameParameter)
				return true;
			return json[nameCommand].contains(_nameParameter);
		}

		template<typename T>
		T getParameter(const std::string& _nameParameter) const {
			if(nameCommand == _nameParameter)
				return json[nameCommand].get<T>();
			return json[nameCommand][_nameParameter].get<T>();
		}

		template<typename T>
		T* getParameterPtr(const std::string& _nameParameter) const {
			if (nameCommand == _nameParameter)
				return (T*)json[nameCommand].get<std::uintptr_t>();
			return (T*)json[nameCommand][_nameParameter].get<std::uintptr_t>();
		}

		inline bool operator==(const CommandInfo& other) const { return nameCommand == other.nameCommand; }
		inline bool operator<(const CommandInfo& other) const	{ return nameCommand < other.nameCommand; }
		inline CommandInfo& operator=(const CommandInfo& other) { nameCommand = other.nameCommand; json = other.json; recordable = other.recordable; return *this; }

		inline const std::string& getNameCommand() const { return nameCommand; }
		inline const std::string toString() const { return json.dump(4); }
		inline const size_t nbParameters() const { return json.empty() ? 0 : json[nameCommand].size(); }
		inline const bool isRecordable() const { return recordable; }
		inline const bool empty() const { return nameCommand == ""; }
		inline void errorMessage(const std::string& _mess) const { std::cout << "ERROR! Command " << nameCommand << " was not runt with error message: " << _mess << std::endl; }
		inline std::string errorMessageToStdString(const std::string& _mess) const { return std::string("ERROR! Command " + nameCommand + " was not runt with error message: " + _mess); }

	public:
		std::string nameCommand;
		nlohmann::json json;
		bool recordable;
	};

	typedef std::map <std::string, CommandInfo> CommandInfos;

	class Command {
	public:
		~Command() {}

		virtual const std::string& name() const { return m_name; }

		virtual void loadParameters(const CommandInfo& _ci) {
			if (m_commandInfos.find(_ci.nameCommand) == m_commandInfos.end()) return;
			m_commandInfos.at(_ci.nameCommand) = _ci;
		}

		virtual const bool hasCommand(const std::string& _nameCommand) const {
			return m_commandInfos.find(_nameCommand) != m_commandInfos.end();
		}

		virtual const bool hasParameter(const std::string& _nameCommand) const {
			if (m_commandInfos.find(_nameCommand) == m_commandInfos.end()) return false;
			return m_commandInfos.at(_nameCommand).hasParameter(_nameCommand);
		}

		virtual const bool hasParameter(const std::string& _nameCommand, const std::string& _nameParameter) const {
			if (m_commandInfos.find(_nameCommand) == m_commandInfos.end()) return false;
			return m_commandInfos.at(_nameCommand).hasParameter(_nameParameter);
		}

		template <typename T>
		T getParameter(const std::string& _nameCommand) const {
			if (m_commandInfos.find(_nameCommand) == m_commandInfos.end())
				throw std::runtime_error(std::string("Parameter " + _nameCommand + " not found"));
			if (!m_commandInfos.at(_nameCommand).hasParameter(_nameCommand)) 
				throw std::runtime_error(std::string("Parameter " + _nameCommand + " not found"));
			return m_commandInfos.at(_nameCommand).getParameter<T>(_nameCommand);
		}

		template <typename T>
		T getParameter(const std::string& _nameCommand, const std::string& _nameParameter) const {
			if (m_commandInfos.find(_nameCommand) == m_commandInfos.end()) 
				throw std::runtime_error(std::string("Parameter " + _nameCommand + " for command " + _nameCommand + " not found"));
			if (!m_commandInfos.at(_nameCommand).hasParameter(_nameParameter))
				throw std::runtime_error(std::string("Parameter " + _nameCommand + " for command " + _nameCommand + " not found"));
			return m_commandInfos.at(_nameCommand).getParameter<T>(_nameParameter);
		}

		template <typename T>
		T getParameterPtr(const std::string& _nameCommand) const {
			if (m_commandInfos.find(_nameCommand) == m_commandInfos.end())
				throw std::runtime_error(std::string("Parameter " + _nameCommand + " not found"));
			if (!m_commandInfos.at(_nameCommand).hasParameter(_nameCommand))
				throw std::runtime_error(std::string("Parameter " + _nameCommand + " not found"));
			return m_commandInfos.at(_nameCommand).getParameterPtr<T>(_nameCommand);
		}

		template <typename T>
		T getParameterPtr(const std::string& _nameCommand, const std::string& _nameParameter) const {
			if (m_commandInfos.find(_nameCommand) == m_commandInfos.end())
				throw std::runtime_error(std::string("Parameter " + _nameCommand + " for command " + _nameCommand + " not found"));
			if (!m_commandInfos.at(_nameCommand).hasParameter(_nameParameter))
				throw std::runtime_error(std::string("Parameter " + _nameCommand + " for command " + _nameCommand + " not found"));
			return m_commandInfos.at(_nameCommand).getParameterPtr<T>(_nameParameter);
		}

		void addCommandInfo(const CommandInfo& _com) {
			m_commandInfos.insert(std::pair(_com.nameCommand, _com));
		}


		virtual const CommandInfos saveParameters() const = 0;
		virtual void execute(CommandInfo*) = 0;
		virtual Command* copy() = 0;
		virtual CommandInfo createCommand(const std::string&, const nlohmann::json&) = 0;

		virtual void saveCommands(nlohmann::json& _json) {
			for (std::map <std::string, CommandInfo>::const_iterator it = m_commandInfos.begin(); it != m_commandInfos.end(); it++)
				_json[it->first] = it->second.json[it->first];
		}

	protected:
		Command(const std::string& _name): m_name(_name) {}
		Command(const Command& _o) :m_commandInfos(_o.m_commandInfos), m_name(_o.m_name) {
		}

	protected:
		std::string m_name;
		CommandInfos m_commandInfos;
	};
}

#endif // Command_h__

