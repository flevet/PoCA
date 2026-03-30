/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Command.hpp
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

#ifndef Command_h__
#define Command_h__

#include <any>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "json.hpp"

namespace poca::core {

	class CommandRuntimeContext
	{
	public:
		CommandRuntimeContext() = default;

		template<typename T>
		void set(const T& _value) const {
			m_data[std::type_index(typeid(T))] = _value;
		}

		template<typename T>
		bool has() const {
			return m_data.find(std::type_index(typeid(T))) != m_data.end();
		}

		template<typename T>
		T get() const {
			auto it = m_data.find(std::type_index(typeid(T)));
			if (it == m_data.end())
				throw std::runtime_error("Runtime context entry not found");
			return std::any_cast<T>(it->second);
		}

	private:
		mutable std::unordered_map<std::type_index, std::any> m_data;
	};

	class CommandExecutionResult
	{
	public:
		CommandExecutionResult() = default;

		template<typename T>
		void set(const T& _value) {
			m_data[std::type_index(typeid(T))] = _value;
		}

		template<typename T>
		bool has() const {
			return m_data.find(std::type_index(typeid(T))) != m_data.end();
		}

		template<typename T>
		T get() const {
			auto it = m_data.find(std::type_index(typeid(T)));
			if (it == m_data.end())
				throw std::runtime_error("Execution result entry not found");
			return std::any_cast<T>(it->second);
		}

	private:
		std::unordered_map<std::type_index, std::any> m_data;
	};

	class CommandInfo
	{
	public:
		CommandInfo() : recordable(false) {}

		CommandInfo(const char* _name) : recordable(false), nameCommand(_name) {}

		CommandInfo(const std::string& _name) : recordable(false), nameCommand(_name) {}

		CommandInfo(const bool _record, const std::string& _name) : recordable(_record), nameCommand(_name) {}

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

		CommandInfo(const CommandInfo& _o) :recordable(_o.recordable), nameCommand(_o.nameCommand), json(_o.json) {}

		~CommandInfo() {}

		static CommandInfo fromJson(const nlohmann::json& _json, const bool _recordable = false) {
			if (_json.empty() || !_json.is_object())
				return CommandInfo();

			if (_json.contains("name")) {
				const std::string name = _json["name"].get<std::string>();
				const bool record = _json.contains("recordable") ? _json["recordable"].get<bool>() : _recordable;
				CommandInfo ci(record, name);
				if (_json.contains("params"))
					ci.json[name] = _json["params"];
				else if (_json.contains("value"))
					ci.json[name] = _json["value"];
				else
					ci.json[name] = nlohmann::json::object();
				return ci;
			}

			if (_json.size() == 1) {
				const auto it = _json.begin();
				CommandInfo ci(_recordable, it.key());
				ci.json[it.key()] = it.value();
				return ci;
			}

			return CommandInfo();
		}

		static CommandInfo fromJson(const std::string& _name, const nlohmann::json& _params, const bool _recordable = false) {
			CommandInfo ci(_recordable, _name);
			ci.json[_name] = _params;
			return ci;
		}

		template<typename T, typename... Args>
		void addParameters(const std::string& _nameP, const T& _param, const Args& ... more) {
			addParameter(_nameP, _param);
			addParameters(more...);
		}

		template<typename T>
		void addParameter(const std::string& _nameP, const T& _param) {
			try{
				parameters()[_nameP] = _param;
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
				parameters()[_nameP] = reinterpret_cast<std::uintptr_t>(_param);
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
			const nlohmann::json& params = parameters();
			return params.is_object() && params.contains(_nameParameter);
		}

		template<typename T>
		T getParameter(const std::string& _nameParameter) const {
			if(nameCommand == _nameParameter)
				return parameters().get<T>();
			return parameters()[_nameParameter].get<T>();
		}

		template<typename T>
		T* getParameterPtr(const std::string& _nameParameter) const {
			if (nameCommand == _nameParameter)
				return (T*)parameters().get<std::uintptr_t>();
			return (T*)parameters()[_nameParameter].get<std::uintptr_t>();
		}

		inline const nlohmann::json& parameters() const {
			static const nlohmann::json empty = nlohmann::json::object();
			if (nameCommand.empty() || json.empty() || !json.contains(nameCommand))
				return empty;
			return json.at(nameCommand);
		}

		inline nlohmann::json& parameters() {
			if (nameCommand.empty())
				throw std::runtime_error("Cannot access parameters of an unnamed command");
			if (!json.contains(nameCommand))
				json[nameCommand] = nlohmann::json::object();
			return json[nameCommand];
		}

		inline void setParameters(const nlohmann::json& _params) {
			if (nameCommand.empty())
				return;
			json[nameCommand] = _params;
		}

		inline nlohmann::json toJson() const { return json; }

		inline nlohmann::json toNormalizedJson() const {
			return nlohmann::json{
				{ "name", nameCommand },
				{ "params", parameters() },
				{ "recordable", recordable }
			};
		}

		inline bool operator==(const CommandInfo& other) const { return nameCommand == other.nameCommand; }
		inline bool operator<(const CommandInfo& other) const	{ return nameCommand < other.nameCommand; }
		inline CommandInfo& operator=(const CommandInfo& other) { nameCommand = other.nameCommand; json = other.json; recordable = other.recordable; return *this; }

		inline const std::string& getNameCommand() const { return nameCommand; }
		inline const std::string toString() const { return json.dump(4); }
		inline const size_t nbParameters() const { return parameters().is_object() ? parameters().size() : (parameters().empty() ? 0 : 1); }
		inline const bool isRecordable() const { return recordable; }
		inline const bool empty() const { return nameCommand == ""; }
		inline void setRecordable(const bool _value) { recordable = _value; }
		inline void errorMessage(const std::string& _mess) const { std::cout << "ERROR! Command " << nameCommand << " was not runt with error message: " << _mess << std::endl; }
		inline std::string errorMessageToStdString(const std::string& _mess) const { return std::string("ERROR! Command " + nameCommand + " was not runt with error message: " + _mess); }

	public:
		std::string nameCommand;
		nlohmann::json json;
		bool recordable;
	};

	typedef std::map <std::string, CommandInfo> CommandInfos;

	enum class CommandParameterType {
		Any,
		Boolean,
		Integer,
		UnsignedInteger,
		Number,
		String,
		Object,
		Array
	};

	struct CommandParameterSpec {
		std::string name;
		CommandParameterType type{ CommandParameterType::Any };
		bool required{ false };
		nlohmann::json defaultValue;
	};

	class CommandSpec {
	public:
		CommandSpec() = default;
		CommandSpec(const std::string& _name, std::initializer_list<CommandParameterSpec> _params) : m_name(_name), m_params(_params) {}

		inline const std::string& name() const { return m_name; }
		inline const std::vector<CommandParameterSpec>& parameters() const { return m_params; }
		inline bool matches(const std::string& _name) const { return m_name == _name; }

		CommandInfo create(const bool _recordable, const nlohmann::json& _rawParams) const {
			nlohmann::json normalized = nlohmann::json::object();
			const nlohmann::json params = (_rawParams.is_null() ? nlohmann::json::object() : _rawParams);

			if (m_params.empty()) {
				normalized = params;
				return CommandInfo::fromJson(m_name, normalized, _recordable);
			}

			if (!params.is_object() && m_params.size() == 1) {
				if (!matchesType(params, m_params.front().type))
					return CommandInfo();
				normalized[m_params.front().name] = params;
				return CommandInfo::fromJson(m_name, normalized, _recordable);
			}

			for (const CommandParameterSpec& spec : m_params) {
				if (params.is_object() && params.contains(spec.name)) {
					const nlohmann::json& value = params[spec.name];
					if (!matchesType(value, spec.type))
						return CommandInfo();
					normalized[spec.name] = value;
				}
				else if (!spec.defaultValue.is_null()) {
					normalized[spec.name] = spec.defaultValue;
				}
				else if (spec.required) {
					return CommandInfo();
				}
			}

			return CommandInfo::fromJson(m_name, normalized, _recordable);
		}

	private:
		static bool matchesType(const nlohmann::json& _value, const CommandParameterType _type) {
			switch (_type) {
			case CommandParameterType::Any: return true;
			case CommandParameterType::Boolean: return _value.is_boolean();
			case CommandParameterType::Integer: return _value.is_number_integer();
			case CommandParameterType::UnsignedInteger: return _value.is_number_unsigned() || (_value.is_number_integer() && _value.get<long long>() >= 0);
			case CommandParameterType::Number: return _value.is_number();
			case CommandParameterType::String: return _value.is_string();
			case CommandParameterType::Object: return _value.is_object();
			case CommandParameterType::Array: return _value.is_array();
			default: return false;
			}
		}

	private:
		std::string m_name;
		std::vector<CommandParameterSpec> m_params;
	};

	class Command {
	public:
		virtual ~Command() {}

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
			m_commandInfos[_com.nameCommand] = _com;
		}

		virtual std::vector<CommandSpec> commandSpecs() const {
			return {};
		}

		CommandInfo createCommandFromSpecs(const std::string& _nameCommand, const nlohmann::json& _parameters) const {
			for (const CommandSpec& spec : commandSpecs()) {
				if (spec.matches(_nameCommand))
					return spec.create(false, _parameters);
			}
			return CommandInfo();
		}

		virtual const CommandInfos saveParameters() const = 0;
		virtual void execute(CommandInfo*) = 0;
		virtual void execute(CommandInfo* _ci, const CommandRuntimeContext& _context) {
			execute(_ci);
		}
		virtual void execute(CommandInfo* _ci, const CommandRuntimeContext& _context, CommandExecutionResult& _result) {
			execute(_ci, _context);
		}
		virtual Command* copy() = 0;
		virtual CommandInfo createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters) {
			return createCommandFromSpecs(_nameCommand, _parameters);
		}

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

