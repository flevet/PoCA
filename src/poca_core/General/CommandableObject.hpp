/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      CommandableObject.hpp
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

#ifndef CommandableObject_h__
#define CommandableObject_h__

#include <string>
#include <vector>

#include "../Interfaces/CommandableObjectInterface.hpp"

namespace poca::core {

	class Command;
	class CommandInfo;

	class CommandableObject : public CommandableObjectInterface {
	public:
		virtual ~CommandableObject();

		virtual void addCommand(Command*);
		virtual void clearCommands();
		virtual const std::vector < Command* > getCommands() const;
		virtual void executeCommand(CommandInfo*);
		virtual CommandInfo createCommand(const std::string&, const nlohmann::json&);

		virtual const size_t nbCommands() const;

		virtual void loadParameters(CommandInfo*);

		virtual const bool hasParameter(const std::string&);
		virtual const bool hasParameter(const std::string&, const std::string&);

		virtual void reorganizeCommands(int, int);

		inline const std::string& name() const { return m_nameCommandableObject; }

		template <typename T>
		T getParameter(const std::string& _nameCommand)
		{
			for (std::vector < Command* >::const_iterator it = m_commands.begin(); it != m_commands.end(); it++) {
				Command* com = *it;
				if(com->hasParameter(_nameCommand))
					return com->getParameter<T>(_nameCommand, _nameCommand);
			}
			throw std::runtime_error(std::string("Parameter " + _nameCommand + " not found"));
		}

		template <typename T>
		T getParameter(const std::string& _nameCommand, const std::string& _nameParameter)
		{
			for (std::vector < Command* >::const_iterator it = m_commands.begin(); it != m_commands.end(); it++) {
				Command* com = *it;
				if (com->hasParameter(_nameCommand, _nameParameter))
					return com->getParameter<T>(_nameCommand, _nameParameter);
			}
			throw std::runtime_error(std::string("Parameter " + _nameParameter + " for command " + _nameCommand + " not found"));
		}

		template <typename T>
		T* getParameterPtr(const std::string& _nameCommand)
		{
			for (std::vector < Command* >::const_iterator it = m_commands.begin(); it != m_commands.end(); it++) {
				Command* com = *it;
				if (com->hasParameter(_nameCommand))
					return com->getParameterPtr<T>(_nameCommand, _nameCommand);
			}
			throw std::runtime_error(std::string("Parameter " + _nameCommand + " not found"));
		}

		template <typename T>
		T* getParameterPtr(const std::string& _nameCommand, const std::string& _nameParameter)
		{
			for (std::vector < Command* >::const_iterator it = m_commands.begin(); it != m_commands.end(); it++) {
				Command* com = *it;
				if (com->hasParameter(_nameCommand, _nameParameter))
					return com->getParameterPtr<T>(_nameCommand, _nameParameter);
			}
			throw std::runtime_error(std::string("Parameter " + _nameParameter + " for command " + _nameCommand + " not found"));
		}


		virtual void executeCommand(const bool _record, const std::string& _name);

		template<typename T>
		void executeCommand(const bool _record, const std::string& _name, const T& _param) {
			CommandInfo ci(_record, _name, _param);
			executeCommand(&ci);
		}

		template<typename T>
		void executeCommand(const bool _record, const std::string& _name, T* _param) {
			CommandInfo ci(_record, _name, _param);
			executeCommand(&ci);
		}

		template<typename T, typename... Args>
		void executeCommand(const bool _record, const std::string& _nameCommand, const std::string& _nameParameter, const T& _param, Args... more) {
			CommandInfo ci(_record, _nameCommand);
			ci.addParameters(_nameParameter, _param, more...);
			executeCommand(&ci);
		}

		template<typename T, typename... Args>
		void executeCommand(const bool _record, const std::string& _nameCommand, const std::string& _nameParameter, T* _param, Args... more) {
			CommandInfo ci(_record, _nameCommand);
			ci.addParameters(_nameParameter, _param, more...);
			executeCommand(&ci);
		}

		template <class T>
		T* getCommand() {
			for (Command* com : m_commands) {
				T* castedCom = dynamic_cast <T*>(com);
				if (castedCom)
					return castedCom;
			}
			return NULL;
		}

		virtual void saveCommands(nlohmann::json&);

	protected:
		CommandableObject(const std::string&);
		CommandableObject(const CommandableObject&);

	protected:
		std::vector < Command* > m_commands;
		std::string m_nameCommandableObject;
	};
}

#endif // CommandableObject_h__

