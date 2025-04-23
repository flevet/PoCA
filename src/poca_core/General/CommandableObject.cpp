/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      CommandableObject.cpp
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

#include "../DesignPatterns/MacroRecorderSingleton.hpp"


#include "CommandableObject.hpp"
#include "Command.hpp"

namespace poca::core {

	CommandableObject::CommandableObject(const std::string& _name): m_nameCommandableObject(_name)
	{
	}

	CommandableObject::CommandableObject(const CommandableObject& _o): m_nameCommandableObject(_o.m_nameCommandableObject)
	{
		for (std::vector < Command* >::const_iterator it = _o.m_commands.begin(); it != _o.m_commands.end(); it++) {
			Command* com = *it;
			this->addCommand(com->copy());
		}
	}

	CommandableObject::~CommandableObject()
	{
		clearCommands();
	}

	void CommandableObject::addCommand(Command* _com)
	{
		m_commands.push_back(_com);
	}

	void CommandableObject::clearCommands()
	{
		for (std::vector < Command* >::iterator it = m_commands.begin(); it != m_commands.end(); it++) {
			Command* com = *it;
			delete com;
		}
		m_commands.clear();
	}

	const std::vector < Command* > CommandableObject::getCommands() const
	{
		return m_commands;
	}

	void CommandableObject::executeCommand(CommandInfo* _ci)
	{
		if (_ci->isRecordable()) {
			MacroRecorderSingleton::instance()->addCommand(m_nameCommandableObject, _ci);
			_ci->recordable = false;
		}
		for (std::vector < Command* >::iterator it = m_commands.begin(); it != m_commands.end(); it++) {
			Command* com = *it;
			com->execute(_ci);
		}
	}

	CommandInfo CommandableObject::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
	{
		for (std::vector < Command* >::iterator it = m_commands.begin(); it != m_commands.end(); it++) {
			Command* com = *it;
			CommandInfo ci = com->createCommand(_nameCommand, _parameters);
			if (!ci.empty())
				return ci;
		}
		return CommandInfo();
	}

	const bool CommandableObject::hasParameter(const std::string& _nameCommand) {
		bool ok = false;
		for (std::vector < Command* >::const_iterator it = m_commands.begin(); it != m_commands.end() && !ok; it++) {
			Command* com = *it;
			ok = com->hasParameter(_nameCommand, _nameCommand);
		}
		return ok;
	}

	const bool CommandableObject::hasParameter(const std::string& _nameCommand, const std::string& _nameParameter) {
		bool ok = false;
		for (std::vector < Command* >::const_iterator it = m_commands.begin(); it != m_commands.end() && !ok; it++) {
			Command* com = *it;
			ok = com->hasParameter(_nameCommand, _nameParameter);
		}
		return ok;
	}
	
	void CommandableObject::loadParameters(CommandInfo* _ci)
	{
		for (std::vector < Command* >::iterator it = m_commands.begin(); it != m_commands.end(); it++) {
			Command* com = *it;
			com->loadParameters(*_ci);
		}
	}

	const size_t CommandableObject::nbCommands() const
	{
		return m_commands.size();
	}

	void CommandableObject::executeCommand(const bool _record, const std::string& _nameCommand)
	{
		CommandInfo ci(_record, _nameCommand);
		executeCommand(&ci);
	}

	void CommandableObject::saveCommands(nlohmann::json& _json)
	{
		for (Command* com : m_commands)
			com->saveCommands(_json[com->name()]);
	}
}

