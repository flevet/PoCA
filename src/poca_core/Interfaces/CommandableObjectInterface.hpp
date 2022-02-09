/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      CommandableObjectInterface.hpp
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

#ifndef CommandableObjectInterface_h__
#define CommandableObjectInterface_h__

#include <vector>
#include <string>
#include <any>

#include "../General/json.hpp"

namespace poca::core {

	class Command;
	class CommandInfo;

	class CommandableObjectInterface {
	public:
		virtual ~CommandableObjectInterface() = default;

		virtual void addCommand(Command*) = 0;
		virtual void clearCommands() = 0;
		virtual const std::vector < Command* > getCommands() const = 0;
		virtual void executeCommand(CommandInfo*) = 0;
		virtual CommandInfo createCommand(const std::string&, const nlohmann::json&) = 0;

		virtual void loadParameters(CommandInfo*) = 0;
		virtual const bool hasParameter(const std::string&) = 0;
		virtual const bool hasParameter(const std::string&, const std::string&) = 0;

		virtual void saveCommands(nlohmann::json&) = 0;
	};
}

#endif // CommandableObject_h__

