/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListsCommands.cpp
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
#include <iomanip>

#include <QtCore/QString>
#include <QtCore/QFileInfo>
#include <QtWidgets/QMessageBox>

#include <General/Engine.hpp>
#include <Geometry/DetectionSet.hpp>
#include <Objects/MyObject.hpp>
#include <General/MyData.hpp>
#include <General/Histogram.hpp>

#include "ObjectListsCommands.hpp"

ObjectListsCommands::ObjectListsCommands(poca::geometry::ObjectLists* _objs) :poca::core::Command("ObjectListsCommands")
{
	m_objects = _objs;
}

ObjectListsCommands::ObjectListsCommands(const ObjectListsCommands& _o) : poca::core::Command(_o)
{
	m_objects = _o.m_objects;
}

ObjectListsCommands::~ObjectListsCommands()
{
}

void ObjectListsCommands::execute(poca::core::CommandInfo* _infos)
{
	if (_infos->nameCommand == "display") {
		for(auto * obj : m_objects->components())
			obj->executeCommand(_infos);
	}
	else
		if(m_objects->nbComponents() != 0)
			m_objects->currentObjectList()->executeCommand(_infos);
}

poca::core::CommandInfo ObjectListsCommands::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (m_objects->nbComponents() != 0)
		return m_objects->currentObjectList()->createCommand(_nameCommand, _parameters);

	return poca::core::CommandInfo();
}

poca::core::Command* ObjectListsCommands::copy()
{
	return new ObjectListsCommands(*this);
}