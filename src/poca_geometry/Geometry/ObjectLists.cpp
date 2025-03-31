/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectLists.cpp
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

#include <algorithm>

#include <General/MyData.hpp>
#include <General/BasicComponent.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <General/Misc.h>

#include "ObjectLists.hpp"
#include "BasicComputation.hpp"
#include "DelaunayTriangulation.hpp"
#include "../Interfaces/ObjectFeaturesFactoryInterface.hpp"

namespace poca::geometry {
	ObjectLists::ObjectLists(ObjectListInterface* _obj, const poca::core::CommandInfo& _com, const std::string& _plugin, const std::string& _name):BasicComponentList("ObjectLists")
	{
		m_components.push_back(_obj);
		m_currentComponent = 0;
		m_infos.push_back(std::make_tuple(_com, _plugin, _name));
	}

	ObjectLists::~ObjectLists(){

	}

	poca::core::BasicComponentInterface* ObjectLists::copy()
	{
		return new ObjectLists(*this);
	}

	void ObjectLists::addObjectList(ObjectListInterface* _obj, const poca::core::CommandInfo& _com, const std::string& _plugin, const std::string& _name)
	{
		m_currentComponent = m_components.size();
		m_components.push_back(_obj);
		m_infos.push_back(std::make_tuple(_com, _plugin, _name));
	}

	ObjectListInterface* ObjectLists::currentObjectList()
	{
		return static_cast<ObjectListInterface*>(m_components[m_currentComponent]);
	}

	uint32_t ObjectLists::currentObjectListIndex() const
	{
		return m_currentComponent;
	}

	ObjectListInterface* ObjectLists::getObjectList(const uint32_t _idx)
	{
		return static_cast<ObjectListInterface*>(m_components[_idx]);
	}

	void ObjectLists::eraseObjectList(const uint32_t _index) 
	{
		if (m_components.empty()) return;
		poca::core::BasicComponentList::eraseComponent(_index);
		m_infos.erase(m_infos.begin() + _index); 
	}
}

