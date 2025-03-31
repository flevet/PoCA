/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectLists.hpp
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

#ifndef ObjectList_hpp__
#define ObjectList_hpp__

#include <any>
#include <tuple>

#include <General/BasicComponentList.hpp>
#include <General/MyArray.hpp>
#include <General/Vec3.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/ObjectListInterface.hpp>

namespace poca::geometry {
	class ObjectLists : public poca::core::BasicComponentList {
	public:
		ObjectLists(ObjectListInterface*, const poca::core::CommandInfo&, const std::string&, const std::string& = "");
		~ObjectLists();

		poca::core::BasicComponentInterface* copy();

		void addObjectList(ObjectListInterface*, const poca::core::CommandInfo&, const std::string&, const std::string & = "");
		ObjectListInterface* currentObjectList();
		ObjectListInterface* getObjectList(const uint32_t);
		uint32_t currentObjectListIndex() const;

		void eraseCurrentObjectList() { eraseObjectList(m_currentComponent); }
		void eraseObjectList(const uint32_t);

		inline const poca::core::CommandInfo& currentCommand() const { return std::get<0>(m_infos[m_currentComponent]); }
		inline const std::string& currentPlugin() const { return std::get<1>(m_infos[m_currentComponent]); }
		inline const std::string& currentName() const { return std::get<2>(m_infos[m_currentComponent]); }

		inline const poca::core::CommandInfo& getCommand(const uint32_t _index) const { return std::get<0>(m_infos[_index]); }
		inline const std::string& getPlugin(const uint32_t _index) const { return std::get<1>(m_infos[_index]); }
		inline const std::string& getName(const uint32_t _index) const { return std::get<2>(m_infos[_index]); }

		inline void setName(const uint32_t _index, const std::string& _name) { std::get<2>(m_infos[_index]) = _name; }
		inline void setCurrentName(const std::string& _name) { std::get<2>(m_infos[m_currentComponent]) = _name; }

	protected:
		std::vector <std::tuple<poca::core::CommandInfo, std::string, std::string>> m_infos;
	};
}

#endif

