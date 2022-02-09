/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ListDatasetsSingleton.cpp
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

#include "ListDatasetsSingleton.hpp"
#include "../Interfaces/MyObjectInterface.hpp"
#include "../General/BasicComponent.hpp"

namespace poca::core {
	ListDatasetsSingleton* ListDatasetsSingleton::m_instance = 0;

	ListDatasetsSingleton* ListDatasetsSingleton::instance()
	{
		if (m_instance == 0)
			m_instance = new ListDatasetsSingleton;
		return m_instance;
	}

	void ListDatasetsSingleton::setListDatasetsSingleton(poca::core::ListDatasetsSingleton* _lds)
	{
		m_instance = _lds;
	}

	void ListDatasetsSingleton::deleteInstance()
	{
		if (m_instance != 0)
			delete m_instance;
		m_instance = 0;
	}

	ListDatasetsSingleton::ListDatasetsSingleton()
	{

	}

	ListDatasetsSingleton::~ListDatasetsSingleton()
	{
		m_datasets.clear();
	}

	void ListDatasetsSingleton::Register(MyObjectInterface* _obj)
	{
		m_datasets.push_back(_obj);
	}

	void ListDatasetsSingleton::Unregister(MyObjectInterface* _obj)
	{
		std::vector <MyObjectInterface*>::iterator it1 = std::find(m_datasets.begin(), m_datasets.end(), _obj);
		if (it1 != m_datasets.end())
			m_datasets.erase(it1);
	}

	MyObjectInterface* ListDatasetsSingleton::getObject(BasicComponent* _bci)
	{
		for (MyObjectInterface* obj : m_datasets) {
			if (!obj->hasBasicComponent(_bci->getName())) continue;
			BasicComponent* bci = obj->getBasicComponent(_bci->getName());
			if (bci == _bci)
				return obj;
		}
		return NULL;
	}

	MyObjectInterface* ListDatasetsSingleton::getObject(MyObjectInterface* _obj)
	{
		MyObjectInterface* obj = _obj;
		for (MyObjectInterface* obj : m_datasets) {
			if (obj->nbColors() == 1) continue;
			for (size_t n = 0; n < obj->nbColors(); n++) {
				MyObjectInterface* obj2 = obj->getObject(n);
				if (obj2 == _obj)
					return obj;
			}
		}
		return obj;
	}
}

