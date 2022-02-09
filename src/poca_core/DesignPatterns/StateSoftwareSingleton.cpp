/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      StateSoftwareSingleton.cpp
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

#include "StateSoftwareSingleton.hpp"

namespace poca::core {
	StateSoftwareSingleton* StateSoftwareSingleton::m_instance = 0;

	StateSoftwareSingleton* StateSoftwareSingleton::instance()
	{
		if (m_instance == 0)
			m_instance = new StateSoftwareSingleton;
		return m_instance;
	}

	void StateSoftwareSingleton::setStateSoftwareSingleton(poca::core::StateSoftwareSingleton* _lds)
	{
		m_instance = _lds;
	}

	void StateSoftwareSingleton::deleteInstance()
	{
		if (m_instance != 0)
			delete m_instance;
		m_instance = 0;
	}

	StateSoftwareSingleton::StateSoftwareSingleton()
	{

	}

	StateSoftwareSingleton::~StateSoftwareSingleton()
	{
	}
}

