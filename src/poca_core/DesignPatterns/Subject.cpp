/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Subject.cpp
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

#include "../DesignPatterns/Subject.hpp"
#include "../DesignPatterns/Observer.hpp"
#include "../DesignPatterns/ChangeManagerSingleton.hpp"

namespace poca::core {

	Subject::Subject()
	{
		m_manager = ChangeManagerSingleton::instance();
	}

	Subject::~Subject()
	{
		m_manager->UnregisterFromAllObservers(this);
	}

	void Subject::attach(Observer* _obs, const CommandInfo& _aspect)
	{
		m_manager->Register(this, _obs, _aspect);
	}

	void Subject::detach(Observer* _obs)
	{
		m_manager->Unregister(this, _obs);
	}

	void Subject::notify(const CommandInfo& _aspect)
	{
		m_manager->notify(this, _aspect);
	}

	void Subject::notifyAll(const CommandInfo& _aspect)
	{
		m_manager->notifyAll(this, _aspect);
	}

	void Subject::detachFromAll()
	{
		m_manager->Unregister(this);
	}
}

