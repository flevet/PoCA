/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Observer.hpp
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

#ifndef Observer_h__
#define Observer_h__

class QString;

#include <set>

#include "../Interfaces/SubjectInterface.hpp"

namespace poca::core {

	class MyObject;
	class CommandInfo;
	class ObserverForMediator;
	class MyObjectInterface;

	class MediatorWObjectFWidgetInterface {
	public:
		virtual ~MediatorWObjectFWidgetInterface() = default;

		virtual void actionAsked(ObserverForMediator*, CommandInfo*) = 0;
		virtual void addWidget(ObserverForMediator*) = 0;
		virtual void addObserversToSubject(SubjectInterface*, const CommandInfo&) = 0;
	};

	class Observer {
	public:
		virtual ~Observer() {}
		virtual void update(SubjectInterface*, const CommandInfo&) = 0;

		virtual void addActionToObserve(const CommandInfo& _action) {
			std::set < CommandInfo >::iterator it = m_aspects.find(_action);
			if (it == m_aspects.end())
				m_aspects.insert(_action);
		}

		virtual bool hasActionToObserve(const CommandInfo& _action) {
			std::set < CommandInfo >::iterator it = m_aspects.find(_action);
			return it != m_aspects.end();
		}

	protected:
		Observer() {}

	protected:
		std::set < CommandInfo > m_aspects;
	};

	class ObserverForMediator : public Observer {
	public:
		virtual ~ObserverForMediator() {}

		virtual void performAction(MyObjectInterface*, CommandInfo*) = 0;
		virtual void update(SubjectInterface*, const CommandInfo&) = 0;
		virtual void executeMacro(MyObjectInterface*, CommandInfo*) = 0;

	protected:
		ObserverForMediator() {}
	};
}

#endif // Observer_h__

