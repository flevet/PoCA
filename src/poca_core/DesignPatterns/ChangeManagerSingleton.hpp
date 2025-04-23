/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ChangeManagerSingleton.hpp
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

#ifndef ChangeManagerSingleton_h__
#define ChangeManagerSingleton_h__

#include <list>
#include <utility>
#include <set>
#include <map>

#include "../Interfaces/SubjectInterface.hpp"
#include "../DesignPatterns/Observer.hpp"

namespace poca::core {

	typedef std::set < Observer* > Observers;
	typedef std::map < SubjectInterface*, Observers* > MappingSubjectObservers;

	class ChangeManagerSingleton {
	public:
		static ChangeManagerSingleton* instance();
		static void deleteInstance();
		~ChangeManagerSingleton();

		void Register(SubjectInterface*, Observer*, const CommandInfo&);
		void Unregister(SubjectInterface*, Observer*);
		void Unregister(SubjectInterface*);
		void notify(SubjectInterface*, const CommandInfo&);
		void notifyAll(SubjectInterface*, const CommandInfo&);

		void UnregisterFromAllSubjects(Observer*);
		void UnregisterFromAllObservers(SubjectInterface*);

		void RegisterForMyObjects(Observer*, const CommandInfo&);

	protected:
		ChangeManagerSingleton();

	private:
		static ChangeManagerSingleton* m_instance;

		MappingSubjectObservers m_mapping;
	};
}

#endif // ChangeManagerSingleton_h__

