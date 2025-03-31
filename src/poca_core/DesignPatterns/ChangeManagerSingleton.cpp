/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ChangeManagerSingleton.cpp
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

#include "../DesignPatterns/ChangeManagerSingleton.hpp"
#include "../DesignPatterns/Subject.hpp"
#include "../DesignPatterns/Observer.hpp"
#include "../Interfaces/MyObjectInterface.hpp"

namespace poca::core {

	ChangeManagerSingleton* ChangeManagerSingleton::m_instance = 0;

	ChangeManagerSingleton::ChangeManagerSingleton()
	{

	}

	ChangeManagerSingleton::~ChangeManagerSingleton()
	{
		for (MappingSubjectObservers::iterator it1 = m_mapping.begin(); it1 != m_mapping.end(); it1++) {
			it1->second->clear();
			delete it1->second;
		}
		m_mapping.clear();
	}

	ChangeManagerSingleton* ChangeManagerSingleton::instance()
	{
		if (m_instance == 0)
			m_instance = new ChangeManagerSingleton;
		return m_instance;
	}

	void ChangeManagerSingleton::deleteInstance()
	{
		if (m_instance != 0)
			delete m_instance;
		m_instance = 0;
	}

	void ChangeManagerSingleton::Register(SubjectInterface* _subject, Observer* _observer, const CommandInfo& _action)
	{
		MappingSubjectObservers::iterator it1 = m_mapping.find(_subject);
		if (it1 != m_mapping.end()) {
			Observers* observers = it1->second;
			Observers::iterator it2 = observers->find(_observer);
			if (it2 != observers->end()) {
				Observer* observer = *it2;
				if (!observer->hasActionToObserve(_action))
					observer->addActionToObserve(_action);
			}
			else {
				if (!_observer->hasActionToObserve(_action))
					_observer->addActionToObserve(_action);
				observers->insert(_observer);
			}
		}
		else {
			if (!_observer->hasActionToObserve(_action))
				_observer->addActionToObserve(_action);
			Observers* observers = new Observers;
			observers->insert(_observer);
			m_mapping[_subject] = observers;
		}
	}

	void ChangeManagerSingleton::Unregister(SubjectInterface* _subject, Observer* _observer)
	{
		MappingSubjectObservers::iterator it1 = m_mapping.find(_subject);
		if (it1 != m_mapping.end()) {
			Observers* observers = it1->second;
			Observers::iterator it2 = observers->find(_observer);
			if (it2 != observers->end())
				observers->erase(it2);
		}
	}

	void ChangeManagerSingleton::Unregister(SubjectInterface* _subject)
	{
		MappingSubjectObservers::iterator it1 = m_mapping.find(_subject);
		if (it1 != m_mapping.end())
			m_mapping.erase(it1);
	}

	void ChangeManagerSingleton::notify(SubjectInterface* _subject, const CommandInfo& _action)
	{
		MappingSubjectObservers::iterator it1 = m_mapping.find(_subject);
		if (it1 != m_mapping.end()) {
			Observers* observers = it1->second;
			for (Observers::reverse_iterator it2 = observers->rbegin(); it2 != observers->rend(); it2++) {
				Observer* observer = *it2;
				if (observer->hasActionToObserve(_action))
					observer->update(_subject, _action);
			}
		}
	}

	void ChangeManagerSingleton::notifyAll(SubjectInterface* _subject, const CommandInfo& _action)
	{
		for (MappingSubjectObservers::iterator it1 = m_mapping.begin(); it1 != m_mapping.end(); it1++) {
			Observers* observers = it1->second;
			for (Observers::reverse_iterator it2 = observers->rbegin(); it2 != observers->rend(); it2++) {
				Observer* observer = *it2;
				if (observer->hasActionToObserve(_action))
					observer->update(_subject, _action);
			}
		}
	}

	void ChangeManagerSingleton::UnregisterFromAllSubjects(Observer* _observer) {
		for (MappingSubjectObservers::iterator it1 = m_mapping.begin(); it1 != m_mapping.end(); it1++) {
			Unregister(it1->first, _observer);
		}
	}

	void ChangeManagerSingleton::UnregisterFromAllObservers(SubjectInterface* _subject)
	{
		MappingSubjectObservers::iterator it1 = m_mapping.find(_subject);
		if (it1 != m_mapping.end())
			m_mapping.erase(it1);
	}

	void ChangeManagerSingleton::RegisterForMyObjects(Observer* _observer, const CommandInfo& _action) {
		for (MappingSubjectObservers::iterator it1 = m_mapping.begin(); it1 != m_mapping.end(); it1++) {
			SubjectInterface* subject = it1->first;
			if (dynamic_cast <MyObjectInterface*>(subject))
				Register(subject, _observer, _action);
		}
	}
}

