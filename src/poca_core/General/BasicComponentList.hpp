/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      BasicComponentList.hpp
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

#ifndef BasicComponentList_h__
#define BasicComponentList_h__

#include <vector>
#include <map>
#include <any>
#include <string>

#include "../General/BasicComponent.hpp"

namespace poca::core {
	class BasicComponentList : public BasicComponentInterface {
	public:
		virtual ~BasicComponentList() {
			if(!m_dontDeleteComponents)
				for (auto* component : m_components)
					delete component;
			m_components.clear();
		}

		//For BasicComponentInterface
		stringList getNameData() const { return m_components[m_currentComponent]->getNameData(); }
		stringList getNameData(const std::string&) const { return m_components[m_currentComponent]->getNameData(); }
		
		HistogramInterface* getHistogram(const std::string& _name) { return m_components[m_currentComponent]->getHistogram(_name); }
		HistogramInterface* getHistogram(const std::string& _name) const { return m_components[m_currentComponent]->getHistogram(_name); }
		const bool isLogHistogram(const std::string& _name) const { return m_components[m_currentComponent]->isLogHistogram(_name); }
		HistogramInterface* getOriginalHistogram(const std::string& _name) { return m_components[m_currentComponent]->getOriginalHistogram(_name); }
		HistogramInterface* getOriginalHistogram(const std::string& _name) const { return m_components[m_currentComponent]->getOriginalHistogram(_name); }
		const bool isCurrentHistogram(const std::string& _name) { return m_components[m_currentComponent]->isCurrentHistogram(_name); }
		const bool hasData(const std::string& _name) { return m_components[m_currentComponent]->hasData(_name); }
		const bool hasData(const std::string& _name) const { return m_components[m_currentComponent]->hasData(_name); }
		virtual const std::string currentHistogramType() const { return m_components[m_currentComponent]->currentHistogramType(); }
		virtual void setCurrentHistogramType(const std::string _name) { m_components[m_currentComponent]->setCurrentHistogramType(_name); }
		virtual HistogramInterface* getCurrentHistogram() { return m_components[m_currentComponent]->getCurrentHistogram(); }
		virtual inline void setSelected(const bool _selected) { m_components[m_currentComponent]->setSelected(_selected); }
		virtual inline const bool isSelected() const { return m_components[m_currentComponent]->isSelected(); }

		virtual const Color4uc getColor(const float _val) const { return m_components[m_currentComponent]->getColor(_val); }
		virtual PaletteInterface* getPalette() const { return m_components[m_currentComponent]->getPalette(); }

		inline void setName(const std::string& _name) { m_nameComponent = _name; }
		inline const std::string& getName() const { return m_nameComponent; }
		virtual void setData(const std::map <std::string, std::vector <float>>& _data) { return m_components[m_currentComponent]->setData(_data); }

		template <class T>
		std::vector <T>& getData(const std::string& _name) { return m_components[m_currentComponent]->getData(_name); }
		template <class T>
		const std::vector <T>& getData(const std::string& _name) const { return m_components[m_currentComponent]->getData(_name); }
		template <class T>
		T* getDataPtr(const std::string& _name) { return m_components[m_currentComponent]->getDataPtr(_name); }
		template <class T>
		const T* getDataPtr(const std::string& _name) const { return m_components[m_currentComponent]->getDataPtr(_name); }

		template <class T>
		std::vector <T>& getOriginalData(const std::string& _name) { return m_components[m_currentComponent]->getOriginalData(_name); }
		template <class T>
		const std::vector <T>& getOriginalData(const std::string& _name) const { return m_components[m_currentComponent]->getOriginalData(_name); }
		template <class T>
		T* getOriginalDataPtr(const std::string& _name) { return m_components[m_currentComponent]->getOriginalDataPtr(_name); }
		template <class T>
		const T* getOriginalDataPtr(const std::string& _name) const { return m_components[m_currentComponent]->getOriginalDataPtr(_name); }

		virtual inline const std::vector <bool>& getSelection() const { return m_components[m_currentComponent]->getSelection(); }
		virtual inline std::vector <bool>& getSelection() { return m_components[m_currentComponent]->getSelection(); }
		virtual inline void setSelection(const std::vector <bool>& _vals) { m_components[m_currentComponent]->setSelection(_vals); }

		virtual BasicComponentInterface* copy() = 0;

		inline void setBoundingBox(const float _x, const float _y, const float _z, const float _w, const float _h, const float _t) {  m_components[m_currentComponent]->setBoundingBox(_x, _y, _z, _w, _h, _t); }
		inline const BoundingBox& boundingBox() const { return m_components[m_currentComponent]->boundingBox(); }
		inline void setWidth(const float _w) { m_components[m_currentComponent]->setWidth(_w); }
		inline void setHeight(const float _h) { m_components[m_currentComponent]->setHeight(_h); }
		inline void setThick(const float _t) { m_components[m_currentComponent]->setThick(_t); }

		const size_t nbElements() const { return m_components[m_currentComponent]->nbElements(); }

		const uint32_t dimension() const { return m_components[m_currentComponent]->dimension(); }

		//Others
		void forceRegenerateSelection() { m_components[m_currentComponent]->forceRegenerateSelection(); }
		virtual void addFeature(const std::string& _name, MyData* _data) { m_components[m_currentComponent]->addFeature(_name, _data); }
		MyData* getMyData(const std::string& _name) { return m_components[m_currentComponent]->getMyData(_name); }
		MyData* getCurrentMyData() { return m_components[m_currentComponent]->getCurrentMyData(); }
		void deleteFeature(const std::string& _name) { return m_components[m_currentComponent]->deleteFeature(_name); }
		const std::map <std::string, MyData*>& getData() const { return m_components[m_currentComponent]->getData(); }
		std::map <std::string, MyData*>& getData() { return m_components[m_currentComponent]->getData(); }
		virtual const unsigned int memorySize() const { return m_components[m_currentComponent]->memorySize(); }
		virtual void setPalette(Palette* _name) { m_components[m_currentComponent]->setPalette(_name); }

		inline void setBoundingBox(const BoundingBox& _bbox) { m_components[m_currentComponent]->setBoundingBox(_bbox); }
		virtual inline void setHiLow(const bool _selected) { m_components[m_currentComponent]->setHiLow(_selected); }
		virtual inline const bool isHiLow() const { return m_components[m_currentComponent]->isHiLow(); }
		virtual const unsigned int getNbSelection() const { return m_components[m_currentComponent]->getNbSelection(); }

		virtual void addComponent(BasicComponent* _bci) {
			m_currentComponent = m_components.size();
			m_components.push_back(_bci);
		}

		virtual void copyComponentsPtr(BasicComponentList* _list) {
			for (auto* comp : _list->m_components)
				m_components.push_back(comp);
			m_currentComponent = m_components.size() - 1;
		}

		virtual void copyComponents(BasicComponentList* _list) {
			for (auto* comp : _list->m_components)
				m_components.push_back(static_cast<BasicComponent*>(comp->copy()));
		}

		virtual const std::vector <BasicComponent*>& components() const { return m_components; }
		virtual std::vector <BasicComponent*>& components() { return m_components; }
		virtual const bool hasComponent(BasicComponentInterface* _bci) const {
			for (auto component : m_components)
				if (component == _bci)
					return true;
			return false;
		}

		inline const uint32_t currentComponentIndex() const { return m_currentComponent; }
		inline void setCurrentComponentIndex(const uint32_t _index) { m_currentComponent = _index; }
		inline BasicComponent* currentComponent() const { return m_components[m_currentComponent]; }
		inline BasicComponent* getComponent(const uint32_t _index) const { return m_components[_index]; }
		inline const size_t nbComponents() const { return m_components.size(); }

		inline void eraseCurrentComponent() { eraseComponent(m_currentComponent); }
		void eraseComponent(const uint32_t _index) {
			if (m_components.empty()) return;
			delete m_components[_index];
			m_components.erase(m_components.begin() + _index);
			if (m_currentComponent == _index)
				if (m_currentComponent != 0)
					m_currentComponent--;
		}

		inline void dontDeleteComponents() { m_dontDeleteComponents = true; }

		void executeCommand(CommandInfo* _com) { poca::core::BasicComponentInterface::executeCommand(_com); }
		CommandInfo createCommand(const std::string& _name, const nlohmann::json& _com) { return poca::core::BasicComponentInterface::createCommand(_name, _com); }

		void executeCommand(const bool _record, const std::string& _name) { poca::core::BasicComponentInterface::executeCommand(_record, _name); }
		template<typename T>
		void executeCommand(const bool _record, const std::string& _name, const T& _param) { poca::core::BasicComponentInterface::executeCommand(_record, _name, _param); }
		template<typename T>
		void executeCommand(const bool _record, const std::string& _name, T* _param) { poca::core::BasicComponentInterface::executeCommand(_record, _name, _param); }
		template<typename T, typename... Args>
		void executeCommand(const bool _record, const std::string& _nameCommand, const std::string& _nameParameter, const T& _param, Args... more) { poca::core::BasicComponentInterface::executeCommand(_record, _nameCommand, _nameParameter, _param, more...); }
		template<typename T, typename... Args>
		void executeCommand(const bool _record, const std::string& _nameCommand, const std::string& _nameParameter, T* _param, Args... more) { poca::core::BasicComponentInterface::executeCommand(_record, _nameCommand, _nameParameter, _param, more...); }

		/*
		void executeCommand(CommandInfo* _com) { m_components[m_currentComponent]->executeCommand(_com); }
		CommandInfo createCommand(const std::string& _name, const nlohmann::json& _com) { return m_components[m_currentComponent]->createCommand(_name, _com); }

		void executeCommand(const bool _record, const std::string& _name) { m_components[m_currentComponent]->executeCommand(_record, _name); }
		template<typename T>
		void executeCommand(const bool _record, const std::string& _name, const T& _param) { m_components[m_currentComponent]->executeCommand(_record, _name, _param); }
		template<typename T>
		void executeCommand(const bool _record, const std::string& _name, T* _param) { m_components[m_currentComponent]->executeCommand(_record, _name, _param); }
		template<typename T, typename... Args>
		void executeCommand(const bool _record, const std::string& _nameCommand, const std::string& _nameParameter, const T& _param, Args... more) { m_components[m_currentComponent]->executeCommand(_record, _nameCommand, _nameParameter, _param, more...); }
		template<typename T, typename... Args>
		void executeCommand(const bool _record, const std::string& _nameCommand, const std::string& _nameParameter, T* _param, Args... more) { m_components[m_currentComponent]->executeCommand(_record, _nameCommand, _nameParameter, _param, more...); }
		*/
	protected:
		BasicComponentList(const std::string& _name) : BasicComponentInterface(_name), m_nameComponent(_name), m_dontDeleteComponents(false){}
		BasicComponentList(const BasicComponentList& _o) : BasicComponentInterface(_o), m_nameComponent(_o.m_nameComponent), m_dontDeleteComponents(false){
			for (auto* component : m_components)
				delete component;
			m_components.clear();
			for (auto* component : _o.m_components)
				m_components.push_back(static_cast<BasicComponent*>(component->copy()));
		}

	protected:
		std::string m_nameComponent;

		std::vector <BasicComponent*> m_components;
		uint32_t m_currentComponent;
		bool m_dontDeleteComponents;
	};
}

#endif // BasicComponent_h__

