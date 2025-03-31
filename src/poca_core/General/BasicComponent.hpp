/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      BasicComponent.hpp
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

#ifndef BasicComponent_h__
#define BasicComponent_h__

#include <vector>
#include <map>
#include <any>

#include "../Interfaces/BasicComponentInterface.hpp"
#include "../DesignPatterns/Subject.hpp"

namespace poca::core {
	class BasicComponent : public BasicComponentInterface {
	public:
		virtual ~BasicComponent();

		//For BasicComponentInterface
		virtual stringList getNameData() const;
		stringList getNameData(const std::string&) const;
		
		HistogramInterface* getHistogram(const std::string&);
		HistogramInterface* getHistogram(const std::string&) const;
		const bool isLogHistogram(const std::string&) const;
		HistogramInterface* getOriginalHistogram(const std::string&);
		HistogramInterface* getOriginalHistogram(const std::string&) const;
		const bool isCurrentHistogram(const std::string&);
		const bool hasData(const std::string&);
		const bool hasData(const std::string&) const;
		virtual const std::string currentHistogramType() const;
		virtual void setCurrentHistogramType(const std::string);
		virtual HistogramInterface* getCurrentHistogram();
		virtual inline void setSelected(const bool _selected) { m_selected = _selected; }
		virtual inline const bool isSelected() const { return m_selected; }

		virtual const Color4uc getColor(const float) const;
		virtual PaletteInterface* getPalette() const;

		inline void setName(const std::string& _name) { m_nameComponent = _name; }
		inline const std::string& getName() const { return m_nameComponent; }
		virtual void setData(const std::map <std::string, std::vector <float>>&);

		template <class T>
		std::vector <T>& getData(const std::string&);
		template <class T>
		const std::vector <T>& getData(const std::string&) const;
		template <class T>
		T* getDataPtr(const std::string&);
		template <class T>
		const T* getDataPtr(const std::string&) const;

		template <class T>
		std::vector <T>& getOriginalData(const std::string&);
		template <class T>
		const std::vector <T>& getOriginalData(const std::string&) const;
		template <class T>
		T* getOriginalDataPtr(const std::string&);
		template <class T>
		const T* getOriginalDataPtr(const std::string&) const;

		virtual inline const std::vector <bool>& getSelection() const { return m_selection; }
		virtual inline std::vector <bool>& getSelection() { return m_selection; }
		virtual inline void setSelection(const std::vector <bool>& _vals) { if (m_selection.size() != _vals.size()) return; m_selection = _vals; m_nbSelection = std::count(m_selection.begin(), m_selection.end(), true);}

		virtual BasicComponentInterface* copy() = 0;
		virtual const uint32_t dimension() const = 0;

		inline void setBoundingBox(const float _x, const float _y, const float _z, const float _w, const float _h, const float _t) { m_bbox = { _x, _y, _z, _w, _h, _t }; }
		inline const BoundingBox& boundingBox() const { return m_bbox; }
		inline void setWidth(const float _w) { m_bbox.setWidth(_w); }
		inline void setHeight(const float _h) { m_bbox.setHeight(_h); }
		inline void setThick(const float _t) { m_bbox.setThick(_t); }

		const size_t nbElements() const { return m_selection.size(); }
		const size_t nbComponents() const { return 1; }
		virtual const bool hasComponent(BasicComponentInterface* _bci) const { return this == _bci; }

		void executeCommand(CommandInfo*);
		CommandInfo createCommand(const std::string&, const nlohmann::json&);
		
		//Others
		void forceRegenerateSelection();
		virtual void addFeature(const std::string&, MyData*);
		MyData* getMyData(const std::string&);
		MyData* getCurrentMyData();
		void deleteFeature(const std::string&);
		const std::map <std::string, MyData*>& getData() const;
		std::map <std::string, MyData*>& getData();
		virtual const unsigned int memorySize() const;
		virtual void setPalette(Palette*);

		inline void setBoundingBox(const BoundingBox& _bbox) { m_bbox = _bbox; }
		virtual inline void setHiLow(const bool _selected) { m_hilow = _selected; }
		virtual inline const bool isHiLow() const { return m_hilow; }
		virtual const unsigned int getNbSelection() const { return m_nbSelection; }

		void executeCommand(const bool _record, const std::string& _name) { BasicComponentInterface::executeCommand(_record, _name); }
		template<typename T>
		void executeCommand(const bool _record, const std::string& _name, const T& _param) { BasicComponentInterface::executeCommand(_record, _name, _param); }
		template<typename T>
		void executeCommand(const bool _record, const std::string& _name, T* _param) { BasicComponentInterface::executeCommand(_record, _name, _param); }
		template<typename T, typename... Args>
		void executeCommand(const bool _record, const std::string& _nameCommand, const std::string& _nameParameter, const T& _param, Args... more) { BasicComponentInterface::executeCommand(_record, _nameCommand, _nameParameter, _param, more...); }
		template<typename T, typename... Args>
		void executeCommand(const bool _record, const std::string& _nameCommand, const std::string& _nameParameter, T* _param, Args... more) { BasicComponentInterface::executeCommand(_record, _nameCommand, _nameParameter, _param, more...); }

	protected:
		BasicComponent(const std::string&);
		BasicComponent(const BasicComponent&);

	protected:
		std::string m_nameComponent;
		BoundingBox m_bbox;

		std::map <std::string, MyData*> m_data;

		std::string m_currentHistogram;
		bool m_log, m_selected, m_hilow;
		Palette* m_palette, * m_paletteSaved;
		std::vector <bool> m_selection;
		unsigned int m_nbSelection;
	};

	template <class T>
	std::vector <T>& BasicComponent::getData(const std::string& _type)
	{
		return m_data[_type]->getData<T>();
	}

	template <class T>
	const std::vector <T>& BasicComponent::getData(const std::string& _type) const
	{
		return m_data.at(_type)->getData<T>();
	}

	template <class T>
	T* BasicComponent::getDataPtr(const std::string& _type)
	{
		return hasData(_type) ? m_data[_type]->getData<T>().data() : nullptr;
	}

	template <class T>
	const T* BasicComponent::getDataPtr(const std::string& _type) const
	{
		return hasData(_type) ? m_data.at(_type)->getData<T>().data() : nullptr;
	}

	template <class T>
	std::vector <T>& BasicComponent::getOriginalData(const std::string& _type)
	{
		if (hasData(_type))
			return m_data.at(_type)->getOriginalHistogram()->getValues();
		else
			throw std::runtime_error(std::string("data " + _type + " was not found for compoent " + m_nameComponent));
	}

	template <class T>
	const std::vector <T>& BasicComponent::getOriginalData(const std::string& _type) const
	{
		if (hasData(_type))
			return m_data.at(_type)->getOriginalHistogram()->getValues();
		else
			throw std::runtime_error(std::string("data " + _type + " was not found for compoent " + m_nameComponent));
	}

	template <class T>
	T* BasicComponent::getOriginalDataPtr(const std::string& _type)
	{
		return hasData(_type) ? m_data[_type]->getOriginalData().data() : nullptr;
	}

	template <class T>
	const T* BasicComponent::getOriginalDataPtr(const std::string& _type) const
	{
		return hasData(_type) ? m_data.at(_type)->getOriginalData().data() : nullptr;
	}
}

#endif // BasicComponent_h__

