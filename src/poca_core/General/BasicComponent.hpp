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

#include "../DesignPatterns/Subject.hpp"
#include "../General/CommandableObject.hpp"
#include "../General/Vec3.hpp"
#include "../General/Vec4.hpp"
#include "../General/Vec6.hpp"

namespace poca::core {

	class MyData;
	class Palette;
	class HistogramInterface;
	class PaletteInterface;

	typedef std::vector <std::string> stringList;

	class BasicComponent : public CommandableObject {
	public:
		virtual ~BasicComponent();

		void forceRegenerateSelection();

		inline void setName(const std::string& _name) { m_nameComponent = _name; }
		inline const std::string& getName() const { return m_nameComponent; }

		virtual void setData(const std::map <std::string, std::vector <float>>&);
		virtual void addFeature(const std::string&, MyData*);

		inline void setBoundingBox(const BoundingBox& _bbox) { m_bbox = _bbox; }
		inline void setBoundingBox(const float _x, const float _y, const float _z, const float _w, const float _h, const float _t) { m_bbox = { _x, _y, _z, _w, _h, _t }; }
		inline const BoundingBox& boundingBox() const { return m_bbox; }
		inline void setWidth(const float _w) { m_bbox.setWidth(_w); }
		inline void setHeight(const float _h) { m_bbox.setHeight(_h); }
		inline void setThick(const float _t) { m_bbox.setThick(_t); }

		MyData* getMyData(const std::string&);
		void deleteFeature(const std::string&);

		std::vector <float>& getData(const std::string&);
		const std::vector <float>& getData(const std::string&) const;
		float* getDataPtr(const std::string&);
		const float* getDataPtr(const std::string&) const;

		std::vector <float>& getOriginalData(const std::string&);
		const std::vector <float>& getOriginalData(const std::string&) const;
		float* getOriginalDataPtr(const std::string&);
		const float* getOriginalDataPtr(const std::string&) const;

		const std::map <std::string, MyData*>& getData() const;
		std::map <std::string, MyData*>& getData();

		const bool hasData(const std::string&);
		const bool hasData(const std::string&) const;

		void executeCommand(CommandInfo*);
		CommandInfo createCommand(const std::string&, const nlohmann::json&);

		virtual inline void setSelected(const bool _selected) { m_selected = _selected; }
		virtual inline const bool isSelected() const { return m_selected; }

		virtual inline void setHiLow(const bool _selected) { m_hilow = _selected; }
		virtual inline const bool isHiLow() const { return m_hilow; }

		virtual inline const std::vector <bool>& getSelection() const { return m_selection; }
		virtual inline std::vector <bool>& getSelection() { return m_selection; }
		virtual inline void setSelection(const std::vector <bool>& _vals) { if (m_selection.size() != _vals.size()) return; m_selection = _vals; }
		virtual const unsigned int getNbSelection() const { return m_nbSelection; }

		virtual const unsigned int memorySize() const;

		virtual BasicComponent* copy() = 0;

		virtual void setPalette(Palette*);

		HistogramInterface* getHistogram(const std::string&);
		HistogramInterface* getHistogram(const std::string&) const;
		const bool isLogHistogram(const std::string&) const;
		HistogramInterface* getOriginalHistogram(const std::string&);
		HistogramInterface* getOriginalHistogram(const std::string&) const;
		const bool isCurrentHistogram(const std::string&);
		virtual const Color4uc getColor(const float) const;
		virtual PaletteInterface* getPalette() const;

		virtual const std::string currentHistogramType() const;
		virtual void setCurrentHistogramType(const std::string);
		virtual HistogramInterface* getCurrentHistogram();

		virtual stringList getNameData() const;
		stringList getNameData(const std::string&) const;

		const bool hasParameter(const std::string& _nameCommand) { return poca::core::CommandableObject::hasParameter(_nameCommand); }
		const bool hasParameter(const std::string& _nameCommand, const std::string& _nameParameter) { return poca::core::CommandableObject::hasParameter(_nameCommand, _nameParameter); }

		template <typename T>
		T getParameter(const std::string& _nameCommand) { return poca::core::CommandableObject::getParameter<T>(_nameCommand); }

		template <typename T>
		T getParameter(const std::string& _nameCommand, const std::string& _nameParameter) { return poca::core::CommandableObject::getParameter<T>(_nameCommand, _nameParameter); }

		template <typename T>
		T* getParameterPtr(const std::string& _nameCommand) { return poca::core::CommandableObject::getParameterPtr<T>(_nameCommand); }

		template <typename T>
		T* getParameterPtr(const std::string& _nameCommand, const std::string& _nameParameter) { return poca::core::CommandableObject::getParameterPtr<T>(_nameCommand, _nameParameter); }

		const size_t nbCommands() const { return CommandableObject::nbCommands(); }
		
		void executeCommand(const bool _record, const std::string& _name){ CommandableObject::executeCommand(_record, _name); }
		template<typename T>
		void executeCommand(const bool _record, const std::string& _name, const T& _param) { CommandableObject::executeCommand(_record, _name, _param); }
		template<typename T>
		void executeCommand(const bool _record, const std::string& _name, T* _param) { CommandableObject::executeCommand(_record, _name, _param); }
		template<typename T, typename... Args>
		void executeCommand(const bool _record, const std::string& _nameCommand, const std::string& _nameParameter, const T& _param, Args... more) { CommandableObject::executeCommand(_record, _nameCommand, _nameParameter, _param, more...); }
		template<typename T, typename... Args>
		void executeCommand(const bool _record, const std::string& _nameCommand, const std::string& _nameParameter, T* _param, Args... more) { CommandableObject::executeCommand(_record, _nameCommand, _nameParameter, _param, more...); }

		const size_t nbElements() const { return m_selection.size(); }

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
}

#endif // BasicComponent_h__

