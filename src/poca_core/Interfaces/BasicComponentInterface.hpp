/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      BasicComponentInterface.hpp
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

#ifndef BasicComponentInterface_h__
#define BasicComponentInterface_h__

#include <vector>
#include <string>

#include "../General/Vec4.hpp"
#include "../General/Vec6.hpp"
#include "../DesignPatterns/Observer.hpp"
#include "../General/CommandableObject.hpp"

namespace poca::core {

	class HistogramInterface;
	class PaletteInterface;
	class MyData;
	class Palette;

	typedef std::vector <std::string> stringList;

	class BasicComponentInterface : public CommandableObject {
	public:
		virtual ~BasicComponentInterface() {}

		virtual stringList getNameData() const = 0;
		virtual stringList getNameData(const std::string&) const = 0;
		
		virtual HistogramInterface* getHistogram(const std::string&) = 0;
		virtual HistogramInterface* getHistogram(const std::string&) const = 0;
		virtual const bool isLogHistogram(const std::string&) const = 0;
		virtual HistogramInterface* getOriginalHistogram(const std::string&) = 0;
		virtual HistogramInterface* getOriginalHistogram(const std::string&) const = 0;
		virtual const bool isCurrentHistogram(const std::string&) = 0;
		virtual const bool hasData(const std::string&) = 0;
		virtual const bool hasData(const std::string&) const = 0;
		virtual const std::string currentHistogramType() const = 0;
		virtual void setCurrentHistogramType(const std::string) = 0;
		virtual HistogramInterface* getCurrentHistogram() = 0;
		virtual void setSelected(const bool) = 0;
		virtual const bool isSelected() const = 0;

		virtual const Color4uc getColor(const float) const = 0;
		virtual PaletteInterface* getPalette() const = 0;

		virtual void setName(const std::string&) = 0;
		virtual const std::string& getName() const = 0;
		virtual void setData(const std::map <std::string, std::vector <float>>&) = 0;

		//virtual void executeCommand(CommandInfo*) = 0;
		//virtual const bool getParameters(const std::string&, CommandParameters*) const = 0;

		virtual const std::vector <bool>& getSelection() const = 0;
		virtual std::vector <bool>& getSelection() = 0;
		virtual void setSelection(const std::vector <bool>&) = 0;

		virtual BasicComponentInterface* copy() = 0;

		virtual void setBoundingBox(const float, const float, const float, const float, const float, const float) = 0;
		virtual const BoundingBox& boundingBox() const = 0;
		virtual void setWidth(const float) = 0;
		virtual void setHeight(const float) = 0;
		virtual void setThick(const float) = 0;

		virtual void forceRegenerateSelection() = 0;
		virtual void addFeature(const std::string&, MyData*) = 0;
		virtual MyData* getMyData(const std::string&) = 0;
		virtual MyData* getCurrentMyData() = 0;
		virtual void deleteFeature(const std::string&) = 0;
		virtual const std::map <std::string, MyData*>& getData() const = 0;
		virtual std::map <std::string, MyData*>& getData() = 0;
		virtual const unsigned int memorySize() const = 0;
		virtual void setPalette(Palette*) = 0;

		virtual inline void setBoundingBox(const BoundingBox&) = 0;
		virtual inline void setHiLow(const bool) = 0;
		virtual inline const bool isHiLow() const = 0;
		virtual const unsigned int getNbSelection() const = 0;

		virtual const size_t nbElements() const = 0;
		virtual const size_t nbComponents() const = 0;
		virtual const bool hasComponent(BasicComponentInterface*) const = 0;

		virtual const uint32_t dimension() const = 0;

		//virtual const bool hasParameter(const std::string&, const std::string&) = 0;
		//virtual const size_t nbCommands() const = 0;

		//virtual const bool hasParameter(const std::string& _nameCommand) = 0;
		//virtual const bool hasParameter(const std::string& _nameCommand, const std::string& _nameParameter) = 0;
		//virtual const size_t nbCommands() const = 0;

		//For CommandableObject
		virtual void executeCommand(CommandInfo* _com) { CommandableObject::executeCommand(_com); }
		virtual CommandInfo createCommand(const std::string& _name, const nlohmann::json& _com) { return CommandableObject::createCommand(_name, _com); }

		const bool hasParameter(const std::string& _nameCommand) { return CommandableObject::hasParameter(_nameCommand); }
		const bool hasParameter(const std::string& _nameCommand, const std::string& _nameParameter) { return CommandableObject::hasParameter(_nameCommand, _nameParameter); }

		template <typename T>
		T getParameter(const std::string& _nameCommand) { return CommandableObject::getParameter<T>(_nameCommand); }

		template <typename T>
		T getParameter(const std::string& _nameCommand, const std::string& _nameParameter) { return CommandableObject::getParameter<T>(_nameCommand, _nameParameter); }

		template <typename T>
		T* getParameterPtr(const std::string& _nameCommand) { return CommandableObject::getParameterPtr<T>(_nameCommand); }

		template <typename T>
		T* getParameterPtr(const std::string& _nameCommand, const std::string& _nameParameter) { return CommandableObject::getParameterPtr<T>(_nameCommand, _nameParameter); }

		const size_t nbCommands() const { return CommandableObject::nbCommands(); }

		void executeCommand(const bool _record, const std::string& _name) { CommandableObject::executeCommand(_record, _name); }
		template<typename T>
		void executeCommand(const bool _record, const std::string& _name, const T& _param) { CommandableObject::executeCommand(_record, _name, _param); }
		template<typename T>
		void executeCommand(const bool _record, const std::string& _name, T* _param) { CommandableObject::executeCommand(_record, _name, _param); }
		template<typename T, typename... Args>
		void executeCommand(const bool _record, const std::string& _nameCommand, const std::string& _nameParameter, const T& _param, Args... more) { CommandableObject::executeCommand(_record, _nameCommand, _nameParameter, _param, more...); }
		template<typename T, typename... Args>
		void executeCommand(const bool _record, const std::string& _nameCommand, const std::string& _nameParameter, T* _param, Args... more) { CommandableObject::executeCommand(_record, _nameCommand, _nameParameter, _param, more...); }

	protected:
		BasicComponentInterface(const std::string& _name): CommandableObject(_name){}
		BasicComponentInterface(const BasicComponentInterface& _o): CommandableObject(_o){}
	};
}

#endif

