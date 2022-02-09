/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      BasicComponentInterface.hpp
*
* Copyright: Florian Levet (2020-2021)
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
#include <string>

#include "../General/Command.hpp"
#include "../General/Vec4.hpp"
#include "../General/Vec6.hpp"
#include "../DesignPatterns/Observer.hpp"

namespace poca::core {

	class HistogramInterface;
	class PaletteInterface;

	typedef std::vector <std::string> stringList;

	class BasicComponent {
	public:
		virtual ~BasicComponent() = default;

		virtual stringList getNameData() const = 0;
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

		virtual std::vector <float>& getData(const std::string&) = 0;
		virtual const std::vector <float>& getData(const std::string&) const = 0;
		virtual float* getDataPtr(const std::string&) = 0;
		virtual const float* getDataPtr(const std::string&) const = 0;

		virtual std::vector <float>& getOriginalData(const std::string&) = 0;
		virtual const std::vector <float>& getOriginalData(const std::string&) const = 0;
		virtual float* getOriginalDataPtr(const std::string&) = 0;
		virtual const float* getOriginalDataPtr(const std::string&) const = 0;

		virtual void executeCommand(CommandInfo*) = 0;
		//virtual const bool getParameters(const std::string&, CommandParameters*) const = 0;

		virtual const std::vector <bool>& getSelection() const = 0;
		virtual std::vector <bool>& getSelection() = 0;
		virtual void setSelection(const std::vector <bool>&) = 0;

		virtual BasicComponent* copy() = 0;

		virtual void setBoundingBox(const float, const float, const float, const float, const float, const float) = 0;
		virtual const BoundingBox& boundingBox() const = 0;
		virtual void setWidth(const float) = 0;
		virtual void setHeight(const float) = 0;
		virtual void setThick(const float) = 0;

		virtual stringList getNameData(const std::string&) const = 0;
		virtual const size_t nbElements() const = 0;

		virtual const bool hasParameter(const std::string&, const std::string&) = 0;
		virtual const size_t nbCommands() const = 0;
	};
}

#endif

