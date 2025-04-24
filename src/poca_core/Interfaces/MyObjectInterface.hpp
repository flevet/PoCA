/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MyObjectInterface.hpp
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

#ifndef MyObjectInterface_h__
#define MyObjectInterface_h__

#include <vector>
#include <string>

#include "../General/Command.hpp"
#include "../General/Vec6.hpp"
#include "../DesignPatterns/Observer.hpp"

namespace poca::core {

	class HistogramInterface;
	class CommandableObjectInterface;
	class BasicComponentInterface;
	class ROIInterface;

	typedef std::vector <std::string> stringList;

	class MyObjectInterface {
	public:
		virtual ~MyObjectInterface() = default;

		virtual bool hasBasicComponent(const std::string&) = 0;
		virtual stringList getNameData(const std::string&) const = 0;
		virtual HistogramInterface* getHistogram(const std::string&, const std::string&) = 0;

		virtual void addBasicComponent(BasicComponentInterface*) = 0;
		virtual bool hasBasicComponent(BasicComponentInterface*) = 0;
		virtual size_t nbBasicComponents() const = 0;
		virtual BasicComponentInterface* getBasicComponent(const size_t) const = 0;
		virtual BasicComponentInterface* getBasicComponent(const std::string&) const = 0;
		virtual BasicComponentInterface* getLastAddedBasicComponent() const = 0;
		virtual stringList getNameBasicComponents() const = 0;
		virtual const std::vector < poca::core::BasicComponentInterface* >& getComponents() const = 0;
		virtual void removeBasicComponent(const std::string&) = 0;

		virtual void attach(Observer*, const CommandInfo&) = 0;
		virtual void detach(Observer*) = 0;
		virtual void notify(const CommandInfo&) = 0;
		virtual void notifyAll(const CommandInfo&) = 0;

		virtual void addCommand(Command*) = 0;
		virtual void clearCommands() = 0;
		virtual const std::vector < Command* > getCommands() const = 0;
		virtual void executeCommand(CommandInfo*) = 0;
		virtual void loadParameters(CommandInfo*) = 0;
		virtual const bool hasParameter(const std::string&, const std::string&) = 0;

		virtual float getX() const = 0;
		virtual float getY() const = 0;
		virtual float getZ() const = 0;
		virtual float getWidth() const = 0;
		virtual float getHeight() const = 0;
		virtual float getThick() const = 0;

		virtual const unsigned int currentInternalId() const = 0;
		virtual const std::string& getName() const = 0;
		virtual void setName(const std::string&) = 0;
		virtual const std::string& getDir() const = 0;
		virtual void setDir(const std::string&) = 0;

		virtual void setWidth(const float) = 0;
		virtual void setHeight(const float) = 0;
		virtual void setThick(const float) = 0;
		virtual const BoundingBox boundingBox() const = 0;

		virtual const size_t nbColors() const = 0;
		virtual MyObjectInterface* getObject(const size_t) = 0;
		virtual MyObjectInterface* currentObject() = 0;
		virtual size_t currentObjectID() const = 0;
		virtual void setCurrentObject(const size_t) = 0;

		virtual const size_t dimension() const = 0;

		virtual const std::vector < ROIInterface* >& getROIs() const = 0;
		virtual std::vector < ROIInterface* >& getROIs() = 0;
		virtual const bool hasROIs() const = 0;
		virtual void addROI(ROIInterface*) = 0;
		virtual void clearROIs() = 0;
		virtual void resetROIsSelection() = 0;
		virtual void loadROIs(const std::string&, const float = 1.f) = 0;
		virtual void saveROIs(const std::string&) = 0;

		virtual void executeCommandOnSpecificComponent(const std::string&, CommandInfo*) = 0;
		virtual void executeGlobalCommand(poca::core::CommandInfo*) = 0;

		virtual void saveCommands(const std::string&) = 0;
		virtual void saveCommands(nlohmann::json&) = 0;
		virtual void loadCommandsParameters(const nlohmann::json&) = 0;

		virtual void reorganizeComponents(int, int) = 0;
	};
}

#endif // MyObjectInterface_h__

