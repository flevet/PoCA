/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MyObject.hpp      MyObject.hpp
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

#ifndef MyObject_h__
#define MyObject_h__

#include <QtCore/QString>
#include <vector>

#include "../DesignPatterns/Subject.hpp"
#include "../General/CommandableObject.hpp"
#include "../Interfaces/MyObjectInterface.hpp"
#include "../Interfaces/BasicComponentInterface.hpp"
#include "../General/Command.hpp"

namespace poca::core {
	class MyObject : public poca::core::MyObjectInterface, public poca::core::Subject, public poca::core::CommandableObject {
	public:
		MyObject();
		MyObject(const MyObject&);
		virtual ~MyObject();

		float getX() const;
		float getY() const;
		float getZ() const;
		float getWidth() const;
		float getHeight() const;
		float getThick() const;

		void setWidth(const float);
		void setHeight(const float);
		void setThick(const float);

		inline void setDir(const std::string& _dir) { m_dir = _dir; }
		inline const std::string& getDir() const { return m_dir; }
		inline void setName(const std::string& _name) { m_name = _name; }
		inline const std::string& getName() const { return m_name; }

		bool hasBasicComponent(poca::core::BasicComponentInterface*);
		void addBasicComponent(poca::core::BasicComponentInterface*);
		size_t nbBasicComponents() const { return m_components.size(); }
		poca::core::BasicComponentInterface* getBasicComponent(const size_t) const;
		poca::core::BasicComponentInterface* getBasicComponent(const std::string&) const;
		poca::core::BasicComponentInterface* getLastAddedBasicComponent() const;
		virtual poca::core::stringList getNameBasicComponents() const;
		void executeCommand(poca::core::CommandInfo*);
		poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);
		void removeBasicComponent(const std::string&);

		inline const unsigned int currentInternalId() const { return m_internalId; }
		inline void setInternalId(const unsigned int _val) { m_internalId = _val; }

		inline const std::vector < poca::core::BasicComponentInterface* >& getComponents() const { return m_components; }

		virtual bool hasBasicComponent(const std::string&);
		virtual poca::core::stringList getNameData(const std::string&) const;

		poca::core::HistogramInterface* getHistogram(const std::string&, const std::string&);
		const poca::core::BoundingBox boundingBox() const;

		inline const size_t dimension() const { return m_dimension; }
		inline void setDimension(const uint32_t _dim) { m_dimension = _dim; }

		//SujectInterface
		void attach(poca::core::Observer* _o, const poca::core::CommandInfo& _a) { Subject::attach(_o, _a); }
		void detach(poca::core::Observer* _o) { Subject::detach(_o); }
		void notify(const poca::core::CommandInfo& _a) { Subject::notify(_a); }
		void notifyAll(const poca::core::CommandInfo& _a) { Subject::notifyAll(_a); }

		//CommandableObjectInterface
		void addCommand(poca::core::Command* _c) { poca::core::CommandableObject::addCommand(_c); }
		void clearCommands() { poca::core::CommandableObject::clearCommands(); }
		const std::vector < poca::core::Command* > getCommands() const { return poca::core::CommandableObject::getCommands(); }
		void loadParameters(poca::core::CommandInfo* _ci) { poca::core::CommandableObject::loadParameters(_ci); }
		const bool hasParameter(const std::string& _nameCommand, const std::string& _nameParameter) { return poca::core::CommandableObject::hasParameter(_nameCommand, _nameParameter); }

		template <typename T>
		T getParameter(const std::string& _nameCommand) { return poca::core::CommandableObject::getParameter<T>(_nameCommand); }

		template <typename T>
		T getParameter(const std::string& _nameCommand, const std::string& _nameParameter) { return poca::core::CommandableObject::getParameterPtr<T>(_nameCommand, _nameParameter); }

		template <typename T>
		T* getParameterPtr(const std::string& _nameCommand) { return poca::core::CommandableObject::getParameter<T>(_nameCommand); }

		template <typename T>
		T* getParameterPtr(const std::string& _nameCommand, const std::string& _nameParameter) { return poca::core::CommandableObject::getParameterPtr<T>(_nameCommand, _nameParameter); }

		const size_t nbColors() const { return 1; }
		MyObjectInterface* getObject(const size_t) { return this; }
		MyObjectInterface* currentObject() { return this; }
		size_t currentObjectID() const { return 0; }
		void setCurrentObject(const size_t) {}

		virtual const std::vector < poca::core::ROIInterface* >& getROIs() const { return m_ROIs; }
		virtual std::vector < poca::core::ROIInterface* >& getROIs() { return m_ROIs; }
		virtual const bool hasROIs() const { return !m_ROIs.empty(); }
		virtual void addROI(poca::core::ROIInterface* _ROI) { m_ROIs.push_back(_ROI); notify("addOneROI"); }
		virtual void clearROIs();
		virtual void resetROIsSelection();
		virtual void loadROIs(const std::string&, const float = 1.f);
		virtual void saveROIs(const std::string&);

		virtual void executeCommandOnSpecificComponent(const std::string&, poca::core::CommandInfo*);
		virtual void executeGlobalCommand(poca::core::CommandInfo*);

		virtual void saveCommands(const std::string&);
		virtual void saveCommands(nlohmann::json&);
		virtual void loadCommandsParameters(const nlohmann::json&);

	protected:
		std::string m_dir, m_name;
		uint32_t m_internalId, m_dimension;

		std::vector < poca::core::BasicComponentInterface* > m_components;
		std::vector <poca::core::ROIInterface*> m_ROIs;
	};
}

#endif // MyObject_h__

