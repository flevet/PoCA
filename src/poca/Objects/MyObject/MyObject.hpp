/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MyObject.hpp      MyObject.hpp
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

#ifndef MyObject_h__
#define MyObject_h__

#include <QtCore/QString>
#include <vector>

#include <DesignPatterns/Subject.hpp>
#include <General/CommandableObject.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/BasicComponentInterface.hpp>
#include <General/Calibration.hpp>
#include <General/Command.hpp>

class MyObject : public poca::core::MyObjectInterface, public poca::core::Subject, public poca::core::CommandableObject{
public:
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

	inline void setDir(const std::string& _dir){ m_dir = _dir; }
	inline const std::string& getDir() const { return m_dir; }
	inline void setName(const std::string& _name){ m_name = _name; }
	inline const std::string& getName() const { return m_name; }

	bool hasBasicComponent(poca::core::BasicComponentInterface*);
	void addBasicComponent(poca::core::BasicComponentInterface*);
	poca::core::BasicComponentInterface* getBasicComponent(const std::string &) const;
	poca::core::BasicComponentInterface* getLastAddedBasicComponent() const;
	virtual poca::core::stringList getNameBasicComponents() const;
	void executeCommand(poca::core::CommandInfo *);
	//const bool getParameters(const std::string &, poca::core::CommandParameters *) const;

	inline const unsigned int currentInternalId() const { return m_internalId; }
	inline void setInternalId(const unsigned int _val){ m_internalId = _val; }

	inline const std::vector < poca::core::BasicComponentInterface* > & getComponents() const { return m_components; }

	virtual void getDataCurrentHistogram(const std::string&, std::vector <float>&);
	virtual void getBinsCurrentHistogram(const std::string&, std::vector <float>&);
	virtual void getTsCurrentHistogram(const std::string&, std::vector <float>&);
	virtual void getDataHistogram(const std::string&, const std::string&, std::vector <float>&);
	virtual void getBinsHistogram(const std::string&, const std::string&, std::vector <float>&);
	virtual void getTsHistogram(const std::string&, const std::string&, std::vector <float>&);
	virtual bool hasBasicComponent(const std::string&);
	virtual poca::core::stringList getNameData(const std::string&) const;

	poca::core::HistogramInterface* getHistogram(const std::string&, const std::string&);
	const poca::core::BoundingBox boundingBox() const;

	const size_t dimension() const;

	//virtual CommandableObjectInterface* selfAsCommandableObject() { return this; }
	//virtual SubjectInterface* selfAsSubject() { return this; }

	//SujectInterface
	void attach(poca::core::Observer* _o, const poca::core::Action& _a) { Subject::attach(_o, _a); }
	void detach(poca::core::Observer* _o) { Subject::detach(_o); }
	void notify(const poca::core::Action& _a) { Subject::notify(_a); }
	void notifyAll(const poca::core::Action& _a) { Subject::notifyAll(_a); }

	//CommandableObjectInterface
	void addCommand(poca::core::Command* _c) { poca::core::CommandableObject::addCommand(_c); }
	void clearCommands() { poca::core::CommandableObject::clearCommands(); }
	const std::vector < poca::core::Command* > getCommands() const { return poca::core::CommandableObject::getCommands(); }
	void loadParameters(poca::core::CommandInfo* _ci) { poca::core::CommandableObject::loadParameters(_ci); }
	const bool hasParameter(const std::string& _nameCommand, const std::string& _nameParameter) { return poca::core::CommandableObject::hasParameter(_nameCommand, _nameParameter); }
	std::any getParameter(const std::string& _nameCommand, bool& _ok) { return poca::core::CommandableObject::getParameter(_nameCommand, _ok); }
	std::any getParameter(const std::string& _nameCommand, const std::string& _nameParameter, bool& _ok) { return poca::core::CommandableObject::getParameter(_nameCommand, _nameParameter, _ok); }

	const size_t nbColors() const { return 1; }
	MyObjectInterface* getObject(const size_t) { return this; }
	MyObjectInterface* currentObject(){ return this; }
	size_t currentObjectID() const { return 0; }
	void setCurrentObject(const size_t) {}

	virtual const std::vector < poca::core::ROIInterface* >& getROIs() const { return m_ROIs; }
	virtual std::vector < poca::core::ROIInterface* >& getROIs() { return m_ROIs; }
	virtual const bool hasROIs() const { return !m_ROIs.empty(); }
	virtual void addROI(poca::core::ROIInterface* _ROI) { m_ROIs.push_back(_ROI); notify("addOneROI"); }
	virtual void clearROIs();
	virtual void resetROIsSelection();

	virtual void executeCommandOnSpecificComponent(const std::string&, poca::core::CommandInfo*);
	virtual void executeGlobalCommand(poca::core::CommandInfo*);

	virtual nlohmann::json saveCommands();
	virtual void saveCommands(nlohmann::json&);
	virtual void loadCommandsParameters(const nlohmann::json&);

protected:
	MyObject(poca::core::Calibration * = NULL);
	MyObject(const MyObject &);

protected:
	std::string m_dir, m_name;
	unsigned int m_internalId;

	std::vector < poca::core::BasicComponentInterface * > m_components;
	std::vector <poca::core::ROIInterface*> m_ROIs;
};

#endif // MyObject_h__

