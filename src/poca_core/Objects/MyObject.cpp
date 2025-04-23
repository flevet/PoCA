/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MyObject.cpp
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

#include <algorithm>
#include <QtCore/QFileInfo>
#include <QtCore/QDir>

#include "../General/Vec4.hpp"
#include "../General/Histogram.hpp"
#include "../General/MyData.hpp"
#include "../General/Misc.h"
#include "../Interfaces/ROIInterface.hpp"
#include "../General/BasicComponent.hpp"
#include "../General/Vec3.hpp"
//#include "../General/BasicComponentList.hpp"

#include "MyObject.hpp"

namespace poca::core {
	MyObject::MyObject() :poca::core::CommandableObject("Object")
	{
		m_internalId = poca::core::NbObjects++;
	}

	MyObject::MyObject(const MyObject& _o) :poca::core::CommandableObject(_o), m_dir(_o.m_dir), m_name(_o.m_name)
	{
		m_internalId = poca::core::NbObjects++;
		for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = _o.m_components.begin(); it != _o.m_components.end(); it++)
			this->addBasicComponent((*it)->copy());
	}

	MyObject::~MyObject()
	{
		for (poca::core::BasicComponentInterface* bci : m_components)
			delete bci;
		m_components.clear();
	}

	bool MyObject::hasBasicComponent(poca::core::BasicComponentInterface* _bc)
	{
		bool found = false;
		for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end() && !found; it++) {
			poca::core::BasicComponentInterface* bc = *it;
			found = bc == _bc;
		}
		return found;
	}

	void MyObject::addBasicComponent(poca::core::BasicComponentInterface* _bc)
	{
		bool found = false;
		for (unsigned int n = 0; n < m_components.size() && !found; n++) {
			if (m_components[n]->getName() == _bc->getName()) {
				found = true;
				poca::core::BasicComponent* bc = dynamic_cast<poca::core::BasicComponent*>(_bc);
				if (bc) {
					delete m_components[n];
					m_components[n] = _bc;
				}

				/* //This is not working because of deleting bcl -> remove for now and this needs to be done on creation of the BasicComponent to be added to BasicComponentList
				   //In the future it will be better to try to find a better way
				poca::core::BasicComponentList* bcl = dynamic_cast<poca::core::BasicComponentList*>(_bc);
				if (bcl) {
					((poca::core::BasicComponentList * )m_components[n])->copyComponentsPtr(bcl);
					delete bcl;
				}*/
			}
		}
		if (!found)
			m_components.insert(m_components.begin(), _bc);
	}

	poca::core::stringList MyObject::getNameBasicComponents() const
	{
		poca::core::stringList names;
		for (poca::core::BasicComponentInterface* bci : m_components)
			names.push_back(bci->getName());
		return names;
	}

	float MyObject::getX() const
	{
		double x = FLT_MAX;
		for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++) {
			const poca::core::BoundingBox& bbox = (*it)->boundingBox();
			if (bbox[0] < x) x = bbox[0];
		}
		return x;
	}

	float MyObject::getY() const
	{
		double y = FLT_MAX;
		for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++) {
			const poca::core::BoundingBox& bbox = (*it)->boundingBox();
			if (bbox[1] < y) y = bbox[1];
		}
		return y;
	}

	float MyObject::getZ() const
	{
		double z = FLT_MAX;
		for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++) {
			const poca::core::BoundingBox& bbox = (*it)->boundingBox();
			if (bbox[2] < z) z = bbox[2];
		}
		return z;
	}

	float MyObject::getWidth() const
	{
		double w = -FLT_MAX;
		for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++) {
			const poca::core::BoundingBox& bbox = (*it)->boundingBox();
			if (bbox[3] > w) w = bbox[3];
		}
		return w;
	}

	float MyObject::getHeight() const
	{
		double h = -FLT_MAX;
		for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++) {
			const poca::core::BoundingBox& bbox = (*it)->boundingBox();
			if (bbox[4] > h) h = bbox[4];
		}
		return h;
	}

	float MyObject::getThick() const
	{
		double t = -FLT_MAX;
		for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++) {
			const poca::core::BoundingBox& bbox = (*it)->boundingBox();
			if (bbox[5] > t) t = bbox[5];
		}
		return t;
	}

	void MyObject::setWidth(const float _w)
	{
		for (std::vector < poca::core::BasicComponentInterface* >::iterator it = m_components.begin(); it != m_components.end(); it++)
			(*it)->setWidth(_w);
	}

	void MyObject::setHeight(const float _h)
	{
		for (std::vector < poca::core::BasicComponentInterface* >::iterator it = m_components.begin(); it != m_components.end(); it++)
			(*it)->setHeight(_h);
	}

	void MyObject::setThick(const float _t)
	{
		for (std::vector < poca::core::BasicComponentInterface* >::iterator it = m_components.begin(); it != m_components.end(); it++)
			(*it)->setThick(_t);
	}

	void MyObject::executeCommand(poca::core::CommandInfo* _ci)
	{
		if (_ci->nameCommand == "loadROIs") {
			float cal = 1.f;
			if(_ci->hasParameter("calibrationXY"))
				cal = _ci->getParameter<float>("calibrationXY");
			std::string filename = _ci->getParameter<std::string>("filename");
			loadROIs(filename, cal);
		}
		else if (_ci->nameCommand == "saveROIs") {
			std::string filename = _ci->getParameter<std::string>("filename");
			saveROIs(filename);
		}
		else if (_ci->hasParameter("widthDataset"))
			setWidth(_ci->getParameter<double>("widthDataset"));
		else if (_ci->hasParameter("heightDataset"))
			setHeight(_ci->getParameter<double>("heightDataset"));
		poca::core::CommandableObject::executeCommand(_ci);
	}

	poca::core::CommandInfo MyObject::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
	{
		if (_nameCommand == "loadROIs" || _nameCommand == "saveROIs") {
			std::string filename;
			if (_parameters.contains("filename"))
				filename = _parameters["filename"].get<std::string>();
			QString curDir = QDir::currentPath();
			QDir::setCurrent(getDir().c_str());
			QFileInfo info(filename.c_str());
			filename = info.absoluteFilePath().toStdString();
			QDir::setCurrent(curDir);
			return poca::core::CommandInfo(false, _nameCommand, "filename", filename);
		}

		return poca::core::CommandInfo();
	}

	void MyObject::executeCommandOnSpecificComponent(const std::string& _nameComponent, poca::core::CommandInfo* _ci)
	{
		poca::core::BasicComponentInterface* bci = getBasicComponent(_nameComponent);
		if (bci)
			bci->executeCommand(_ci);
	}

	void MyObject::executeGlobalCommand(poca::core::CommandInfo* _ci)
	{
		executeCommand(_ci);
		for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++) {
			poca::core::BasicComponentInterface* bc = *it;
			bc->executeCommand(_ci);
		}
	}

	bool MyObject::hasBasicComponent(const std::string& _componentName)
	{
		return getBasicComponent(_componentName) != NULL;
	}

	poca::core::stringList MyObject::getNameData(const std::string& _componentName) const
	{
		poca::core::stringList list;
		poca::core::BasicComponentInterface* bc = getBasicComponent(_componentName);
		if (bc == NULL) return list;
		return bc->getNameData();
	}

	poca::core::BasicComponentInterface* MyObject::getBasicComponent(const size_t _idx) const
	{
		return m_components[_idx];
	}

	poca::core::BasicComponentInterface* MyObject::getBasicComponent(const std::string& _nameBC) const
	{
		for (unsigned int n = 0; n < m_components.size(); n++)
			if (m_components[n]->getName() == _nameBC)
				return m_components[n];
		return NULL;
	}

	void MyObject::removeBasicComponent(const std::string& _nameBC)
	{
		uint32_t n = 0;
		bool found = false;
		for (unsigned int n = 0; n < m_components.size() && !found; n++)
			found = m_components[n]->getName() == _nameBC;
		if (found) {
			delete m_components[n];
			m_components.erase(m_components.begin() + n);
		}
	}

	poca::core::BasicComponentInterface* MyObject::getLastAddedBasicComponent() const
	{
		if (m_components.empty()) return NULL;
		return m_components.back();
	}

	poca::core::HistogramInterface* MyObject::getHistogram(const std::string& _componentName, const std::string& _nameHist)
	{
		poca::core::BasicComponentInterface* bc = getBasicComponent(_componentName);
		if (bc == NULL) return NULL;
		poca::core::HistogramInterface* hist = bc->getHistogram(_nameHist);
		return hist;
	}

	const poca::core::BoundingBox MyObject::boundingBox() const
	{
		poca::core::BoundingBox bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
		for (unsigned int n = 0; n < m_components.size(); n++) {
			poca::core::BoundingBox bboxComp = m_components.at(n)->boundingBox();
			for (size_t i = 0; i < 3; i++)
				bbox[i] = bboxComp[i] < bbox[i] ? bboxComp[i] : bbox[i];
			for (size_t i = 3; i < 6; i++)
				bbox[i] = bboxComp[i] > bbox[i] ? bboxComp[i] : bbox[i];
		}
		return bbox;
	}

	void MyObject::clearROIs()
	{
		for (poca::core::ROIInterface* ROI : m_ROIs)
			delete ROI;
		m_ROIs.clear();
	}

	void MyObject::resetROIsSelection()
	{
		for (poca::core::ROIInterface* ROI : m_ROIs)
			ROI->setSelected(false);
	}

	void MyObject::saveCommands(const std::string& _name)
	{
		nlohmann::json parameters;
		std::ifstream ifs(_name);
		if (ifs.good())
			ifs >> parameters;
		ifs.close();
		saveCommands(parameters);
		std::string text = parameters.dump();
		std::cout << text << std::endl;
		std::ofstream ofs(_name);
		ofs << text;
		ofs.close();
	}

	void MyObject::saveCommands(nlohmann::json& _parameters)
	{
		for (poca::core::Command* com : m_commands)
			com->saveCommands(_parameters[com->name()]);
		for (poca::core::BasicComponentInterface* bci : m_components) {
			poca::core::CommandableObject* com = dynamic_cast <poca::core::CommandableObject*>(bci);
			com->saveCommands(_parameters);
		}
	}

	void MyObject::loadCommandsParameters(const nlohmann::json& _json)
	{
		//First grab all the available commands
		std::map <std::string, poca::core::Command*> mapCommands;
		for (poca::core::Command* com : m_commands)
			mapCommands[com->name()] = com;
		for (poca::core::BasicComponentInterface* bci : m_components) {
			poca::core::CommandableObject* com = dynamic_cast <poca::core::CommandableObject*>(bci);
			std::vector <poca::core::Command*> tmp = com->getCommands();
			for (poca::core::Command* com : tmp)
				mapCommands[com->name()] = com;
		}

		// the same code as range for
		for (const auto& el : _json.items()) {
			if (el.value().empty()) continue;
			std::cout << el.key() << " : " << el.value() << "\n";
		}
	}

	void MyObject::loadROIs(const std::string& _filename, const float _calibrationXY)
	{
		clearROIs();
		string tmp = _filename.substr(_filename.size() - 4, 4);
		if (tmp.compare(".rgn") == 0) {
			std::ifstream fs(_filename);
			std::string s;
			uint32_t curROI = 0;
			while (std::getline(fs, s)) {
				std::vector <std::array<float, 2>> points;
				std::vector <string> elems;
				std::string type;
				if (s[2] == '1') {
					type = "SquareROI";

					auto i1 = s.find(", 2") + 4, i2 = s.find(", 3");
					poca::core::split(s.substr(i1, i2 - i1), ' ', elems);
					points.push_back(std::array <float, 2>{ _calibrationXY * std::stof(elems[0]), _calibrationXY * std::stof(elems[1]) });

					elems.clear();
					i1 = s.find(", 6") + 4;
					i2 = s.find(", 7");
					poca::core::split(s.substr(i1, i2 - i1), ' ', elems);
					points.push_back(std::array <float, 2>{ points[0][0] + std::stof(elems[1]), points[0][1] + std::stof(elems[2]) });
				}
				else {
					type = "PolygonROI";
					auto i1 = s.find(", 6") + 4, i2 = s.find(", 7");
					poca::core::split(s.substr(i1, i2 - i1), ' ', elems);

					for (auto n = 1; n < elems.size(); n += 2)
						points.push_back(std::array <float, 2>{ _calibrationXY * std::stof(elems[n]), _calibrationXY * std::stof(elems[n + 1]) });
				}
				poca::core::ROIInterface* ROI = getROIFromType(type);
				ROI->load(points);
				ROI->setName("r" + curROI++);
				m_ROIs.push_back(ROI);
			}
		}
		if (_filename.substr(_filename.size() - 4, 4).compare(".txt") == 0) {
			std::ifstream fs(_filename);
			int nbRois;
			bool ok;

			std::string s;
			std::getline(fs, s);
			std::istringstream is(s);

			is >> nbRois;
			for (int n = 0; n < nbRois; n++) {
				std::getline(fs, s);
				std::istringstream is2(s);
				poca::core::ROIInterface* ROI = getROIFromType(s);
				ROI->load(fs);
				ROI->setName("r" + n);
				ROI->applyCalibrationXY(_calibrationXY);
				m_ROIs.push_back(ROI);
			}
			fs.close();
		}
	}

	void MyObject::saveROIs(const std::string& _filename)
	{
		std::ofstream fs(_filename);
		if (!fs) {
			std::cout << "Failed to open " << _filename << " to save ROIs" << std::endl;
			return;
		}
		fs << m_ROIs.size() << std::endl;
		for (const poca::core::ROIInterface* roi : m_ROIs)
			roi->save(fs);
		fs.close();
	}
}

