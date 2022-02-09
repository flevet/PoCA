/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MyObject.cpp
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

#include <algorithm>

#include <General/Vec4.hpp>
#include <General/Histogram.hpp>
#include <General/MyData.hpp>
#include <General/Misc.h>
#include <Geometry/DetectionSet.hpp>
#include <DesignPatterns/ListDatasetsSingleton.hpp>
#include <Interfaces/ROIInterface.hpp>

#include "../../Objects/MyObject/MyObject.hpp"

MyObject::MyObject(poca::core::Calibration * _cal) :poca::core::CommandableObject()
{
	//m_ROIs = new RoiList;
	m_internalId = poca::core::NbObjects++;
	poca::core::ObjectCalibrationsSingleton * cals = poca::core::ObjectCalibrationsSingleton::instance();
	if (_cal == NULL)
		cals->Register(this, new poca::core::Calibration(1, 1, 1, "nm"));
	else
		cals->Register(this, new poca::core::Calibration(_cal->getPixelXY(), _cal->getPixelZ(), _cal->getTime(), _cal->getDimensionUnit()));
	poca::core::ListDatasetsSingleton* lds = poca::core::ListDatasetsSingleton::instance();
	lds->Register(this);
}

MyObject::MyObject(const MyObject & _o) :poca::core::CommandableObject(_o), m_dir(_o.m_dir), m_name(_o.m_name)
{
	m_internalId = poca::core::NbObjects++;
	poca::core::ObjectCalibrationsSingleton * cals = poca::core::ObjectCalibrationsSingleton::instance();
	poca::core::Calibration * cal = cals->getCalibration((MyObject *)(&_o));
	if (cal != NULL)
		cals->Register(this, new poca::core::Calibration(160, 1, 1, "nm"));
	else
		cals->Register(this, new poca::core::Calibration(cal->getPixelXY(), cal->getPixelZ(), cal->getTime(), cal->getDimensionUnit()));
	for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = _o.m_components.begin(); it != _o.m_components.end(); it++)
		this->addBasicComponent((*it)->copy());
	poca::core::ListDatasetsSingleton* lds = poca::core::ListDatasetsSingleton::instance();
	lds->Register(this);
}

MyObject::~MyObject()
{
	poca::core::ListDatasetsSingleton* lds = poca::core::ListDatasetsSingleton::instance();
	lds->Unregister(this);
	/*for (poca::core::BasicComponentInterface* bci : m_components)
		delete bci;*/
	m_components.clear();
}

bool MyObject::hasBasicComponent(poca::core::BasicComponentInterface* _bc)
{
	bool found = false;
	for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end() && !found; it++){
		poca::core::BasicComponentInterface* bc = *it;
		found = bc == _bc;
	}
	return found;
}

void MyObject::addBasicComponent(poca::core::BasicComponentInterface* _bc)
{
	bool found = false;
	for (unsigned int n = 0; n < m_components.size() && !found; n++){
		if (m_components[n]->getName() == _bc->getName()){
			found = true;
			delete m_components[n];
			m_components[n] = _bc;
		}
	}
	if (!found)
		m_components.push_back(_bc);
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
	for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++){
		const poca::core::BoundingBox& bbox = (*it)->boundingBox();
		if (bbox[3] > w) w = bbox[3];
	}
	return w;
}

float MyObject::getHeight() const
{
	double h = -FLT_MAX;
	for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++){
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

void MyObject::executeCommand(poca::core::CommandInfo * _ci)
{
	/*if (_ci->getNameComponent() == "All") {
		poca::core::CommandableObject::executeCommand(_ci);
		for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++) {
			poca::core::BasicComponentInterface* bc = *it;
			bc->executeCommand(_ci);
		}
	}*/
	//if (_ci->getNameComponent() == "MyObject")
	{// || _ci->getNameComponent() == "All") {
		bool ok;
		std::any param = _ci->getParameter("widthDataset", ok);
		if (ok) {
			double w = std::any_cast<double>(param);
			setWidth(w);
		}
		param = _ci->getParameter("heightDataset", ok);
		if (ok) {
			double h = std::any_cast<double>(param);
			setHeight(h);
		}
		poca::core::CommandableObject::executeCommand(_ci);
	}
	/*else if(_ci->getNameComponent() != "MyObject"){
		if (hasBasicComponent(_ci->getNameComponent())) {
			poca::core::BasicComponentInterface* bc = getBasicComponent(_ci->getNameComponent());
			bc->executeCommand(_ci);
		}
	}*/
}

void MyObject::executeCommandOnSpecificComponent(const std::string& _nameComponent, poca::core::CommandInfo* _ci)
{
	poca::core::BasicComponentInterface* bci = getBasicComponent(_nameComponent);
	if (bci)
		bci->executeCommand(_ci);
}

void MyObject::executeGlobalCommand(poca::core::CommandInfo* _ci)
{
	poca::core::CommandableObject::executeCommand(_ci);
	for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++) {
		poca::core::BasicComponentInterface* bc = *it;
		bc->executeCommand(_ci);
	}
}

/*const bool MyObject::getParameters(const std::string& _name, poca::core::CommandParameters * _parameters) const
{
	bool ok = poca::core::CommandableObject::getParameters(_name, _parameters);
	if (ok)
		return ok;
	for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end() && !ok; it++){
		poca::core::BasicComponentInterface* bc = *it;
		ok = bc->getParameters(_name, _parameters);
	}
	return ok;
}*/

void MyObject::getDataCurrentHistogram(const std::string& _componentName, std::vector <float>& _data)
{
	_data.clear();
	poca::core::BasicComponentInterface* bc = getBasicComponent(_componentName);
	if (bc == NULL) return;
	poca::core::HistogramInterface* hist = bc->getCurrentHistogram();
	const std::vector<float>& data = hist->getValues();
	std::copy(data.begin(), data.end(), std::back_inserter(_data));
}

void MyObject::getBinsCurrentHistogram(const std::string& _componentName, std::vector <float>& _data)
{
	_data.clear();
	poca::core::BasicComponentInterface* bc = getBasicComponent(_componentName);
	if (bc == NULL) return;
	poca::core::HistogramInterface* hist = bc->getCurrentHistogram();
	const std::vector<float>& data = hist->getBins();
	std::copy(data.begin(), data.end(), std::back_inserter(_data));
}

void MyObject::getTsCurrentHistogram(const std::string& _componentName, std::vector <float>& _data)
{
	_data.clear();
	poca::core::BasicComponentInterface* bc = getBasicComponent(_componentName);
	if (bc == NULL) return;
	poca::core::HistogramInterface* hist = bc->getCurrentHistogram();
	const std::vector<float>& data = hist->getTs();
	std::copy(data.begin(), data.end(), std::back_inserter(_data));
}

void MyObject::getDataHistogram(const std::string& _componentName, const std::string& _nameHist, std::vector <float>& _data)
{
	_data.clear();
	poca::core::BasicComponentInterface* bc = getBasicComponent(_componentName);
	if (bc == NULL) return;
	poca::core::HistogramInterface* hist = bc->getHistogram(_nameHist);
	if (hist == NULL) return;
	const std::vector<float>& data = hist->getValues();
	std::copy(data.begin(), data.end(), std::back_inserter(_data));
}

void MyObject::getBinsHistogram(const std::string& _componentName, const std::string& _nameHist, std::vector <float>& _data)
{
	_data.clear();
	poca::core::BasicComponentInterface* bc = getBasicComponent(_componentName);
	if (bc == NULL) return;
	poca::core::HistogramInterface* hist = bc->getHistogram(_nameHist);
	if (hist == NULL) return;
	const std::vector<float>& data = hist->getBins();
	std::copy(data.begin(), data.end(), std::back_inserter(_data));
}

void MyObject::getTsHistogram(const std::string& _componentName, const std::string& _nameHist, std::vector <float>& _data)
{
	_data.clear();
	poca::core::BasicComponentInterface* bc = getBasicComponent(_componentName);
	if (bc == NULL) return;
	poca::core::HistogramInterface* hist = bc->getHistogram(_nameHist);
	if (hist == NULL) return;
	const std::vector<float>& data = hist->getTs();
	std::copy(data.begin(), data.end(), std::back_inserter(_data));
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

poca::core::BasicComponentInterface* MyObject::getBasicComponent(const std::string& _nameBC) const
{
	for (unsigned int n = 0; n < m_components.size(); n++)
		if (m_components[n]->getName() == _nameBC)
			return m_components[n];
	return NULL;
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

const size_t MyObject::dimension() const
{
	poca::core::BasicComponentInterface* bc = getBasicComponent("DetectionSet");
	if (bc == NULL) return 0;
	poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bc);
	if (dset == NULL) return 0;
	return dset->dimension();
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

nlohmann::json MyObject::saveCommands() 
{
	nlohmann::json parameters;
	saveCommands(parameters/*["Parameters"]*/);
	std::string text = parameters.dump();
	std::cout << text << std::endl;
	std::ofstream fs("poca.ini");
	fs << text;
	fs.close();
	return parameters;
}

void MyObject::saveCommands(nlohmann::json& _json)
{
	for (poca::core::Command* com : m_commands)
		com->saveCommands(_json[com->name()]);
	for (poca::core::BasicComponentInterface* bci : m_components) {
		poca::core::CommandableObject* com = dynamic_cast <poca::core::CommandableObject*>(bci);
		com->saveCommands(_json/*[bci->getName()]*/);
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

	/*for (nlohmann::json::const_iterator it = _json.begin(); it != _json.end(); ++it) {
		std::cout << *it << '\n';
	}*/
	// the same code as range for
	for (const auto& el : _json.items()) {
		if (el.value().empty()) continue;
		std::cout << el.key() << " : " << el.value() << "\n";
	}
}

