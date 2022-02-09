/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListBasicCommands.cpp
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

#include <fstream>
#include <iomanip>

#include <QtCore/QString>
#include <QtWidgets/QMessageBox>

#include <DesignPatterns/ListDatasetsSingleton.hpp>
#include <Geometry/DetectionSet.hpp>
#include <Objects/MyObject.hpp>
#include <General/MyData.hpp>
#include <General/Histogram.hpp>

#include "ObjectListBasicCommands.hpp"

ObjectListBasicCommands::ObjectListBasicCommands(poca::geometry::ObjectList* _objs) :poca::core::Command("ObjectListBasicCommands")
{
	m_objects = _objs;
}

ObjectListBasicCommands::ObjectListBasicCommands(const ObjectListBasicCommands& _o) : poca::core::Command(_o)
{
	m_objects = _o.m_objects;
}

ObjectListBasicCommands::~ObjectListBasicCommands()
{
}

void ObjectListBasicCommands::execute(poca::core::CommandInfo* _infos)
{
	if (_infos->nameCommand == "saveStatsObjs") {
		std::string filename, separator(",");
		if (!_infos->hasParameter("filename")) return;
		filename = _infos->getParameter<std::string>("filename");
		if (_infos->hasParameter("separator"))
			separator = _infos->getParameter<std::string>("separator");
		saveStatsObj(filename, separator);
	}
	else if (_infos->nameCommand == "saveLocsObjs") {
		std::string filename, separator(",");
		if (!_infos->hasParameter("filename")) return;
		filename = _infos->getParameter<std::string>("filename");
		if (_infos->hasParameter("separator"))
			separator = _infos->getParameter<std::string>("separator");
		saveLocsObj(filename, separator);
	}
	else if (_infos->nameCommand == "duplicateCentroids") {
		poca::core::MyObjectInterface* obj = duplicateCentroids();
		
		_infos->addParameter("object", static_cast <poca::core::MyObjectInterface*>(obj));
	}
	else if (_infos->nameCommand == "duplicateSelectedObjects") {
		std::set <int> selectedObjects = _infos->hasParameter("selection")? _infos->getParameter<std::set <int>>("selection") : std::set<int>();
		if (selectedObjects.empty()) {
			_infos->errorMessage("the selection of objects is empty.");
			return;
		}
		poca::core::MyObjectInterface* obj = duplicateSelectedObjects(selectedObjects);
		if (obj == NULL) {
			_infos->errorMessage("selected objects were not duplicated.");
			return;
		}
		_infos->addParameter("object", static_cast <poca::core::MyObjectInterface*>(obj));
	}
}

poca::core::CommandInfo ObjectListBasicCommands::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "saveStatsObjs" || _nameCommand == "saveLocsObjs") {
		if (_parameters.contains("filename")) {
			std::string filename = _parameters["filename"].get<std::string>();
			std::string separator = _parameters.contains("separator") ? _parameters["separator"].get<std::string>() : ",";
			return poca::core::CommandInfo(false, _nameCommand, "filename", filename, "separator", separator);
		}
	}
	else if (_nameCommand == "duplicateCentroids") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	else if (_nameCommand == "duplicateSelectedObjects") {
		if (_parameters.contains("selection")) {
			std::set<int> selectedObjects = _parameters["selection"].get<std::set<int>>();
			return poca::core::CommandInfo(false, _nameCommand, "selection", selectedObjects);
		}
	}
	return poca::core::CommandInfo();
}

poca::core::Command* ObjectListBasicCommands::copy()
{
	return new ObjectListBasicCommands(*this);
}

void ObjectListBasicCommands::saveStatsObj(const std::string& _filename, const std::string& _separator) const
{
	const std::map <std::string, poca::core::MyData*>& data = m_objects->getData();

	std::ofstream fs(_filename);

	if (!fs.is_open()) {
		std::cout << "Failed to open file " << _filename << std::endl;
		return;
	}

	std::vector<std::string> names;
	fs << "id" << _separator;
	for (auto& [name, values] : data) {
		fs << name << _separator;
		names.push_back(name);
	}
	fs << std::endl;
	for (size_t n = 0; n < m_objects->nbObjects(); n++) {
		fs << (n + 1) << _separator;
		for (const std::string& name : names)
			fs << m_objects->getData(name)[n] << _separator;
		fs << std::endl;
	}
	fs.close();
	std::cout << "File " << _filename << " was written" << std::endl;
}

void ObjectListBasicCommands::saveLocsObj(const std::string& _filename, const std::string& _separator) const
{
	poca::core::MyObjectInterface* obj = poca::core::ListDatasetsSingleton::instance()->getObject(m_objects);
	poca::core::MyObjectInterface* oneColorObj = obj->currentObject();
	poca::core::BasicComponent* bci = oneColorObj->getBasicComponent("DetectionSet");
	if (bci == NULL) {
		std::cout << "PoCA did not succeed in save the localizations <-> objects link" << std::endl;
		return;
	}

	const std::map <std::string, poca::core::MyData*>& datacomp = bci->getData();
	poca::core::stringList columnNames = bci->getNameData();
	std::ofstream fs(_filename);
	if (!fs.is_open()) {
		std::cout << "Failed to open file " << _filename << std::endl;
		return;
	}

	size_t nbLocs = bci->nbElements();
	std::vector <uint32_t> idx(nbLocs, 0);

	const poca::core::MyArrayUInt32& objs = m_objects->getLocsObjects();
	const std::vector <uint32_t>& data = objs.getData();
	const std::vector <uint32_t>& firsts = objs.getFirstElements();

	unsigned int currentLine = 0, totalNb = data.size(), nbForUpdate = totalNb / 100.;
	if (nbForUpdate == 0) nbForUpdate = 1;
	printf("Computing id loc <-> id obj link: %.2f %%", ((double)currentLine / totalNb * 100.));

	for (size_t n = 0; n < objs.nbElements(); n++)
		for (uint32_t cur = firsts[n]; cur < firsts[n + 1]; cur++) {
			idx[data[cur]] = n + 1;
			if (currentLine++ % nbForUpdate == 0) printf("\rComputing id loc <-> id obj link: %.2f %%", ((double)currentLine / totalNb * 100.));
		}
	printf("\rComputing id loc <-> id obj link: 100 %%\n");

	currentLine = 0; totalNb = nbLocs;
	printf("Saving id loc <-> id obj link: %.2f %%", ((double)currentLine / totalNb * 100.));

	fs << "id object";
	for (auto cur : datacomp) {
		fs << _separator << cur.first;
	}
	fs << std::endl;
	for (size_t n = 0; n < idx.size(); n++) {
		fs << idx[n];
		for (auto cur : datacomp) {
			fs << std::setprecision(8) << _separator << cur.second->getOriginalData()[n];
		}
		fs << std::endl;
		if (currentLine++ % nbForUpdate == 0) printf("\rSaving id loc <-> id obj link: %.2f %%", ((double)currentLine / totalNb * 100.));
	}
	printf("\rSaving id loc <-> id obj link: 100 %%\n");

	fs.close();
	std::cout << "File " << _filename << " was written" << std::endl;
}

poca::core::MyObjectInterface* ObjectListBasicCommands::duplicateCentroids() const
{
	std::map <std::string, std::vector <float>> features;
	std::vector <poca::core::Vec3mf> centroids(m_objects->nbElements());
	for (size_t n = 0; n < m_objects->nbElements(); n++)
		centroids[n] = m_objects->computeBarycenterElement(n);
	std::vector <float> xs(centroids.size()), ys(centroids.size()), zs(centroids.size());
	for (size_t n = 0; n < centroids.size(); n++) {
		xs[n] = centroids[n][0];
		ys[n] = centroids[n][1];
		zs[n] = centroids[n][2];
	}
	features["x"] = xs;
	features["y"] = ys;
	features["z"] = zs;

	std::map <std::string, poca::core::MyData*> featuresObjects = m_objects->getData();
	for (const auto& feature : featuresObjects)
		features[feature.first] = feature.second->getOriginalHistogram()->getValues();

	poca::geometry::DetectionSet* dset = new poca::geometry::DetectionSet(features);

	poca::core::ListDatasetsSingleton* lds = poca::core::ListDatasetsSingleton::instance();
	poca::core::MyObjectInterface* obj = lds->getObject(m_objects);
	const std::string& dir = obj->getDir(), name = obj->getName();
	QString newName(name.c_str());
	int index = newName.lastIndexOf(".");
	newName.insert(index, "_objectsCentroids");

	poca::core::MyObject* wobj = new poca::core::MyObject();
	wobj->setDir(dir.c_str());
	wobj->setName(newName.toLatin1().data());
	wobj->addBasicComponent(dset);
	wobj->setDimension(dset->dimension());

	return wobj;
}

poca::core::MyObjectInterface* ObjectListBasicCommands::duplicateSelectedObjects(const std::set<int>& _selectedObjects) const
{
	poca::core::MyObjectInterface* obj = poca::core::ListDatasetsSingleton::instance()->getObject(m_objects);
	poca::core::MyObjectInterface* oneColorObj = obj->currentObject();
	poca::core::BasicComponent* bc = oneColorObj->getBasicComponent("DetectionSet");
	if (bc == NULL) {
		std::cout << "PoCA did not succeed in save the localizations <-> objects link" << std::endl;
		return NULL;
	}

	std::map <std::string, poca::core::MyData*> featuresDset = bc->getData();
	const poca::core::MyArrayUInt32& objs = m_objects->getLocsObjects();
	const std::vector <uint32_t>& data = objs.getData();
	const std::vector <uint32_t>& firsts = objs.getFirstElements();

	std::vector <bool> selectedLocs(bc->nbElements(), false);
	for (auto idx : _selectedObjects)
		for (uint32_t cur = firsts[idx]; cur < firsts[idx + 1]; cur++)
			selectedLocs[data[cur]] = true;

	auto nbSelectedLocs = std::count(selectedLocs.begin(), selectedLocs.end(), true);

	std::map <std::string, std::vector <float>> features;
	std::vector <float> feature(nbSelectedLocs);
	for (std::map <std::string, poca::core::MyData*>::const_iterator it = featuresDset.begin(); it != featuresDset.end(); it++) {
		size_t cpt = 0;
		const std::vector <float>& values = it->second->getOriginalHistogram()->getValues();
		for (size_t n = 0; n < values.size(); n++)
			if (selectedLocs[n])
				feature[cpt++] = values[n];
		features[it->first] = feature;
	}

	poca::core::BasicComponent* voro = oneColorObj->getBasicComponent("VoronoiDiagram");
	const std::vector <float>& densities = voro != NULL && voro->hasData("density") ? voro->getOriginalData("density") : std::vector <float>();
	if (!densities.empty()) {
		for (size_t n = 0, cpt = 0; n < densities.size(); n++)
			if (selectedLocs[n])
				feature[cpt++] = densities[n];
		features["density"] = feature;
	}

	poca::geometry::DetectionSet* dset = new poca::geometry::DetectionSet(features);

	const std::string& dir = obj->getDir(), name = obj->getName();
	QString newName(name.c_str());
	int index = newName.lastIndexOf(".");
	newName.insert(index, "_selectedObjects");

	poca::core::MyObject* wobj = new poca::core::MyObject();
	wobj->setDir(dir.c_str());
	wobj->setName(newName.toLatin1().data());
	wobj->addBasicComponent(dset);
	wobj->setDimension(m_objects->dimension());

	return wobj;
}