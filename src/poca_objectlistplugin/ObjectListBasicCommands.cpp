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
#include <algorithm>
#include <glm/gtc/quaternion.hpp>
#include <math.h>

#include <QtCore/QString>
#include <QtCore/QFileInfo>
#include <QtWidgets/QMessageBox>

#include <General/Engine.hpp>
#include <Geometry/DetectionSet.hpp>
#include <Objects/MyObject.hpp>
#include <General/MyData.hpp>
#include <General/Histogram.hpp>
#include <Interfaces/PaletteInterface.hpp>
#include <General/Misc.h>
#include <Interfaces/CameraInterface.hpp>
#include <Geometry/ObjectListMesh.hpp>

#include "ObjectListBasicCommands.hpp"

ObjectListBasicCommands::ObjectListBasicCommands(poca::geometry::ObjectListInterface* _objs) :poca::core::Command("ObjectListBasicCommands")
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
		if (_infos->hasParameter("filename"))
			filename = _infos->getParameter<std::string>("filename");
		if (_infos->hasParameter("separator"))
			separator = _infos->getParameter<std::string>("separator");
		saveStatsObj(filename, separator);
	}
	else if (_infos->nameCommand == "saveLocsObjs") {
		std::string filename, separator(",");
		if (_infos->hasParameter("filename"))
			filename = _infos->getParameter<std::string>("filename");
		if (_infos->hasParameter("separator"))
			separator = _infos->getParameter<std::string>("separator");
		saveLocsObj(filename, separator);
	}
	else if (_infos->nameCommand == "saveOutlineLocsObjs") {
		std::string filename, separator(",");
		if (!_infos->hasParameter("filename")) return;
		filename = _infos->getParameter<std::string>("filename");
		if (_infos->hasParameter("separator"))
			separator = _infos->getParameter<std::string>("separator");
		saveOutlineLocsObj(filename, separator);
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
	else if (_infos->nameCommand == "saveSelectedObjectsForVectorHeat") {
		std::set <int> selectedObjects = _infos->hasParameter("selection") ? _infos->getParameter<std::set <int>>("selection") : std::set<int>();
		if (selectedObjects.empty()) {
			_infos->errorMessage("the selection of objects is empty.");
			return;
		}
		saveSelectedObjectsForVectorHeat(selectedObjects);
	}
	else if (_infos->nameCommand == "saveAsSVG") {
		QString filename = (_infos->getParameter<std::string>("filename")).c_str();
		saveAsSVG(filename);
	}
	else if (_infos->nameCommand == "saveAsOBJ") {
		QString filename = (_infos->getParameter<std::string>("filename")).c_str();
		saveAsOBJ(filename);
	}
	else if (_infos->nameCommand == "computeSkeletons") {
		poca::geometry::ObjectListMesh* omesh = dynamic_cast <poca::geometry::ObjectListMesh*>(m_objects);
		if (omesh == NULL) return;
		omesh->computeSkeletons();
	}
}

poca::core::CommandInfo ObjectListBasicCommands::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "saveStatsObjs" || _nameCommand == "saveLocsObjs" || _nameCommand == "saveOutlineLocsObjs") {
		poca::core::CommandInfo ci(_nameCommand);
		if (_parameters.contains("filename"))
			ci.addParameter("filename", _parameters["filename"].get<std::string>());
		
		std::string separator = _parameters.contains("separator") ? _parameters["separator"].get<std::string>() : ",";
		ci.addParameter("separator", separator);
			
		return ci;
	}
	else if (_nameCommand == "duplicateCentroids" || _nameCommand == "computeSkeletons") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	else if (_nameCommand == "duplicateSelectedObjects") {
		if (_parameters.contains("selection")) {
			std::set<int> selectedObjects = _parameters["selection"].get<std::set<int>>();
			return poca::core::CommandInfo(false, _nameCommand, "selection", selectedObjects);
		}
	}
	else if (_nameCommand == "saveSelectedObjectsVectorHeat") {
		if (_parameters.contains("selection")) {
			std::set<int> selectedObjects = _parameters["selection"].get<std::set<int>>();
			return poca::core::CommandInfo(false, _nameCommand, "selection", selectedObjects);
		}
	}
	else if (_nameCommand == "saveAsSVG" || _nameCommand == "saveAsOBJ") {
		if (_parameters.contains("filename")) {
			std::string val = _parameters["filename"].get<std::string>();
			return poca::core::CommandInfo(false, _nameCommand, "filename", val);
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

	QString filename(_filename.c_str());

	if (_filename.empty()) {
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_objects);
		if (obj == NULL) return;

		std::string dir = obj->getDir(), name = obj->getName();
		filename = QString(dir.c_str());
		if (!filename.endsWith('/'))
			filename.append('/');
		filename.append(name.c_str());
		int index = filename.lastIndexOf(".");
		filename.insert(index, "_statsObjs");
	}

	QFileInfo info(filename);
	std::ofstream fs(info.absoluteFilePath().toStdString());

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
			fs << m_objects->getMyData(name)->getData<float>()[n] << _separator;
		fs << std::endl;
	}
	fs.close();
	std::cout << "File " << filename.toStdString() << " was written" << std::endl;
}

void ObjectListBasicCommands::saveLocsObj(const std::string& _filename, const std::string& _separator) const
{
	poca::core::MyObjectInterface* obj = poca::core::Engine::instance()->getObject(m_objects);
	poca::core::MyObjectInterface* oneColorObj = obj->currentObject();
	poca::core::BasicComponentInterface* bci = oneColorObj->getBasicComponent("DetectionSet");
	if (bci == NULL) {
		std::cout << "PoCA did not succeed in save the localizations <-> objects link" << std::endl;
		return;
	}

	QString filename(_filename.c_str());

	if (_filename.empty()) {
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_objects);
		if (obj == NULL) return;

		std::string dir = obj->getDir(), name = obj->getName();
		filename = QString(dir.c_str());
		if (!filename.endsWith('/'))
			filename.append('/');
		filename.append(name.c_str());
		int index = filename.lastIndexOf(".");
		filename.insert(index, "_statsObjs");
	}

	const std::map <std::string, poca::core::MyData*>& datacomp = bci->getData();
	poca::core::stringList columnNames = bci->getNameData();
	QFileInfo info(filename);
	std::ofstream fs(info.absoluteFilePath().toStdString());
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
			fs << std::setprecision(8) << _separator << cur.second->getData<float>()[n];
		}
		fs << std::endl;
		if (currentLine++ % nbForUpdate == 0) printf("\rSaving id loc <-> id obj link: %.2f %%", ((double)currentLine / totalNb * 100.));
	}
	printf("\rSaving id loc <-> id obj link: 100 %%\n");

	fs.close();
	std::cout << "File " << filename.toStdString() << " was written" << std::endl;
}

void ObjectListBasicCommands::saveOutlineLocsObj(const std::string& _filename, const std::string& _separator) const
{
	poca::core::MyObjectInterface* obj = poca::core::Engine::instance()->getObject(m_objects);
	poca::core::MyObjectInterface* oneColorObj = obj->currentObject();
	poca::core::BasicComponentInterface* bci = oneColorObj->getBasicComponent("DetectionSet");
	if (bci == NULL) {
		std::cout << "PoCA did not succeed in save the localizations <-> objects link" << std::endl;
		return;
	}

	const std::map <std::string, poca::core::MyData*>& datacomp = bci->getData();
	poca::core::stringList columnNames = bci->getNameData();
	QFileInfo info(_filename.c_str());
	std::ofstream fs(info.absoluteFilePath().toStdString());
	if (!fs.is_open()) {
		std::cout << "Failed to open file " << _filename << std::endl;
		return;
	}

	size_t nbLocs = bci->nbElements();
	std::vector <uint32_t> idx(nbLocs, 0);

	const poca::core::MyArrayUInt32& objs = m_objects->getLocOutlines();
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
		if (idx[n] == 0) continue;
		fs << idx[n];
		for (auto cur : datacomp) {
			fs << std::setprecision(8) << _separator << cur.second->getData<float>()[n];
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
		features[feature.first] = feature.second->getData<float>();

	poca::geometry::DetectionSet* dset = new poca::geometry::DetectionSet(features);

	 poca::core::Engine* engine = poca::core::Engine::instance();
	poca::core::MyObjectInterface* obj = engine->getObject(m_objects);
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
	poca::core::MyObjectInterface* obj = poca::core::Engine::instance()->getObject(m_objects);
	poca::core::MyObjectInterface* oneColorObj = obj->currentObject();
	poca::core::BasicComponentInterface* bc = oneColorObj->getBasicComponent("DetectionSet");
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
		const std::vector <float>& values = it->second->getData<float>();
		for (size_t n = 0; n < values.size(); n++)
			if (selectedLocs[n])
				feature[cpt++] = values[n];
		features[it->first] = feature;
	}

	poca::core::BasicComponentInterface* voro = oneColorObj->getBasicComponent("VoronoiDiagram");
	const std::vector <float>& densities = voro != NULL && voro->hasData("density") ? voro->getMyData("density")->getData<float>() : std::vector <float>();
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

void ObjectListBasicCommands::saveSelectedObjectsForVectorHeat(const std::set<int>& _selectedObjects) const
{
	if (_selectedObjects.empty())
		return;

	poca::core::MyObjectInterface* obj = poca::core::Engine::instance()->getObject(m_objects);
	const std::string& dir = obj->getDir(), name = obj->getName();
	QString newName(name.c_str());
	int index = newName.lastIndexOf(".");
	newName.insert(index, "_selectedObjectVectorHeat");

	const poca::core::MyArrayUInt32& objs = m_objects->getLocOutlines();
	const std::vector <uint32_t>& data = objs.getData();
	const std::vector <uint32_t>& firsts = objs.getFirstElements();
	const std::vector <poca::core::Vec3mf>& normals = m_objects->getNormalOutlineLocs();

	std::vector <poca::core::Vec3mf> outlineLocs;
	m_objects->generateOutlineLocs(outlineLocs);

	QString filename = dir.c_str() + QString("/") + newName;
	std::ofstream fs(filename.toStdString());

	auto idObj = *_selectedObjects.begin();
	for (uint32_t cur = firsts[idObj]; cur < firsts[idObj + 1]; cur++) {
		fs << "v " << outlineLocs[cur].x() << " " << outlineLocs[cur].y() << " " << outlineLocs[cur].z() << std::endl;
		fs << "vn " << normals[cur].x() << " " << normals[cur].y() << " " << normals[cur].z() << std::endl;
	}
	fs.close();
}

void ObjectListBasicCommands::saveAsSVG(const QString& _filename) const
{
	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::opengl::CameraInterface* cam = engine->getCamera(m_objects);
	poca::core::Vec3mf direction = poca::core::Vec3mf(cam->getEye().x, cam->getEye().y, cam->getEye().z);

	poca::core::BoundingBox bbox = m_objects->boundingBox();
	std::ofstream fs(_filename.toLatin1().data());
	fs << std::setprecision(5) << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
	fs << "<svg xmlns=\"http://www.w3.org/2000/svg\"\n";
	fs << "     xmlns:xlink=\"http://www.w3.org/1999/xlink\"\n     width=\"" << (bbox[3] - bbox[0]) << "\" height=\"" << (bbox[4] - bbox[1]) << "\" viewBox=\"" << bbox[0] << " " << bbox[1] << " " << bbox[3] << " " << bbox[4] << " " "\">\n";
	fs << "<title>d:/gl2ps/type_svg_outSimple.svg</title>\n";
	fs << "<desc>\n";
	fs << "Creator: Florian Levet\n";
	fs << "</desc>\n";
	fs << "<defs>\n";
	fs << "</defs>\n";

	std::vector <poca::core::Vec3mf> triangles;
	m_objects->generateTriangles(triangles);
	std::vector <poca::core::Vec3mf> normals;
	m_objects->generateNormals(normals);

	poca::core::Histogram<float>* histogram = dynamic_cast <poca::core::Histogram<float>*>(m_objects->getCurrentHistogram());
	const std::vector<float>& values = histogram->getValues();
	const std::vector<bool>& selection = m_objects->getSelection();
	float minH = histogram->getMin(), maxH = histogram->getMax(), interH = maxH - minH;

	std::vector <float> featureValues;
	m_objects->getFeatureInSelection(featureValues, values, selection, std::numeric_limits <float>::max());

	bool fill = false;
	if (m_objects->hasParameter("fill"))
		fill = m_objects->getParameter<bool>("fill");

	/*glm::vec3 orientation = cam->getRotationSum() * glm::vec3(0.f, 0.f, 1.f);
	glm::vec3 posTmp(orientation + cam->getCenter());
	posTmp *= 2 * cam->getOriginalDistanceOrtho(); 
	poca::core::Vec3mf lightPos(posTmp.x, posTmp.y, posTmp.z), lightColor(1.0f, 1.0f, 1.0f);
	poca::core::Vec3mf viewPos(lightPos);*/

	poca::core::Vec3mf lightPos = direction * 2, lightColor(1.0f, 1.0f, 1.0f);
	poca::core::Vec3mf viewPos(lightPos);

	char col[32], black[32];
	unsigned char rb = 0, gb = 0, bb = 0;
	poca::core::getColorStringUC(rb, gb, bb, black);
	poca::core::PaletteInterface* pal = m_objects->getPalette();

	for (size_t n = 0; n < triangles.size(); n += 3) {
		if (featureValues[n] == std::numeric_limits <float>::max()) continue;
		float valPal = (featureValues[n] - minH) / interH;
		poca::core::Color4uc c = pal->getColor(valPal);
		unsigned char r = c[0], g = c[1], b = c[2];
		poca::core::getColorStringUC(r, g, b, col);

		poca::core::Vec3mf normal = (normals[n] + normals[n + 1] + normals[n + 2]) / 3.f;
		normal.normalize();

		if (direction.dot(normal) < 0.f) {
			/*poca::core::Vec3mf centroid = (triangles[n] + triangles[n + 1] + triangles[n + 2]) / 3.f;

			float ambientStrength = 0.1;
			poca::core::Vec3mf ambient = ambientStrength * lightColor;
			// diffuse \n"
			poca::core::Vec3mf lightDir = (centroid - lightPos).normalize();
			float diff = fabs(normal.dot(lightDir));
			poca::core::Vec3mf diffuse = diff * lightColor;
			// specular\n"
			float specularStrength = 0.5;
			poca::core::Vec3mf viewDir = (viewPos - centroid).normalize();
			poca::core::Vec3mf reflectDir = lightDir - normal * 2.0 * normal.dot(lightDir);
			float spec = pow(fabs(viewDir.dot(reflectDir)), 32);
			poca::core::Vec3mf specular = specularStrength * spec * lightColor;
			poca::core::Vec3mf result;
			poca::core::Vec3mf newColor(c[0], c[1], c[2]); 
			newColor = newColor * (ambient + diffuse + specular);
			unsigned char r = newColor[0], g = newColor[1], b = newColor[2];
			poca::core::getColorStringUC(r, g, b, col);*/

			if (fill) {
				glm::vec2 p1 = cam->worldToScreenCoordinates(glm::vec3(triangles[n].x(), triangles[n].y(), triangles[n].z()));
				glm::vec2 p2 = cam->worldToScreenCoordinates(glm::vec3(triangles[n + 1].x(), triangles[n + 1].y(), triangles[n + 1].z()));
				glm::vec2 p3 = cam->worldToScreenCoordinates(glm::vec3(triangles[n + 2].x(), triangles[n + 2].y(), triangles[n + 2].z()));

				fs << "<polygon points =\"";
				fs << p1.x << ",";
				fs << p1.y << " ";
				fs << p2.x << ",";
				fs << p2.y << " ";
				fs << p3.x << ",";
				fs << p3.y << "\" stroke=\"" << col << "\" fill=\"" << col << "\" stroke-width=\"0.1\"/>\n";
			}
			else {
				size_t idx[] = { n, n + 1, n + 2 };
				for (size_t i = 0; i < 3; i++) {
					size_t i1 = idx[i], i2 = idx[(i + 1) % 3];
					glm::vec2 p1 = cam->worldToScreenCoordinates(glm::vec3(triangles[i1].x(), triangles[i1].y(), triangles[i1].z()));
					glm::vec2 p2 = cam->worldToScreenCoordinates(glm::vec3(triangles[i2].x(), triangles[i2].y(), triangles[i2].z()));
					fs << "<line x1 =\"";
					fs << p1.x << "\" y1=\"";
					fs << p1.y << "\" x2=\"";
					fs << p2.x << "\" y2=\"";
					fs << p2.y << "\" stroke=\"" << black << "\" stroke-width=\"1\"/>\n";
				}
			}
		}
	}
	fs.close();
}

void ObjectListBasicCommands::saveAsOBJ(const QString& _filename) const
{
	poca::geometry::ObjectListMesh* omesh = dynamic_cast <poca::geometry::ObjectListMesh*>(m_objects);
	if (omesh == NULL) return;
	omesh->saveAsOBJ(_filename.toStdString());
	std::cout << "File " << _filename.toStdString() << "has been saved." << std::endl;
}
