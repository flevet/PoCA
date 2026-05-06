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
#include <glm/gtx/transform.hpp>

#include <OpenGL/Camera.hpp>
#include <OpenGL/RenderCommandContext.hpp>

#include "../General/Vec4.hpp"
#include "../General/Histogram.hpp"
#include "../General/MyData.hpp"
#include "../General/Misc.h"
#include "../Interfaces/ROIInterface.hpp"
#include "../General/BasicComponent.hpp"
#include "../General/Vec3.hpp"
#include "../General/BasicComponentList.hpp"

#include "MyObject.hpp"

namespace poca::core {
	namespace {
		bool isComponentFamilyHandled(const poca::core::CommandExecutionResult& _result, const std::string& _componentName)
		{
			if (!_result.has<poca::opengl::RenderedComponentFamilies>())
				return false;
			const std::vector<std::string>& names = _result.get<poca::opengl::RenderedComponentFamilies>().componentNames;
			return std::find(names.begin(), names.end(), _componentName) != names.end();
		}

		poca::core::BoundingBox transformBoundingBox(const poca::core::BoundingBox& _bbox, const glm::mat4& _model)
		{
			poca::core::BoundingBox transformed(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
			const float xs[2] = { _bbox[0], _bbox[3] };
			const float ys[2] = { _bbox[1], _bbox[4] };
			const float zs[2] = { _bbox[2], _bbox[5] };

			for (size_t ix = 0; ix < 2; ix++) {
				for (size_t iy = 0; iy < 2; iy++) {
					for (size_t iz = 0; iz < 2; iz++) {
						const glm::vec4 corner = _model * glm::vec4(xs[ix], ys[iy], zs[iz], 1.f);
						transformed[0] = std::min(transformed[0], corner.x);
						transformed[1] = std::min(transformed[1], corner.y);
						transformed[2] = std::min(transformed[2], corner.z);
						transformed[3] = std::max(transformed[3], corner.x);
						transformed[4] = std::max(transformed[4], corner.y);
						transformed[5] = std::max(transformed[5], corner.z);
					}
				}
			}

			return transformed;
		}

		glm::vec3 bboxCenter(const poca::core::BoundingBox& _bbox)
		{
			return glm::vec3(
				(_bbox[0] + _bbox[3]) * .5f,
				(_bbox[1] + _bbox[4]) * .5f,
				(_bbox[2] + _bbox[5]) * .5f);
		}

		poca::core::BoundingBox localObjectBoundingBox(const std::vector<poca::core::BasicComponentInterface*>& _components)
		{
			if (_components.empty())
				return poca::core::BoundingBox(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);

			poca::core::BoundingBox bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
			bool hasBBox = false;
			for (poca::core::BasicComponentInterface* component : _components) {
				if (component == nullptr)
					continue;
				const poca::core::BoundingBox& componentBBox = component->boundingBox();
				for (int axis = 0; axis < 3; axis++)
					bbox[axis] = std::min(bbox[axis], componentBBox[axis]);
				for (int axis = 3; axis < 6; axis++)
					bbox[axis] = std::max(bbox[axis], componentBBox[axis]);
				hasBBox = true;
			}

			return hasBBox ? bbox : poca::core::BoundingBox(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
		}
	}

	MyObject::MyObject() :poca::core::CommandableObject("Object")
	{
		m_internalId = poca::core::NbObjects++;
		m_modelMatrix = glm::mat4(1.f);
		m_rotationMatrix = glm::mat4(1.f);
		m_translation = glm::vec3(0.f);
	}

	MyObject::MyObject(const MyObject& _o) :poca::core::CommandableObject(_o), m_dir(_o.m_dir), m_name(_o.m_name)
	{
		m_internalId = poca::core::NbObjects++;
		m_modelMatrix = _o.m_modelMatrix;
		m_rotationMatrix = _o.m_rotationMatrix;
		m_translation = _o.m_translation;
		for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = _o.m_components.begin(); it != _o.m_components.end(); it++)
			this->addBasicComponent((*it)->copy());
	}

	MyObject::~MyObject()
	{
		for (poca::core::BasicComponentInterface* bci : m_components)
			delete bci;
		m_components.clear();
	}

	void MyObject::updateModelMatrixFromTransform()
	{
		const glm::vec3 center = bboxCenter(localObjectBoundingBox(m_components));
		const glm::mat4 translationMatrix = glm::translate(glm::mat4(1.f), glm::vec3(-m_translation.x, -m_translation.y, -m_translation.z));
		const glm::mat4 toCenter = glm::translate(glm::mat4(1.f), center);
		const glm::mat4 fromCenter = glm::translate(glm::mat4(1.f), -center);
		m_modelMatrix = translationMatrix * toCenter * m_rotationMatrix * fromCenter;
	}

	bool MyObject::translateCurrentObjectBy(const glm::vec3& _delta)
	{
		poca::core::MyObjectInterface* object = currentObject();
		if (object == nullptr)
			return false;

		object->setTranslationVector(object->getTranslationVector() + _delta);
		object->updateModelMatrixFromTransform();
		return true;
	}

	bool MyObject::rotateCurrentObjectBy(const glm::mat4& _delta)
	{
		poca::core::MyObjectInterface* object = currentObject();
		if (object == nullptr)
			return false;

		object->setRotationMatrix(_delta * object->getRotationMatrix());
		object->updateModelMatrixFromTransform();
		return true;
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
				poca::core::BasicComponentList* objComponentList = dynamic_cast<poca::core::BasicComponentList*>(m_components[n]);
				if (objComponentList) {
					poca::core::BasicComponentList* blist = dynamic_cast<poca::core::BasicComponentList*>(_bc);
					if (blist) {
						objComponentList->copyComponentsPtr(blist);
						blist->dontDeleteComponents();
						delete blist;
					}
					else {
						poca::core::BasicComponent* bc = dynamic_cast<poca::core::BasicComponent*>(_bc);
						if (bc) {
							objComponentList->addComponent(bc);
						}
					}

				}
				else {
					poca::core::BasicComponent* bc = dynamic_cast<poca::core::BasicComponent*>(_bc);
					if (bc) {
						delete m_components[n];
						m_components[n] = _bc;
					}
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
		
		/*if (m_components.size() == 1) {
			const auto& bbox = _bc->boundingBox();
			float translationX = bbox[0] + (bbox[3] - bbox[0]) / 2.f;
			float translationY = bbox[1] + (bbox[4] - bbox[1]) / 2.f;
			float translationZ = bbox[2] + (bbox[5] - bbox[2]) / 2.f;

			m_modelMatrixOG = glm::translate(glm::mat4(1.f), glm::vec3(-translationX, -translationY, -translationZ));
			m_modelMatrix = m_modelMatrixOG;
		}*/
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
		poca::core::CommandExecutionContext context;
		executeCommand(_ci, context);
	}

	void MyObject::executeCommand(poca::core::CommandInfo* _ci, const poca::core::CommandExecutionContext& _context)
	{
		poca::core::CommandExecutionResult result;
		executeCommand(_ci, _context, result);
	}

	void MyObject::executeCommand(poca::core::CommandInfo* _ci, const poca::core::CommandExecutionContext& _context, poca::core::CommandExecutionResult& _result)
	{
		if (_ci == NULL)
			return;
		if (_ci->nameCommand == "loadROIs") {
			float cal = 1.f;
			if(_ci->hasParameter("calibrationXY"))
				cal = _ci->getParameter<float>("calibrationXY");
			if (!_ci->hasParameter("filename"))
				return;
			std::string filename = _ci->getParameter<std::string>("filename");
			loadROIs(filename, cal);
		}
		else if (_ci->nameCommand == "saveROIs") {
			if (!_ci->hasParameter("filename"))
				return;
			std::string filename = _ci->getParameter<std::string>("filename");
			saveROIs(filename);
		}
		else if (_ci->hasParameter("widthDataset"))
			setWidth(_ci->getParameter<double>("widthDataset"));
		else if (_ci->hasParameter("heightDataset"))
			setHeight(_ci->getParameter<double>("heightDataset"));
		else if (_ci->nameCommand == "display") {
			poca::opengl::Camera* cam = nullptr;
			if (_context.has<poca::opengl::ActiveCamera>())
				cam = _context.get<poca::opengl::ActiveCamera>().camera;
			if (!cam) return;
			cam->setModelMatrix(cam->getModelMatrix() * m_modelMatrix);
		}
		poca::core::CommandableObject::executeCommand(_ci, _context, _result);
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
		poca::core::CommandExecutionContext context;
		executeCommandOnSpecificComponent(_nameComponent, _ci, context);
	}

	void MyObject::executeCommandOnSpecificComponent(const std::string& _nameComponent, poca::core::CommandInfo* _ci, const poca::core::CommandExecutionContext& _context)
	{
		poca::core::CommandExecutionResult result;
		executeCommandOnSpecificComponent(_nameComponent, _ci, _context, result);
	}

	void MyObject::executeCommandOnSpecificComponent(const std::string& _nameComponent, poca::core::CommandInfo* _ci, const poca::core::CommandExecutionContext& _context, poca::core::CommandExecutionResult& _result)
	{
		poca::core::BasicComponentInterface* bci = getBasicComponent(_nameComponent);
		if (bci)
			bci->executeCommand(_ci, _context, _result);
	}

	void MyObject::executeGlobalCommand(poca::core::CommandInfo* _ci)
	{
		poca::core::CommandExecutionContext context;
		executeGlobalCommand(_ci, context);
	}

	void MyObject::executeGlobalCommand(poca::core::CommandInfo* _ci, const poca::core::CommandExecutionContext& _context)
	{
		poca::core::CommandExecutionResult result;
		executeGlobalCommand(_ci, _context, result);
	}

	void MyObject::executeGlobalCommand(poca::core::CommandInfo* _ci, const poca::core::CommandExecutionContext& _context, poca::core::CommandExecutionResult& _result)
	{
		executeCommand(_ci, _context, _result);
		for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++) {
			poca::core::BasicComponentInterface* bc = *it;
			if (isComponentFamilyHandled(_result, bc->getName()))
				continue;
			bc->executeCommand(_ci, _context, _result);
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
		for (unsigned int n = 0; n < m_components.size(); n++) {
			poca::core::BasicComponentList* blist = dynamic_cast <poca::core::BasicComponentList*>(m_components[n]);
			if (blist && blist->nbComponents() > 0) {
				if (blist->getComponent(0)->getName() == _nameBC)
					return m_components[n];
			}
		}
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
			poca::core::BoundingBox bboxComp = transformBoundingBox(m_components.at(n)->boundingBox(), m_modelMatrix);
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

	void MyObject::reorganizeComponents(int _oldPosition, int _newPosition)
	{
		if (_oldPosition == _newPosition) return; // Nothing to do

		if (_oldPosition < _newPosition) {
			// Move forward: rotate left
			std::rotate(m_components.begin() + _oldPosition, m_components.begin() + _oldPosition + 1, m_components.begin() + _newPosition + 1);
		}
		else {
			// Move backward: rotate right
			std::rotate(m_components.begin() + _newPosition, m_components.begin() + _oldPosition, m_components.begin() + _oldPosition + 1);
		}
	}
}

