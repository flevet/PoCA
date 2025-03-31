/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DBSCANCommand.cpp
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

#include <Windows.h>
#include <gl/glew.h>
#include <gl/GL.h>
#include <QtWidgets/QMessageBox>
#include <qmath.h>

#include <General/Engine.hpp>
#include <Geometry/DetectionSet.hpp>
#include <General/Misc.h>
#include <Interfaces/HistogramInterface.hpp>
#include <Interfaces/ObjectFeaturesFactoryInterface.hpp>
#include <Interfaces/DelaunayTriangulationInterface.hpp>
#include <OpenGL/Shader.hpp>
#include <OpenGL/Camera.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <Geometry/ObjectLists.hpp>
#include <General/Engine.hpp>
#include <General/PluginList.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <Objects/MyObject.hpp>
#include <General/MyData.hpp>

#include "DBSCANCommand.hpp"
#include "DBSCANPlugin.hpp"

DBSCANCommand::DBSCANCommand(poca::geometry::DetectionSet* _ds) :poca::core::Command("DBSCANCommand"), m_histSizes(NULL), m_updateColorBuffer(false)
{
	m_dset = _ds;

	
	const nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
	addCommandInfo(poca::core::CommandInfo(false, "DBSCAN", "radius", 50.f, "min", (uint32_t)20));
	addCommandInfo(poca::core::CommandInfo(false, "displayDBSCAN", true));
	if (parameters.contains(name())) {
		nlohmann::json param = parameters[name()];
		if(param.contains("DBSCAN") && param["DBSCAN"].contains("radius") && param["DBSCAN"].contains("min"))
			loadParameters(poca::core::CommandInfo(false, "DBSCAN", "radius", param["DBSCAN"]["radius"].get<float>(), "min", param["DBSCAN"]["min"].get<uint32_t>()));
		if (param.contains("displayDBSCAN"))
			loadParameters(poca::core::CommandInfo(false, "displayDBSCAN", param["displayDBSCAN"].get<bool>()));
	}
}

DBSCANCommand::DBSCANCommand(const DBSCANCommand& _o): poca::core::Command(_o)
{
	m_dset = _o.m_dset;
}

DBSCANCommand::~DBSCANCommand()
{
}

poca::core::Command* DBSCANCommand::copy()
{
	return new DBSCANCommand(*this);
}

void DBSCANCommand::execute(poca::core::CommandInfo* _infos)
{
	if (hasCommand(_infos->nameCommand)) {
		loadParameters(*_infos);
	}
	if (_infos->nameCommand == "DBSCAN") {
		clock_t t1, t2;
		t1 = clock();
		float radius = _infos->getParameter<float>("radius");
		uint32_t minNb = _infos->getParameter<uint32_t>("min");
		uint32_t minNbForCluster = _infos->getParameter<uint32_t>("minNbForCluster");
		computeDBSCAN(radius, minNb, minNbForCluster);
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		printf("Time for computing DBSCAN: %ld ms\n", elapsed);
	}
	else if (_infos->nameCommand == "getDBSCANCommand") {
		_infos->addParameter("dbscanCommand", this);
	}
	else if (_infos->nameCommand == "display") {
		poca::opengl::Camera* cam = _infos->getParameterPtr<poca::opengl::Camera>("camera");
		bool offscrean = false;
		if (_infos->hasParameter("offscreen"))
			offscrean = _infos->getParameter<bool>("offscreen");
		display(cam, offscrean);
	}
	else if (_infos->nameCommand == "createDBSCANObjects") {
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_dset);
		if (obj == NULL) return;

		std::vector <uint32_t> selection(m_dset->nbElements(), std::numeric_limits<uint32_t>::max());
		uint32_t cpt = 0;
		for (const std::vector<uint32_t>& cluster : m_dbscan.Clusters) {
			for (const uint32_t& index : cluster)
				selection[index] = cpt;
			cpt++;
		}
		poca::geometry::ObjectListFactory factory;
		poca::geometry::ObjectListInterface* objects = factory.createObjectListAlreadyIdentified(obj, selection);

		if (objects == NULL) return;
		objects->setBoundingBox(m_dset->boundingBox());
		DBSCANPlugin::m_plugins->addCommands(objects);
		if (!obj->hasBasicComponent("ObjectLists")) {
			poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(objects, *_infos, "DBSCANPlugin");
			DBSCANPlugin::m_plugins->addCommands(objsList);
			obj->addBasicComponent(objsList);
		}
		else {
			std::string text = _infos->json.dump(4);
			poca::geometry::ObjectLists* objsList = dynamic_cast<poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
			if (objsList)
				objsList->addObjectList(objects, *_infos, "DBSCANPlugin");
			std::cout << text << std::endl;
		}
		obj->notify("LoadObjCharacteristicsAllWidgets");

		/*objects->setBoundingBox(m_dset->boundingBox());
		obj->addBasicComponent(objects);
		obj->notify(poca::core::CommandInfo(false, "addCommandToSpecificComponent", "component", (poca::core::BasicComponentInterface*)objects));*/
	}
}

poca::core::CommandInfo DBSCANCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "DBSCAN") {
		float radius;
		uint32_t minNb;
		bool complete = _parameters.contains("radius");
		if (complete)
			radius = _parameters["radius"].get<float>();
		complete &= _parameters.contains("min");
		if (complete) {
			minNb = _parameters["min"].get<uint32_t>();
			return poca::core::CommandInfo(false, _nameCommand, "radius", radius, "min", minNb);
		}
	}
	else if (_nameCommand == "displayDBSCAN") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "createDBSCANObjects") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	return poca::core::CommandInfo();
}

void DBSCANCommand::computeDBSCAN(const float _radius, const uint32_t _minNb, const uint32_t _minNbForCluster, const float _dZ)
{
	bool hasZ = m_dset->hasData("z");
	std::vector <float>& xs = m_dset->getMyData("x")->getData<float>();
	std::vector <float>& ys = m_dset->getMyData("y")->getData<float>();
	const std::vector <float>& zs = hasZ ? m_dset->getMyData("z")->getData<float>() : std::vector <float>(xs.size(), 0.f);

	std::vector<dbvec3f> data;
	for (size_t n = 0; n < xs.size(); n++)
		data.push_back(dbvec3f{ xs[n], ys[n], zs[n] });

	m_dbscan.Run(&data, 3, _radius, _minNb, _minNbForCluster, _dZ);
	m_sizeClusters.resize(m_dbscan.Clusters.size());
	m_majorAxisClusters.resize(m_dbscan.Clusters.size());
	m_minorAxisClusters.resize(m_dbscan.Clusters.size());
	m_nbLocsClusters.resize(m_dbscan.Clusters.size());

	poca::geometry::ObjectFeaturesFactoryInterface* factory = poca::geometry::createObjectFeaturesFactory();
	std::vector <float> resPCA(factory->nbFeaturesPCA(m_dset->hasData("z")));
	size_t cpt = 0;
	for (std::vector<std::vector<uint32_t>>::const_iterator it = m_dbscan.Clusters.begin(); it != m_dbscan.Clusters.end(); it++, cpt++) {
		const uint32_t* locIds = it->data();
		factory->computePCA(locIds, it->size(), xs.data(), ys.data(), hasZ ? zs.data() : NULL, &resPCA[0]);
		if (!m_dset->hasData("z")) {
			m_sizeClusters[cpt] = (resPCA[8] + resPCA[9]) / 2.f;
			m_majorAxisClusters[cpt] = resPCA[8];
			m_minorAxisClusters[cpt] = resPCA[9];
			m_nbLocsClusters[cpt] = (uint32_t)it->size();
		}
		else {
			m_sizeClusters[cpt] = (resPCA[3] + resPCA[4] + resPCA[5]) / 3.f;
			m_majorAxisClusters[cpt] = resPCA[3];
			m_minorAxisClusters[cpt] = resPCA[4];
			m_nbLocsClusters[cpt] = (uint32_t)it->size();
		}
	}
	delete factory;
	if (m_histSizes == NULL)
		m_histSizes = new poca::core::Histogram<float>(m_sizeClusters, false, 50);
	else
		m_histSizes->setHistogram(m_sizeClusters, false, 50);
	m_updateColorBuffer = true;
}

void DBSCANCommand::createDisplay()
{
	m_pointBuffer.freeGPUMemory();
	m_colorBuffer.freeGPUMemory();

	bool hasZ = m_dset->hasData("z");
	const std::vector <float>& xs = m_dset->getMyData("x")->getData<float>();
	const std::vector <float>& ys = m_dset->getMyData("y")->getData<float>();
	const std::vector <float>& zs = hasZ ? m_dset->getMyData("z")->getData<float>() : std::vector <float>(xs.size(), 0.f);

	std::vector <poca::core::Vec3mf> points(xs.size());
	for (size_t n = 0; n < xs.size(); n++)
		points[n].set(xs[n], ys[n], zs[n]);
	m_pointBuffer.generateBuffer(points.size(), 512 * 512, 3, GL_FLOAT);
	m_pointBuffer.updateBuffer(points.data());

	m_colorBuffer.generateBuffer(points.size(), 512 * 512, 4, GL_FLOAT);
}

void DBSCANCommand::updateColorBuffer()
{
	std::vector <poca::core::Color4D> colors(m_dset->nbElements());
	for (size_t n = 0; n < m_dbscan.Noise.size(); n++)
		colors[m_dbscan.Noise[n]].set(0.f, 0.f, 0.f, 0.f);
	for (const std::vector<uint32_t>& cluster : m_dbscan.Clusters) {
		float r = (float)rand() / (float)RAND_MAX;
		float g = (float)rand() / (float)RAND_MAX;
		float b = (float)rand() / (float)RAND_MAX;
		for (const uint32_t& index : cluster)
			colors[index].set(r, g, b, 1.f);
	}
	m_colorBuffer.updateBuffer(colors.data());
	m_updateColorBuffer = false;
}

void DBSCANCommand::display(poca::opengl::Camera* _cam, const bool _offscreen)
{
	if (m_histSizes == NULL) return;

	if (m_pointBuffer.empty())
		createDisplay();
	if (m_updateColorBuffer)
		updateColorBuffer();

	bool displayClusters = getParameter<bool>("displayDBSCAN");
	if (!displayClusters) return;

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	_cam->drawSimpleShaderWithColor<poca::core::Vec3mf, poca::core::Color4D>(m_pointBuffer, m_colorBuffer);
	glDisable(GL_BLEND);
}

