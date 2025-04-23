/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      NearestLocsMultiColorCommands.cpp
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

#include <Windows.h>
#include <gl/glew.h>
#include <gl/GL.h>
#include <QtGui/QOpenGLFramebufferObject>

#include <Geometry/ObjectLists.hpp>
#include <Geometry/DelaunayTriangulation.hpp>
#include <OpenGL/Shader.hpp>
#include <General/Engine.hpp>
#include <General/Misc.h>
#include <OpenGL/Helper.h>
#include <Objects/MyObject.hpp>
#include <Interfaces/ROIInterface.hpp>
#include <Geometry/nanoflann.hpp>
#include <General/Histogram.hpp>
#include <General/Palette.hpp>

#include "NearestLocsMultiColorCommands.hpp"

NearestLocsMultiColorCommands::NearestLocsMultiColorCommands(poca::core::MyObject* _obj) :poca::core::Command("NearestLocsMultiColorCommands")
{
	m_obj = _obj;
	m_displayToCentroids = m_displayToOutlines = true;
	float maxD = FLT_MAX;

	
	const nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
	addCommandInfo(poca::core::CommandInfo(false, "computeNearestNeighMulticolor", "inROIs", true, "reference", (uint32_t)1, "maxDistance", maxD));
	addCommandInfo(poca::core::CommandInfo(false, "displayCentroidsNearestNeighMulticolor", true));
	addCommandInfo(poca::core::CommandInfo(false, "displayOutlinesNearestNeighMulticolor", true));

	if (parameters.contains(name())) {
		nlohmann::json param = parameters[name()];
		if (param.contains("computeNearestNeighMulticolor")) {
			bool inROIs = true;
			uint32_t idref = 1;
			if (param["computeNearestNeighMulticolor"].contains("inROIs"))
				inROIs = param["computeNearestNeighMulticolor"]["inROIs"].get<bool>();
			if (param["computeNearestNeighMulticolor"].contains("reference"))
				idref = param["computeNearestNeighMulticolor"]["reference"].get<uint32_t>();
			if (param["computeNearestNeighMulticolor"].contains("maxDistance"))
				maxD = param["computeNearestNeighMulticolor"]["maxDistance"].get<float>();
			loadParameters(poca::core::CommandInfo(false, "computeNearestNeighMulticolor", "inROIs", inROIs, "reference", idref, "maxDistance", maxD));
		}
		if (param.contains("displayCentroidsNearestNeighMulticolor"))
			loadParameters(poca::core::CommandInfo(false, "displayCentroidsNearestNeighMulticolor", param["displayCentroidsNearestNeighMulticolor"].get<bool>()));
		if (param.contains("displayOutlinesNearestNeighMulticolor"))
			loadParameters(poca::core::CommandInfo(false, "displayOutlinesNearestNeighMulticolor", param["displayOutlinesNearestNeighMulticolor"].get<bool>()));
	}
	m_palette = poca::core::Palette::getStaticLutPtr("HotCold2");
	m_histogramCentroids = m_histogramOutlines = NULL;
}

NearestLocsMultiColorCommands::NearestLocsMultiColorCommands(const NearestLocsMultiColorCommands& _o) : poca::core::Command(_o)
{
	m_obj = _o.m_obj;
}

NearestLocsMultiColorCommands::~NearestLocsMultiColorCommands()
{
}

void NearestLocsMultiColorCommands::execute(poca::core::CommandInfo* _infos)
{
	if (_infos->nameCommand == "display") {
		poca::opengl::Camera* cam = _infos->getParameterPtr<poca::opengl::Camera>("camera");
		bool offscrean = false;
		if (_infos->hasParameter("offscreen"))
			offscrean = _infos->getParameter<bool>("offscreen");
		display(cam, offscrean);
	}
	else if (_infos->nameCommand == "freeGPU") {
		freeGPUMemory();
	}
	else if (_infos->nameCommand == "computeNearestNeighMulticolor") {
		bool inROIs = true;
		uint32_t reference = 1;
		float maxD = DBL_MAX;
		if (_infos->hasParameter("inROIs"))
			inROIs = _infos->getParameter<bool>("inROIs");
		if (_infos->hasParameter("reference"))
			reference = _infos->getParameter<uint32_t>("reference");
		if (_infos->hasParameter("maxDistance"))
			maxD = _infos->getParameter<float>("maxDistance");

		computeNearestLocMulticolor(inROIs, reference, maxD);
	}
	else if (_infos->nameCommand == "transferSelectedObjectsNearestNeighMulticolor")
		transferObjects();
	else if (_infos->nameCommand == "saveDistancesNearestNeighMulticolor") {
		std::string path;
		if (_infos->hasParameter("path"))
			path = _infos->getParameter<std::string>("path");
		if (path.empty()) return;
		saveDistances(path);
	}
	else if (hasParameter(_infos->nameCommand)) {
		loadParameters(*_infos);
	}
}

poca::core::CommandInfo NearestLocsMultiColorCommands::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "computeNearestNeighMulticolor") {
		bool val = _parameters.get<bool>();
		bool inROIs = true;
		uint32_t reference = 1;
		float maxD = DBL_MAX;
		if (_parameters.contains("inROIs"))
			inROIs = _parameters["inROIs"].get<bool>();
		if (_parameters.contains("reference"))
			reference = _parameters["reference"].get<uint32_t>();
		if (_parameters.contains("maxDistance"))
			maxD = _parameters["maxDistance"].get<float>();
		return poca::core::CommandInfo(false, _nameCommand, "inROIs", inROIs, "reference", reference, "maxDistance", maxD);
	}
	else if (_nameCommand == "displayCentroidsNearestNeighMulticolor" || _nameCommand == "displayOutlinesNearestNeighMulticolor") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "transferSelectedObjectsNearestNeighMulticolor") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	else if (_nameCommand == "saveDistancesNearestNeighMulticolor") {
		std::string path;
		if (_parameters.contains("path"))
			path = _parameters["path"].get<std::string>();
		return poca::core::CommandInfo(false, _nameCommand, "path", path);
	}

	return poca::core::CommandInfo();
}

poca::core::Command* NearestLocsMultiColorCommands::copy()
{
	return new NearestLocsMultiColorCommands(*this);
}

void NearestLocsMultiColorCommands::freeGPUMemory()
{
	m_toCentroidsBuffer.freeGPUMemory();
	m_toOutlinesBuffer.freeGPUMemory();
}

void NearestLocsMultiColorCommands::createDisplay(const std::vector <poca::core::Vec3mf>& _locToCentroids, const std::vector <poca::core::Vec3mf>& _locToOutlines)
{
	freeGPUMemory();

	m_toCentroidsBuffer.generateBuffer(_locToCentroids.size(), 3, GL_FLOAT);
	m_toOutlinesBuffer.generateBuffer(_locToOutlines.size(), 3, GL_FLOAT);
	m_toCentroidsBuffer.updateBuffer(_locToCentroids.data());
	m_toOutlinesBuffer.updateBuffer(_locToOutlines.data());
}

void NearestLocsMultiColorCommands::display(poca::opengl::Camera* _cam, const bool _offscreen)
{
	//if (!m_colocalization->isSelected()) return;

	drawElements(_cam);
}

void NearestLocsMultiColorCommands::drawElements(poca::opengl::Camera* _cam)
{
	GL_CHECK_ERRORS(); 
	if (m_toCentroidsBuffer.empty()) return;
	bool displayCentroids = getParameter<bool>("displayCentroidsNearestNeighMulticolor");
	bool displayOutlines = getParameter<bool>("displayOutlinesNearestNeighMulticolor");

	if(displayCentroids)
		_cam->drawUniformShader(m_toCentroidsBuffer, poca::core::Color4D(1.f, 0.f, 0.f, 1.f));
	if (displayOutlines)
		_cam->drawUniformShader(m_toOutlinesBuffer, poca::core::Color4D(0.f, 1.f, 1.f, 1.f));
	GL_CHECK_ERRORS();
}

void NearestLocsMultiColorCommands::computeNearestLocMulticolor(const bool _inROIs, const uint32_t _referenceId, const float _maxD)
{
	m_referenceId = _referenceId;
	poca::geometry::ObjectListInterface* objs[2] = { NULL, NULL };
	for (auto n = 0; n < 2; n++) {
		poca::core::MyObjectInterface* obj = m_obj->getObject(n);
		poca::core::BasicComponentInterface* bc = obj->getBasicComponent("ObjectLists");
		poca::geometry::ObjectLists* tmp = dynamic_cast <poca::geometry::ObjectLists*>(bc);
		objs[n] = tmp == NULL ? NULL : tmp->currentObjectList();
	}
	if (objs[0] == NULL || objs[1] == NULL) return;

	poca::geometry::ObjectListInterface* reference = objs[m_referenceId], * objects = objs[(m_referenceId + 1) % 2];

	m_selectedObjects.resize(objects->nbElements());
	std::fill(m_selectedObjects.begin(), m_selectedObjects.end(), true);
	std::vector <poca::core::ROIInterface*>& ROIs = _inROIs ? m_obj->getROIs() : std::vector <poca::core::ROIInterface*>();

	std::vector <poca::core::Vec3mf> centroids(objects->nbElements());
	for (auto n = 0; n < objects->nbElements(); n++)
		centroids[n] = objects->computeBarycenterElement(n);

	if (!ROIs.empty()) {
		for (auto n = 0; n < objects->nbElements(); n++) {
			bool inside = false;
			for (auto n2 = 0; n2 < ROIs.size() && !inside; n2++)
				inside = ROIs[n2]->inside(centroids[n].x(), centroids[n].y(), centroids[n].z());
			m_selectedObjects[n] = inside;
		}
	}

	const std::vector <poca::core::Vec3mf>& outlines = reference->getOutlinesObjects().getData();
	KdPointCloud_3D_D cloud;
	cloud.m_pts.resize(outlines.size());
	for (size_t n = 0; n < outlines.size(); n++) {
		cloud.m_pts[n].m_x = outlines[n].x();
		cloud.m_pts[n].m_y = outlines[n].y();
		cloud.m_pts[n].m_z = outlines[n].z();
	}
	KdTree_3D_double kdTree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	kdTree.buildIndex();
	const std::size_t num_results = 1;
	std::vector<size_t> ret_index(num_results);
	std::vector<double> out_dist_sqr(num_results);


	m_idxObjects.clear();
	m_distancesToCentroids.clear();
	m_distancesToOutlines.clear();

	std::vector <poca::core::Vec3mf> lineToCentroids, lineToOutlines;

	const std::vector <poca::core::Vec3mf>& outObjects = objects->getOutlinesObjects().getData();
	const std::vector <uint32_t>& firsts = objects->getOutlinesObjects().getFirstElements();
	for (auto n = 0; n < objects->nbElements(); n++) {
		if (!m_selectedObjects[n]) continue;

		//compute distance from reference outline to centroids objects
		uint32_t indexLocCendroid = 0;
		const double queryPt[3] = { centroids[n].x(), centroids[n].y(), centroids[n].z() };
		kdTree.knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
		float dtmp = sqrt(out_dist_sqr[0]);
		indexLocCendroid = ret_index[0];

		//compute distance from reference outline to objects outline
		double d = DBL_MAX;
		uint32_t indexLocOut = 0;
		poca::core::Vec3mf posLoc;
		for (auto id = firsts[n]; id < firsts[n + 1]; id++) {
			const double queryPt[3] = { outObjects[id].x(), outObjects[id].y(), outObjects[id].z() };
			kdTree.knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
			if (out_dist_sqr[0] < d) {
				d = out_dist_sqr[0];
				indexLocOut = ret_index[0];
				posLoc = outObjects[id];
			}
		}
		d = sqrt(d);

		//keep if both distances are below maxD
		if (dtmp < _maxD && d < _maxD) {
			m_idxObjects.push_back(n + 1);
			m_distancesToCentroids.push_back(dtmp);
			m_distancesToOutlines.push_back(d);

			lineToCentroids.push_back(centroids[n]);
			lineToCentroids.push_back(poca::core::Vec3mf(cloud.m_pts[indexLocCendroid].m_x, cloud.m_pts[indexLocCendroid].m_y, cloud.m_pts[indexLocCendroid].m_z));

			lineToOutlines.push_back(posLoc);
			lineToOutlines.push_back(poca::core::Vec3mf(cloud.m_pts[indexLocOut].m_x, cloud.m_pts[indexLocOut].m_y, cloud.m_pts[indexLocOut].m_z));
		}
	}

	if (m_histogramCentroids != NULL)
		delete m_histogramCentroids;
	m_histogramCentroids = new poca::core::Histogram(m_distancesToCentroids, m_distancesToCentroids.size(), false);

	if (m_histogramOutlines != NULL)
		delete m_histogramOutlines;
	m_histogramOutlines = new poca::core::Histogram(m_distancesToOutlines, m_distancesToOutlines.size(), false);

	createDisplay(lineToCentroids, lineToOutlines);

	m_obj->notify("LoadObjCharacteristicsNearestLocsMultiColorWidget");
	m_obj->notifyAll("updateDisplay");
}

void NearestLocsMultiColorCommands::transferObjects() const
{
	 poca::core::BasicComponentInterface * bc = m_obj->getObject((m_referenceId + 1) % 2)->getBasicComponent("ObjectList");
	 poca::geometry::ObjectListInterface* objs = dynamic_cast <poca::geometry::ObjectListInterface*>(bc);
	 if (objs == NULL) return;

	 std::vector <bool> selection(objs->nbElements(), false);

	 for (auto n = 0; n < m_idxObjects.size(); n++)
		 selection[m_idxObjects[n] - 1] = true;

	 objs->setSelection(selection);
	 objs->executeCommand(false, "updateFeature");
}

void NearestLocsMultiColorCommands::saveDistances(const std::string& _path) const
{
	if (m_distancesToCentroids.empty()) return;

	std::ofstream fs(_path);
	if (!fs) {
		std::cout << "System failed to open " << _path << std::endl;
		return;
	}
	else
		std::cout << "Saving detections in file " << _path << std::endl;

	fs << "index object\tdistance to centroids\tdistance to outline" << std::endl;
	for (auto n = 0; n < m_distancesToCentroids.size(); n++)
		fs << m_idxObjects[n] << "\t" << m_distancesToCentroids[n] << "\t" << m_distancesToOutlines[n] << std::endl;
	fs.close();
}
