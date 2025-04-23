/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagramBasicCommands.cpp
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

#include <QtCore/QString>
#include <fstream>
#include <filesystem>

#include <General/Engine.hpp>
#include <General/Engine.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <Geometry/ObjectLists.hpp>
#include <General/PluginList.hpp>
#include <General/Misc.h>
#include <General/Histogram.hpp>
#include <Interfaces/PaletteInterface.hpp>
#include <General/MyData.hpp>
#include <OpenGL/Camera.hpp>

#include "VoronoiDiagramBasicCommands.hpp"
#include "VoronoiDiagramPlugin.hpp"

VoronoiDiagramBasicCommands::VoronoiDiagramBasicCommands(poca::geometry::VoronoiDiagram* _voro) :poca::core::Command("VoronoiDiagramBasicCommands")
{
	m_voronoi = _voro;

	
	float cutDistance = std::numeric_limits < float >::max(), minArea = 0.f, maxArea = std::numeric_limits < float >::max();
	size_t minLocs = 3, maxLocs = std::numeric_limits <size_t>::max();
	bool inROIs = false;

	const nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
	if (parameters.contains(name())) {
		nlohmann::json param = parameters[name()];
		if (param.contains("objectCreationParameters")) {
			if (param["objectCreationParameters"].contains("minLocs"))
				minLocs = param["objectCreationParameters"]["minLocs"].get<size_t>();
			if (param["objectCreationParameters"].contains("maxLocs"))
				maxLocs = param["objectCreationParameters"]["maxLocs"].get<size_t>();
			if (param["objectCreationParameters"].contains("minArea"))
				minArea = param["objectCreationParameters"]["minArea"].get<float>();
			if (param["objectCreationParameters"].contains("maxArea"))
				maxArea = param["objectCreationParameters"]["maxArea"].get<float>();
			if (param["objectCreationParameters"].contains("cutDistance"))
				cutDistance = param["objectCreationParameters"]["cutDistance"].get<float>();
			if (param["objectCreationParameters"].contains("inROIs"))
				inROIs = param["objectCreationParameters"]["inROIs"].get<bool>();
		}
	}
	addCommandInfo(poca::core::CommandInfo(true, "objectCreationParameters",
		"minLocs", minLocs,	"maxLocs", maxLocs, "minArea", minArea, "maxArea", maxArea,
		"cutDistance", cutDistance, "inROIs", inROIs));

}

VoronoiDiagramBasicCommands::VoronoiDiagramBasicCommands(const VoronoiDiagramBasicCommands& _o) : poca::core::Command(_o)
{
	m_voronoi = _o.m_voronoi;
}

VoronoiDiagramBasicCommands::~VoronoiDiagramBasicCommands()
{
}

void VoronoiDiagramBasicCommands::execute(poca::core::CommandInfo* _infos)
{
	if (hasCommand(_infos->nameCommand)) {
		loadParameters(*_infos);
	}
	else if (_infos->nameCommand == "createFilteredObjects") {
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_voronoi);
		if (obj == NULL) return;

		size_t minLocs = 3, maxLocs = std::numeric_limits <size_t>::max();
		float minArea = 0.f, maxArea = std::numeric_limits < float >::max(), cutDistance = std::numeric_limits < float >::max();
		bool inROIs = false;

		if (_infos->nbParameters() == 0) {
			minLocs = hasParameter("objectCreationParameters", "minLocs") ? getParameter<size_t>("objectCreationParameters", "minLocs") : minLocs;
			maxLocs = hasParameter("objectCreationParameters", "maxLocs") ? getParameter<size_t>("objectCreationParameters", "maxLocs") : maxLocs;
			minArea = hasParameter("objectCreationParameters", "minArea") ? getParameter<float>("objectCreationParameters", "minArea") : minArea;
			maxArea = hasParameter("objectCreationParameters", "maxArea") ? getParameter<float>("objectCreationParameters", "maxArea") : maxArea;
			cutDistance = hasParameter("objectCreationParameters", "cutDistance") ? getParameter<float>("objectCreationParameters", "cutDistance") : cutDistance;
			inROIs = hasParameter("objectCreationParameters", "inROIs") ? getParameter<bool>("objectCreationParameters", "inROIs") : inROIs;
		}
		else {
			minLocs = _infos->hasParameter("minLocs") ? _infos->getParameter<size_t>("minLocs") : minLocs;
			maxLocs = _infos->hasParameter("maxLocs") ? _infos->getParameter<size_t>("maxLocs") : maxLocs;
			minArea = _infos->hasParameter("minArea") ? _infos->getParameter<float>("minArea") : minArea;
			maxArea = _infos->hasParameter("maxArea") ? _infos->getParameter<float>("maxArea") : maxArea;
			cutDistance = _infos->hasParameter("cutDistance") ? _infos->getParameter<float>("cutDistance") : cutDistance;
			inROIs = _infos->hasParameter("inROIs") ? _infos->getParameter<bool>("inROIs") : inROIs;
		
			poca::core::CommandInfo ci(true, "objectCreationParameters",
				"cutDistance", cutDistance,
				"minLocs", minLocs,
				"maxLocs", maxLocs,
				"minArea", minArea,
				"maxArea", maxArea,
				"inROIs", inROIs);
			loadParameters(ci);
		}

		const std::vector <bool>& selection = m_voronoi->getSelection();
		poca::geometry::ObjectListFactory factory;
		poca::geometry::ObjectListInterface* objects = factory.createObjectList(obj, selection, cutDistance, minLocs, maxLocs, minArea, maxArea, inROIs);
		if (objects == NULL) return;
		objects->setBoundingBox(m_voronoi->boundingBox());
		VoronoiDiagramPlugin::m_plugins->addCommands(objects);
		if (!obj->hasBasicComponent("ObjectLists")) {
			poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(objects, *_infos, "VoronoiDiagramPlugin");
			VoronoiDiagramPlugin::m_plugins->addCommands(objsList);
			obj->addBasicComponent(objsList);
		}
		else {
			std::string text = _infos->json.dump(4);
			poca::geometry::ObjectLists* objsList = dynamic_cast<poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
			if (objsList)
				objsList->addObjectList(objects, *_infos, "VoronoiDiagramPlugin");
			std::cout << text << std::endl;
		}
		obj->notify("LoadObjCharacteristicsAllWidgets");
	}
	else if (_infos->nameCommand == "densityFactor") {
		float factor = _infos->getParameter<float>("factor");

		poca::core::BoundingBox bbox = m_voronoi->boundingBox();
		float meanD = m_voronoi->averageDensity();
		const std::vector <float>& feature = m_voronoi->getMyData("density")->getData<float>();
		std::vector <bool>& selection = m_voronoi->getSelection();

		for (size_t n = 0; n < feature.size(); n++)
			selection[n] = feature[n] >= (factor * meanD);

		m_voronoi->executeCommand(false, "updateFeature");
	}
	else if (_infos->nameCommand == "invertSelection") {
		std::vector <bool>& selection = m_voronoi->getSelection();
		selection.flip();
		m_voronoi->executeCommand(false, "updateFeature");
	}
	else if (_infos->nameCommand == "randomPointOnTheSphere") {
		poca::geometry::VoronoiDiagram2DOnSphere* voro = dynamic_cast<poca::geometry::VoronoiDiagram2DOnSphere*>(m_voronoi);
		if (!voro) return;
		auto nbPoints = voro->nbFaces();
		auto radius = voro->getRadius();
		auto centroid = voro->getCentroid();
		std::vector <poca::core::Vec3mf> randomPoints;
		poca::core::randomPointsOnUnitSphere(nbPoints, randomPoints);
		std::vector <float> xs(nbPoints), ys(nbPoints), zs(nbPoints);
		for (auto n = 0; n < randomPoints.size(); n++) {
			poca::core::Vec3mf vector = centroid + randomPoints[n] * radius;
			xs[n] = centroid.x() + vector.x();
			ys[n] = centroid.y() + vector.y();
			zs[n] = centroid.z() + vector.z();
		}
		std::map <std::string, std::vector <float>> data;
		data["x"] = xs;
		data["y"] = ys;
		data["z"] = zs;
		poca::geometry::DetectionSet* dset = new poca::geometry::DetectionSet(data);
		_infos->addParameter("newObject", dset);
	}
	else if (_infos->nameCommand == "clustersForChallenge") {
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_voronoi);
		poca::core::BasicComponentInterface* dset = obj->getBasicComponent("DetectionSet");
		if (dset == NULL)
			return;

		size_t minNbLocs = 1, maxNbLocs = std::numeric_limits < size_t >::max();
		float from = 1.f, to = 2.f, step = 0.5f;
		if (_infos->hasParameter("minNbLocs"))
			minNbLocs = _infos->getParameter<size_t>("minNbLocs");
		if (_infos->hasParameter("maxNbLocs"))
			maxNbLocs = _infos->getParameter<size_t>("maxNbLocs");
		if (_infos->hasParameter("from"))
			from = _infos->getParameter<float>("from");
		if (_infos->hasParameter("to"))
			to = _infos->getParameter<float>("to");
		if (_infos->hasParameter("step"))
			step = _infos->getParameter<float>("step");

		int n = 1;
		clock_t t1, t2;
		t1 = clock();
		for (auto factor = from; factor <= to; factor += step, n++) {
			execute(&poca::core::CommandInfo(false, "densityFactor", "factor", factor));
			dset->executeCommand(false, "clustersForChallenge", "minNbLocs", minNbLocs, "maxNbLocs", maxNbLocs, "selection", std::string("VoronoiDiagram"), "factor", factor, "currentScreen", n, "object", obj);
		}
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;

		std::ofstream fs("e:/timings.txt", std::fstream::out | std::fstream::app);
		fs << elapsed << std::endl;
		fs.close();
	}
	else if (_infos->nameCommand == "saveAsSVG") {
		QString filename = (_infos->getParameter<std::string>("filename")).c_str();
		filename.insert(filename.lastIndexOf("."), "_voronoi");
		saveAsSVG(filename);
	}
}

poca::core::CommandInfo VoronoiDiagramBasicCommands::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "objectCreationParameters" || _nameCommand == "createFilteredObjects") {
		poca::core::CommandInfo ci(false, _nameCommand);
		if (_parameters.contains("minLocs"))
			ci.addParameter("minLocs", _parameters["minLocs"].get<size_t>());
		if (_parameters.contains("maxLocs"))
			ci.addParameter("maxLocs", _parameters["maxLocs"].get<size_t>());
		if (_parameters.contains("minArea"))
			ci.addParameter("minArea", _parameters["minArea"].get<float>());
		if (_parameters.contains("maxArea"))
			ci.addParameter("maxArea", _parameters["maxArea"].get<float>());
		if (_parameters.contains("cutDistance"))
			ci.addParameter("cutDistance", _parameters["cutDistance"].get<float>());
		if (_parameters.contains("inROIs"))
			ci.addParameter("inROIs", _parameters["inROIs"].get<bool>());
		return ci;
	}
	else if (_nameCommand == "densityFactor") {
		if (_parameters.contains("factor")) {
			float val = _parameters["factor"].get<float>();
			return poca::core::CommandInfo(false, _nameCommand, "factor", val);
		}
	}
	else if (_nameCommand == "invertSelection" || _nameCommand == "randomPointOnTheSphere") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	else if (_nameCommand == "clustersForChallenge") {
		poca::core::CommandInfo ci(false, _nameCommand);
		if (_parameters.contains("minNbLocs"))
			ci.addParameter("minNbLocs", _parameters["minNbLocs"].get<size_t>());
		if (_parameters.contains("maxNbLocs"))
			ci.addParameter("maxNbLocs", _parameters["maxNbLocs"].get<size_t>());
		if (_parameters.contains("from"))
			ci.addParameter("from", _parameters["from"].get<float>());
		if (_parameters.contains("to"))
			ci.addParameter("to", _parameters["to"].get<float>());
		if (_parameters.contains("step"))
			ci.addParameter("step", _parameters["step"].get<float>());
		return ci;
	}
	else if (_nameCommand == "saveAsSVG") {
		if (_parameters.contains("filename")) {
			std::string val = _parameters["filename"].get<std::string>();
			return poca::core::CommandInfo(false, _nameCommand, "filename", val);
		}
	}
	return poca::core::CommandInfo();
}

poca::core::Command* VoronoiDiagramBasicCommands::copy()
{
	return new VoronoiDiagramBasicCommands(*this);
}

void VoronoiDiagramBasicCommands::saveAsSVG(const QString& _filename) const
{
	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::opengl::CameraInterface* cam = engine->getCamera(m_voronoi);
	poca::core::Vec3mf direction = poca::core::Vec3mf(cam->getEye().x, cam->getEye().y, cam->getEye().z);

	poca::core::BoundingBox bbox = m_voronoi->boundingBox();
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

	/*std::vector <poca::core::Vec3mf> lines;
	m_voronoi->generateLines(lines);
	poca::core::Histogram<float>* histogram = dynamic_cast<poca::core::Histogram<float>*>(m_voronoi->getCurrentHistogram());
	const std::vector<float>& values = histogram->getValues();
	const std::vector<bool>& selection = m_voronoi->getSelection();
	float minH = histogram->getMin(), maxH = histogram->getMax(), interH = maxH - minH;

	std::vector <float> featureValues;
	m_voronoi->getFeatureInSelection(featureValues, values, selection, std::numeric_limits <float>::max(), true);

	char col[32];
	poca::core::PaletteInterface* pal = m_voronoi->getPalette();
	for (size_t n = 0; n < lines.size(); n += 2) {
		if (featureValues[n] == std::numeric_limits <float>::max()) continue;
		float valPal = (featureValues[n] - minH) / interH;
		poca::core::Color4uc c = pal->getColor(valPal);
		unsigned char r = c[0], g = c[1], b = c[2];
		poca::core::getColorStringUC(r, g, b, col);
		fs << "<line x1 =\"";
		fs << lines[n].x() << "\" y1=\"";
		fs << lines[n].y() << "\" x2=\"";
		fs << lines[n+1].x() << "\" y2=\"";
		fs << lines[n+1].y() << "\" stroke=\"" << col << "\" stroke-width=\"1\"/>\n";
	}
	fs.close();*/
	bool fill = false;
	if (m_voronoi->hasParameter("fill"))
		fill = m_voronoi->getParameter<bool>("fill");

	std::vector <poca::core::Vec3mf> cells;
	if(fill)
		m_voronoi->generateTriangles(cells);
	else
		m_voronoi->generateLines(cells);
	poca::core::Histogram<float>* histogram = dynamic_cast<poca::core::Histogram<float>*>(m_voronoi->getCurrentHistogram());
	const std::vector<float>& values = histogram->getValues();
	const std::vector<bool>& selection = m_voronoi->getSelection();
	float minH = histogram->getMin(), maxH = histogram->getMax(), interH = maxH - minH;

	std::vector <float> featureValues;
	m_voronoi->getFeatureInSelection(featureValues, values, selection, std::numeric_limits <float>::max(), !fill);


	char col[32];
	poca::core::PaletteInterface* pal = m_voronoi->getPalette();
	if (fill) {
		for (size_t n = 0; n < cells.size(); n += 3) {
			if (featureValues[n] == std::numeric_limits <float>::max()) continue;
			float valPal = (featureValues[n] - minH) / interH;
			poca::core::Color4uc c = pal->getColor(valPal);
			unsigned char r = c[0], g = c[1], b = c[2];
			poca::core::getColorStringUC(r, g, b, col);
			glm::vec2 p1 = cam->worldToScreenCoordinates(glm::vec3(cells[n].x(), cells[n].y(), cells[n].z()));
			glm::vec2 p2 = cam->worldToScreenCoordinates(glm::vec3(cells[n + 1].x(), cells[n + 1].y(), cells[n + 1].z()));
			glm::vec2 p3 = cam->worldToScreenCoordinates(glm::vec3(cells[n + 2].x(), cells[n + 2].y(), cells[n + 2].z()));

			fs << "<polygon points =\"";
			fs << p1.x << ",";
			fs << p1.y << " ";
			fs << p2.x << ",";
			fs << p2.y << " ";
			fs << p3.x << ",";
			fs << p3.y << "\" stroke=\"" << col << "\" fill=\"" << col << "\" stroke-width=\"0.1\"/>\n";
		}
	}
	else {
		for (size_t n = 0; n < cells.size(); n += 2) {
			if (featureValues[n] == std::numeric_limits <float>::max()) continue;
			float valPal = (featureValues[n] - minH) / interH;
			poca::core::Color4uc c = pal->getColor(valPal);
			unsigned char r = c[0], g = c[1], b = c[2];
			poca::core::getColorStringUC(r, g, b, col);
			glm::vec2 p1 = cam->worldToScreenCoordinates(glm::vec3(cells[n].x(), cells[n].y(), cells[n].z()));
			glm::vec2 p2 = cam->worldToScreenCoordinates(glm::vec3(cells[n + 1].x(), cells[n + 1].y(), cells[n + 1].z()));

			fs << "<line x1 =\"";
			fs << p1.x << "\" y1=\"";
			fs << p1.y << "\" x2=\"";
			fs << p2.x << "\" y2=\"";
			fs << p2.y << "\" stroke=\"" << col << "\" stroke-width=\"1\"/>\n";
		}
	}
	fs.close();
}