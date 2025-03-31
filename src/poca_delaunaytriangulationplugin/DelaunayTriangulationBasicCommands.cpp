/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DelaunayTriangulationBasicCommands.cpp
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

#include <General/Engine.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <Geometry/ObjectLists.hpp>
#include <General/PluginList.hpp>
#include <General/Histogram.hpp>
#include <Interfaces/PaletteInterface.hpp>
#include <General/Misc.h>
#include <General/Engine.hpp>

#include "DelaunayTriangulationBasicCommands.hpp"
#include "DelaunayTriangulationPlugin.hpp"

DelaunayTriangulationBasicCommands::DelaunayTriangulationBasicCommands(poca::geometry::DelaunayTriangulationInterface* _delau) :poca::core::Command("DelaunayTriangulationBasicCommands")
{
	m_delaunay = _delau;

	
	bool useDistance = true, useMinLocs = true, useMaxLocs = false, useMinArea = false, useMaxArea = false;
	size_t minLocs = 3, maxLocs = 500;
	float cutDistance = 50.f, minArea = 0.f, maxArea = 1000.f;
	bool inROIs = false;

	const nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
	if (parameters.contains(name())){
		nlohmann::json param = parameters[name()];
		if (param.contains("objectCreationParameters")) {
			if (param["objectCreationParameters"].contains("useDistance"))
				useDistance = param["objectCreationParameters"]["useDistance"].get<bool>();
			if (param["objectCreationParameters"].contains("useMinLocs"))
				useMinLocs = param["objectCreationParameters"]["useMinLocs"].get<bool>();
			if (param["objectCreationParameters"].contains("useMaxLocs"))
				useMaxLocs = param["objectCreationParameters"]["useMaxLocs"].get<bool>();
			if (param["objectCreationParameters"].contains("useMinArea"))
				useMinArea = param["objectCreationParameters"]["useMinArea"].get<bool>();
			if (param["objectCreationParameters"].contains("useMaxArea"))
				useMaxArea = param["objectCreationParameters"]["useMaxArea"].get<bool>();
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
		"useDistance", useDistance, "useMinLocs", useMinLocs, "useMaxLocs", useMaxLocs,
		"useMinArea", useMinArea, "useMaxArea", useMaxArea, "minLocs", minLocs,
		"maxLocs", maxLocs, "minArea", minArea, "maxArea", maxArea,
		"cutDistance", cutDistance, "inROIs", inROIs));
}

DelaunayTriangulationBasicCommands::DelaunayTriangulationBasicCommands(const DelaunayTriangulationBasicCommands& _o) : poca::core::Command(_o)
{
	m_delaunay = _o.m_delaunay;
}

DelaunayTriangulationBasicCommands::~DelaunayTriangulationBasicCommands()
{
}

void DelaunayTriangulationBasicCommands::execute(poca::core::CommandInfo* _infos)
{
	if (hasCommand(_infos->nameCommand)) {
		loadParameters(*_infos);
	}
	else if (_infos->nameCommand == "createFilteredObjects") {
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_delaunay);
		if (obj == NULL) return;

		size_t minLocs = 3, maxLocs = std::numeric_limits <size_t>::max();
		float minArea = 0.f, maxArea = std::numeric_limits < float >::max(), cutDistance = std::numeric_limits < float >::max();
		bool inROIs = false, useDistance = true, useMinLocs = true, useMaxLocs = false, useMinArea = false, useMaxArea = false;

		if (_infos->nbParameters() == 0) {
			useDistance = hasParameter("objectCreationParameters", "useDistance") ? getParameter<bool>("objectCreationParameters", "useDistance") : true;
			useMinLocs = hasParameter("objectCreationParameters", "useMinLocs") ? getParameter<bool>("objectCreationParameters", "useMinLocs") : true;
			useMaxLocs = hasParameter("objectCreationParameters", "useMaxLocs") ? getParameter<bool>("objectCreationParameters", "useMaxLocs") : false;
			useMinArea = hasParameter("objectCreationParameters", "useMinArea") ? getParameter<bool>("objectCreationParameters", "useMinArea") : false;
			useMaxArea = hasParameter("objectCreationParameters", "useMaxArea") ? getParameter<bool>("objectCreationParameters", "useMaxArea") : false;
			minLocs = hasParameter("objectCreationParameters", "minLocs") ? getParameter<size_t>("objectCreationParameters", "minLocs") : minLocs;
			maxLocs = hasParameter("objectCreationParameters", "maxLocs") ? getParameter<size_t>("objectCreationParameters", "maxLocs") : maxLocs;
			minArea = hasParameter("objectCreationParameters", "minArea") ? getParameter<float>("objectCreationParameters", "minArea") : minArea;
			maxArea = hasParameter("objectCreationParameters", "maxArea") ? getParameter<float>("objectCreationParameters", "maxArea") : maxArea;
			cutDistance = hasParameter("objectCreationParameters", "cutDistance") ? getParameter<float>("objectCreationParameters", "cutDistance") : cutDistance;
			inROIs = hasParameter("objectCreationParameters", "inROIs") ? getParameter<bool>("objectCreationParameters", "inROIs") : inROIs;
		}
		else {
			useDistance = _infos->hasParameter("useDistance") ? _infos->getParameter<bool>("useDistance") : true;
			useMinLocs = _infos->hasParameter("useMinLocs") ? _infos->getParameter<bool>("useMinLocs") : true;
			useMaxLocs = _infos->hasParameter("useMaxLocs") ? _infos->getParameter<bool>("useMaxLocs") : false;
			useMinArea = _infos->hasParameter("useMinArea") ? _infos->getParameter<bool>("useMinArea") : false;
			useMaxArea = _infos->hasParameter("useMaxArea") ? _infos->getParameter<bool>("useMaxArea") : false;
			minLocs = _infos->hasParameter("minLocs") ? _infos->getParameter<size_t>("minLocs") : minLocs;
			maxLocs = _infos->hasParameter("maxLocs") ? _infos->getParameter<size_t>("maxLocs") : maxLocs;
			minArea = _infos->hasParameter("minArea") ? _infos->getParameter<float>("minArea") : minArea;
			maxArea = _infos->hasParameter("maxArea") ? _infos->getParameter<float>("maxArea") : maxArea;
			cutDistance = _infos->hasParameter("cutDistance") ? _infos->getParameter<float>("cutDistance") : cutDistance;
			inROIs = _infos->hasParameter("inROIs") ? _infos->getParameter<bool>("inROIs") : inROIs;

			poca::core::CommandInfo ci(true, "objectCreationParameters",
				"useDistance", useDistance, 
				"useMinLocs", useMinLocs, 
				"useMaxLocs", useMaxLocs,
				"useMinArea", useMinArea, 
				"useMaxArea", useMaxArea, 
				"cutDistance", cutDistance,
				"minLocs", minLocs,
				"maxLocs", maxLocs,
				"minArea", minArea,
				"maxArea", maxArea,
				"inROIs", inROIs);
			loadParameters(ci);
		}
		
		if (!useMinLocs) minLocs = 3;
		if (!useMaxLocs) maxLocs = std::numeric_limits <size_t>::max();
		if (!useMinArea) minArea = 0.f;
		if (!useMaxArea) maxArea = std::numeric_limits < float >::max();
		if (!useDistance) cutDistance = std::numeric_limits < float >::max();

		const std::vector <bool>& selection = m_delaunay->getSelection();
		poca::geometry::ObjectListFactory factory;
		poca::geometry::ObjectListInterface* objects = factory.createObjectListFromDelaunay(obj, selection, cutDistance, minLocs, maxLocs, minArea, maxArea, inROIs);
		if (objects == NULL) {
			QMessageBox msgBox;
			msgBox.setText("No object was created.");
			msgBox.exec();
			return;
		}
		objects->setBoundingBox(m_delaunay->boundingBox());
		DelaunayTriangulationPlugin::m_plugins->addCommands(objects);
		if (!obj->hasBasicComponent("ObjectLists")) {
			poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(objects, *_infos, "DelaunayTriangulationPlugin");
			DelaunayTriangulationPlugin::m_plugins->addCommands(objsList);
			obj->addBasicComponent(objsList);
		}
		else {
			std::string text = _infos->json.dump(4);
			poca::geometry::ObjectLists* objsList = dynamic_cast<poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
			if (objsList)
				objsList->addObjectList(objects, *_infos, "DelaunayTriangulationPlugin");
			std::cout << text << std::endl;
		}
		obj->notify("LoadObjCharacteristicsAllWidgets");
		//obj->addBasicComponent(objects);
		//obj->notify(poca::core::CommandInfo(false, "addCommandToSpecificComponent", "component", (poca::core::BasicComponentInterface*)objects));
	}
	else if (_infos->nameCommand == "saveAsSVG") {
		QString filename = (_infos->getParameter<std::string>("filename")).c_str();
		filename.insert(filename.lastIndexOf("."), "_delaunay");
		saveAsSVG(filename);
	}
	else if (_infos->nameCommand == "applyCutDistance") {
		bool ok;
		float cutDistance = getParameter<float>("objectCreationParameters", "cutDistance");
		std::vector <bool>& selection = m_delaunay->getSelection();
		std::fill(selection.begin(), selection.end(), true);
		double dMaxSqr = cutDistance * cutDistance;
		const std::vector<uint32_t>& triangles = m_delaunay->getTriangles();
		const float* xs = m_delaunay->getXs();
		const float* ys = m_delaunay->getYs();
		if (m_delaunay->dimension() == 3) {
			const poca::core::MyArrayUInt32& neighbors = m_delaunay->getNeighbors();
			const std::vector <uint32_t> indiceTriangles = neighbors.getFirstElements();
			const float* zs = m_delaunay->getZs();
			for (uint32_t n = 0; n < m_delaunay->nbFaces(); n++) {
				for (uint32_t i = indiceTriangles[n]; i < indiceTriangles[n + 1] && selection[n]; i++) {
					uint32_t i1 = triangles[3 * i], i2 = triangles[3 * i + 1], i3 = triangles[3 * i + 2];
					float d0 = poca::geometry::distance3DSqr(xs[i1], ys[i1], zs[i1], xs[i2], ys[i2], zs[i2]);
					float d1 = poca::geometry::distance3DSqr(xs[i2], ys[i2], zs[i2], xs[i3], ys[i3], zs[i3]);
					float d2 = poca::geometry::distance3DSqr(xs[i3], ys[i3], zs[i3], xs[i1], ys[i1], zs[i1]);
					selection[n] = !(d0 > dMaxSqr || d1 > dMaxSqr || d2 > dMaxSqr);
				}
			}
		}
		else {
			for (uint32_t n = 0; n < m_delaunay->nbFaces(); n++) {
				uint32_t i1 = triangles[3 * n], i2 = triangles[3 * n + 1], i3 = triangles[3 * n + 2];
				float d0 = poca::geometry::distanceSqr(xs[i1], ys[i1], xs[i2], ys[i2]);
				float d1 = poca::geometry::distanceSqr(xs[i2], ys[i2], xs[i3], ys[i3]);
				float d2 = poca::geometry::distanceSqr(xs[i3], ys[i3], xs[i1], ys[i1]);
				selection[n] = !(d0 > dMaxSqr || d1 > dMaxSqr || d2 > dMaxSqr);
			}
		}
		m_delaunay->executeCommand(false, "updateFeature");
	}
	else if (_infos->nameCommand == "invertSelection") {
		std::vector <bool>& selection = m_delaunay->getSelection();
		selection.flip();
		m_delaunay->executeCommand(false, "updateFeature");
	}
}

poca::core::CommandInfo DelaunayTriangulationBasicCommands::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "objectCreationParameters" || _nameCommand == "createFilteredObjects") {
		poca::core::CommandInfo ci(false, _nameCommand);
		if (_parameters.contains("useDistance"))
			ci.addParameter("useDistance", _parameters["useDistance"].get<bool>());
		if (_parameters.contains("useMinLocs"))
			ci.addParameter("useMinLocs", _parameters["useMinLocs"].get<bool>());
		if (_parameters.contains("useMaxLocs"))
			ci.addParameter("useMaxLocs", _parameters["useMaxLocs"].get<bool>());
		if (_parameters.contains("useMinArea"))
			ci.addParameter("useMinArea", _parameters["useMinArea"].get<bool>());
		if (_parameters.contains("useMaxArea"))
			ci.addParameter("useMaxArea", _parameters["useMaxArea"].get<bool>());
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
	else if (_nameCommand == "saveAsSVG") {
		if (_parameters.contains("filename")) {
			std::string val = _parameters["filename"].get<std::string>();
			return poca::core::CommandInfo(false, _nameCommand, "filename", val);
		}
	}
	else if (_nameCommand  == "applyCutDistance" || _nameCommand == "invertSelection") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	return poca::core::CommandInfo();
}

poca::core::Command* DelaunayTriangulationBasicCommands::copy()
{
	return new DelaunayTriangulationBasicCommands(*this);
}

void DelaunayTriangulationBasicCommands::saveAsSVG(const QString& _filename) const
{
	poca::core::BoundingBox bbox = m_delaunay->boundingBox();
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
	m_delaunay->generateTriangles(triangles);
	poca::core::Histogram<float>* histogram = dynamic_cast <poca::core::Histogram<float>*>(m_delaunay->getCurrentHistogram());
	const std::vector<float>& values = histogram->getValues();
	const std::vector<bool>& selection = m_delaunay->getSelection();
	float minH = histogram->getMin(), maxH = histogram->getMax(), interH = maxH - minH;

	std::vector <float> featureValues;
	m_delaunay->getFeatureInSelection(featureValues, values, selection, std::numeric_limits <float>::max());

	char col[32];
	poca::core::PaletteInterface* pal = m_delaunay->getPalette();
	for (size_t n = 0; n < triangles.size(); n += 3) {
		if (featureValues[n] == std::numeric_limits <float>::max()) continue;
		float valPal = (featureValues[n] - minH) / interH;
		poca::core::Color4uc c = pal->getColor(valPal);
		unsigned char r = c[0], g = c[1], b = c[2];
		poca::core::getColorStringUC(r, g, b, col);
		/*size_t idx[] = {n, n + 1, n + 2};
		for (size_t i = 0; i < 3; i++) {
			size_t i1 = idx[i], i2 = idx[(i + 1) % 3];
			fs << "<line x1 =\"";
			fs << triangles[i1].x() << "\" y1=\"";
			fs << triangles[i1].y() << "\" x2=\"";
			fs << triangles[i2].x() << "\" y2=\"";
			fs << triangles[i2].y() << "\" stroke=\"" << col << "\" stroke-width=\"1\"/>\n";
		}*/
		fs << "<polygon points =\"";
		fs << triangles[n].x() << ",";
		fs << triangles[n].y() << " ";
		fs << triangles[n + 1].x() << ",";
		fs << triangles[n + 1].y() << " "; 
		fs << triangles[n + 2].x() << ",";
		fs << triangles[n + 2].y() << "\" stroke=\"" << col << "\" stroke-width=\"1\"/>\n";
	}
	fs.close();
}

