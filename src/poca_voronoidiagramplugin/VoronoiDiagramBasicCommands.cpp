/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagramBasicCommands.cpp
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

#include <DesignPatterns/ListDatasetsSingleton.hpp>
#include <DesignPatterns/StateSoftwareSingleton.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <Geometry/ObjectList.hpp>
#include <General/Misc.h>
#include <Interfaces/HistogramInterface.hpp>

#include "VoronoiDiagramBasicCommands.hpp"

VoronoiDiagramBasicCommands::VoronoiDiagramBasicCommands(poca::geometry::VoronoiDiagram* _voro) :poca::core::Command("VoronoiDiagramBasicCommands")
{
	m_voronoi = _voro;

	poca::core::StateSoftwareSingleton* sss = poca::core::StateSoftwareSingleton::instance();
	float cutDistance = std::numeric_limits < float >::max(), minArea = 0.f, maxArea = std::numeric_limits < float >::max();
	size_t minLocs = 3, maxLocs = std::numeric_limits <size_t>::max();
	bool inROIs = false;

	const nlohmann::json& parameters = sss->getParameters();
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
		poca::core::ListDatasetsSingleton* lds = poca::core::ListDatasetsSingleton::instance();
		poca::core::MyObjectInterface* obj = lds->getObject(m_voronoi);
		if (obj == NULL) return;

		size_t minLocs = hasParameter("objectCreationParameters", "minLocs") ? getParameter<size_t>("objectCreationParameters", "minLocs") : 3;
		size_t maxLocs = hasParameter("objectCreationParameters", "maxLocs") ? getParameter<size_t>("objectCreationParameters", "maxLocs") : std::numeric_limits <size_t>::max();
		float minArea = hasParameter("objectCreationParameters", "minArea") ? getParameter<float>("objectCreationParameters", "minArea") : 0.f;
		float maxArea = hasParameter("objectCreationParameters", "maxArea") ? getParameter<float>("objectCreationParameters", "maxArea") : std::numeric_limits < float >::max();
		float cutDistance = hasParameter("objectCreationParameters", "cutDistance") ? getParameter<float>("objectCreationParameters", "cutDistance") : std::numeric_limits < float >::max();
		bool inROIs = hasParameter("objectCreationParameters", "inROIs") ? getParameter<bool>("objectCreationParameters", "inROIs") : false;

		const std::vector <bool>& selection = m_voronoi->getSelection();
		poca::geometry::ObjectListFactory factory;
		poca::geometry::ObjectList* objects = factory.createObjectList(obj, selection, cutDistance, minLocs, maxLocs, minArea, maxArea, inROIs);
		if (objects == NULL) return;
		objects->setBoundingBox(m_voronoi->boundingBox());
		obj->addBasicComponent(objects);
		obj->notify(poca::core::CommandInfo(false, "addCommandToSpecificComponent", "component", (poca::core::BasicComponent*)objects));
	}
	else if (_infos->nameCommand == "densityFactor") {
		float factor = _infos->getParameter<float>("factor");

		poca::core::BoundingBox bbox = m_voronoi->boundingBox();
		float meanD = m_voronoi->averageDensity();
		const std::vector <float>& feature = m_voronoi->getOriginalHistogram("density")->getValues();
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
		poca::core::ListDatasetsSingleton* lds = poca::core::ListDatasetsSingleton::instance();
		poca::core::MyObjectInterface* obj = lds->getObject(m_voronoi);
		poca::core::BasicComponent* dset = obj->getBasicComponent("DetectionSet");
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
		for (auto factor = from; factor <= to; factor += step, n++) {
			execute(&poca::core::CommandInfo(false, "densityFactor", "factor", factor));
			dset->executeCommand(false, "clustersForChallenge", "minNbLocs", minNbLocs, "maxNbLocs", maxNbLocs, "selection", std::string("VoronoiDiagram"), "factor", factor, "currentScreen", n);
		}
	}
}

poca::core::CommandInfo VoronoiDiagramBasicCommands::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "objectCreationParameters") {
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
	else if (_nameCommand == "createFilteredObjects" || _nameCommand == "invertSelection" || _nameCommand == "randomPointOnTheSphere") {
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
	return poca::core::CommandInfo();
}

poca::core::Command* VoronoiDiagramBasicCommands::copy()
{
	return new VoronoiDiagramBasicCommands(*this);
}

