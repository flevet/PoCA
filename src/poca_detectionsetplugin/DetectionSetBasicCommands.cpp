/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DetectionSetBasicCommands.cpp
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

#include <QtWidgets/QMessageBox>
#include <QtCore/QString>
#include <QtCore/QDir>
#include <fstream>
#include <filesystem>

#include <Interfaces/ObjectIndicesFactoryInterface.hpp>
#include <Geometry/ObjectList.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <Geometry/DetectionSet.hpp>
#include <General/Roi.hpp>
#include <DesignPatterns/ListDatasetsSingleton.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <Geometry/ObjectList.hpp>
#include <General/MyData.hpp>
#include <General/Misc.h>

#include "DetectionSetBasicCommands.hpp"

DetectionSetBasicCommands::DetectionSetBasicCommands(poca::geometry::DetectionSet* _dset) :poca::core::Command("DetectionSetBasicCommands")
{
	m_dset = _dset;
}

DetectionSetBasicCommands::DetectionSetBasicCommands(const DetectionSetBasicCommands& _o) : poca::core::Command(_o)
{
	m_dset = _o.m_dset;
}

DetectionSetBasicCommands::~DetectionSetBasicCommands()
{
}

void DetectionSetBasicCommands::execute(poca::core::CommandInfo* _infos)
{
	if (_infos->nameCommand == "selectLocsInROIs") {
		try {
			poca::core::ListDatasetsSingleton* lds = poca::core::ListDatasetsSingleton::instance();
			poca::core::MyObjectInterface* obj = lds->getObject(m_dset);
			if (obj == NULL) return;
			std::vector <poca::core::ROIInterface*> ROIs = obj->getROIs();
			if (ROIs.empty()) return;

			std::vector <bool> selection(m_dset->nbPoints());
			const std::vector <float>& xs = m_dset->getOriginalData("x"), ys = m_dset->getOriginalData("y"), zs = m_dset->dimension() == 3 ? m_dset->getOriginalData("z") : std::vector <float>(xs.size());
			for (size_t i = 0; i < xs.size(); i++) {
				bool inside = false;
				for (size_t n = 0; n < ROIs.size() && !inside; n++)
					inside = ROIs[n]->inside(xs[i], ys[i], zs[i]);
				selection[i] = inside;
			}
			m_dset->setSelection(selection);
			m_dset->executeCommand(false, "updateFeature");
			obj->notifyAll("updateDisplay");
		}
		catch (std::runtime_error const& e) {
			QMessageBox msgBox;
			msgBox.setText(_infos->errorMessageToStdString(e.what()).c_str());
			msgBox.exec();
		}
	}
	else if (_infos->nameCommand == "createObjectsFromHistogram") {
		poca::core::ListDatasetsSingleton* lds = poca::core::ListDatasetsSingleton::instance();
		poca::core::MyObjectInterface* obj = lds->getObject(m_dset);
		if (obj == NULL) return;

		size_t minLocs = _infos->hasParameter("minLocs") ? _infos->getParameter<size_t>("minLocs") : 3;
		size_t maxLocs = _infos->hasParameter("maxLocs") ? _infos->getParameter<size_t>("maxLocs") : std::numeric_limits <size_t>::max();
		float minArea = _infos->hasParameter("minArea") ? _infos->getParameter<float>("minArea") : 0.f;
		float maxArea = _infos->hasParameter("maxArea") ? _infos->getParameter<float>("maxArea") : std::numeric_limits < float >::max();
		float cutDistance = _infos->hasParameter("cutDistance") ? _infos->getParameter<float>("cutDistance") : std::numeric_limits < float >::max();

		const std::vector <bool>& selectionLocs = m_dset->getSelection();
		const std::vector <float>& values = m_dset->getCurrentHistogram()->getValues();
		std::vector <uint32_t> selection(m_dset->nbPoints());
		for (size_t n = 0; n < selection.size(); n++) {
			selection[n] = !selectionLocs[n] ? 0 : (uint32_t)values[n];
		}
		poca::geometry::ObjectListFactory factory;
		poca::geometry::ObjectList* objects = factory.createObjectListAlreadyIdentified(obj, selection, cutDistance, minLocs, maxLocs, minArea, maxArea);
		objects->setBoundingBox(m_dset->boundingBox());
		obj->addBasicComponent(objects);
		obj->notify(poca::core::CommandInfo(false, "addCommandToSpecificComponent", "component", (poca::core::BasicComponent*)objects));
	}
	else if (_infos->nameCommand == "computeDensityWithRadius") {
		float radius = 0.f;
		if (_infos->hasParameter("radius"))
			radius = _infos->getParameter<float>("radius");
		poca::geometry::KdTree_DetectionPoint* tree = m_dset->getKdTree();
		const std::vector <float>& xs = m_dset->getOriginalData("x"), ys = m_dset->getOriginalData("y"), zs = m_dset->dimension() == 3 ? m_dset->getOriginalData("z") : std::vector <float>(xs.size());

		std::vector <float> densities(xs.size());
		double nbs = m_dset->nbPoints();
		unsigned int nbForUpdate = nbs / 100., cpt = 0;
		if (nbForUpdate == 0) nbForUpdate = 1;
		printf("Computing density: %.2f %%", (0. / nbs * 100.));
		double dSqr = radius * radius;
		const double search_radius = static_cast<double>(dSqr);
		std::vector<std::pair<std::size_t, double> > ret_matches;
		nanoflann::SearchParams params;
		for (int n2 = 0; n2 < m_dset->nbPoints(); n2++) {
			const double queryPt[3] = { xs[n2], ys[n2], zs[n2] };
			densities[n2] = tree->radiusSearch(&queryPt[0], search_radius, ret_matches, params);
			if (cpt++ % nbForUpdate == 0) printf("\rComputing density: %.2f %%", ((double)cpt / nbs * 100.));
		}
		printf("\rComputing density: 100 %%\n");
		m_dset->addFeature("density", new poca::core::MyData(densities));

		float mean = poca::core::mean(densities);
		std::transform(densities.begin(), densities.end(), densities.begin(), [mean](auto& c) { return c / mean; });
		float mean2 = poca::core::mean(densities);
		std::cout << "Final mean = " << mean2 << std::endl;
	}
}

poca::core::CommandInfo DetectionSetBasicCommands::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "selectLocsInROIs") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	else if (_nameCommand == "clustersForChallenge") {
		poca::core::CommandInfo ci(false, _nameCommand);
		if (_parameters.contains("minNbLocs"))
			ci.addParameter("minNbLocs", _parameters["minNbLocs"].get<size_t>());
		if (_parameters.contains("maxNbLocs"))
			ci.addParameter("maxNbLocs", _parameters["maxNbLocs"].get<size_t>());
		if (_parameters.contains("selection"))
			ci.addParameter("selection", _parameters["selection"].get<std::string>());
		if (_parameters.contains("currentScreen"))
			ci.addParameter("currentScreen", _parameters["currentScreen"].get<uint32_t>());
		return ci;
	}
	else if (_nameCommand == "createObjectsFromHistogram") {
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
		return ci;
	}
	else if (_nameCommand == "computeDensityWithRadius") {
		poca::core::CommandInfo ci(false, _nameCommand);
		if (_parameters.contains("radius"))
			ci.addParameter("radius", _parameters["radius"].get<float>());
		return ci;
	}
	return poca::core::CommandInfo();
}

poca::core::Command* DetectionSetBasicCommands::copy()
{
	return new DetectionSetBasicCommands(*this);
}

