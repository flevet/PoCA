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
#include <Interfaces/PaletteInterface.hpp>

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
	else if (_infos->nameCommand == "clustersForChallenge") {
		std::string selectedComponent = "DetectionSet";
		uint32_t currentScreen = 1;
		if (_infos->hasParameter("selection"))
			selectedComponent = _infos->getParameter<std::string>("selection");
		size_t minNbLocs = 1, maxNbLocs = std::numeric_limits < size_t >::max();
		float factor = 0.f;
		if (_infos->hasParameter("minNbLocs"))
			minNbLocs = _infos->getParameter<size_t>("minNbLocs");
		if (_infos->hasParameter("maxNbLocs"))
			maxNbLocs = _infos->getParameter<size_t>("maxNbLocs");
		if (_infos->hasParameter("currentScreen"))
			currentScreen = _infos->getParameter<uint32_t>("currentScreen");
		if (_infos->hasParameter("factor"))
			factor = _infos->getParameter<float>("factor");

		poca::core::ListDatasetsSingleton* lds = poca::core::ListDatasetsSingleton::instance();
		poca::core::MyObjectInterface* obj = lds->getObject(m_dset);
		poca::core::BasicComponent* bc = obj->getBasicComponent(selectedComponent);
		if (bc == NULL)
			return;
		const std::vector <bool>& selection = bc->getSelection();

		poca::geometry::ObjectIndicesFactoryInterface* factory = poca::geometry::createObjectIndicesFactory();
		std::vector <uint32_t> indices = factory->createObjects(obj, selection, minNbLocs, maxNbLocs);

		std::vector <float> clusterIndices(indices.size());
		std::transform(indices.begin(), indices.end(), clusterIndices.begin(), [](uint32_t x) { return (float)x; });

		std::map <std::string, poca::core::MyData*>& data = m_dset->getData();
		data["clustersIndices"] = new poca::core::MyData(clusterIndices);

		obj->notifyAll("LoadObjCharacteristicsDetectionSetWidget");
		QString origName = obj->getName().c_str();
		origName = origName.remove(".csv");
		QString directory("e:/Git/ARI-and-IoU-cluster-analysis-evaluation/Challenge/MultiBlinking/");
		directory.append(origName);
		QDir().mkdir(directory);
		QString directoryRes(directory);
		directoryRes.append("/classes");
		QDir().mkdir(directoryRes);

		//Save GT data
		bc = obj->getBasicComponent("DetectionSet");
		poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bc);
		if (dset) {
			const std::vector <float>& xs = dset->getOriginalHistogram("x")->getValues();
			const std::vector <float>& ys = dset->getOriginalHistogram("y")->getValues();
			const std::vector <float>& indices = dset->getOriginalHistogram("index")->getValues();

			std::string filename = directory.toStdString() + "/data.csv";
			std::ofstream fs(filename);
			fs << "x,y,index" << std::endl;
			for (auto n = 0; n < xs.size(); n++)
				fs << xs[n] << "," << ys[n] << "," << ((uint32_t)indices[n]) << std::endl;
			/*for (auto&& [x, y, index] : c9::zip(xs, ys, indices))
				fs << x << "," << y << "," << ((uint32_t)index) << std::endl;*/
			fs.close();
		}

		std::string filename = directoryRes.toStdString() + "/" + std::to_string(currentScreen) + ".csv";
		std::ofstream fs(filename);
		fs << "result" << std::endl;
		for (auto idx : indices)
			fs << idx << std::endl;
		fs.close();

		filename = directory.toStdString() + "/info.csv";
		std::filesystem::path path = filename;
		std::ofstream ofs;
		if (!std::filesystem::exists(path)) {
			ofs.open(filename);
			ofs << "file,factor" << std::endl;
		}
		else {
			ofs.open(filename, std::ofstream::app);
		}
		ofs << (std::to_string(currentScreen) + ".csv") << ',' << std::to_string(factor) << std::endl;
		ofs.close();
		/*QString directory(obj->getDir().c_str());
		if (!directory.endsWith("/"))
			directory.append("/");
		directory.append("results");
		QDir dir(directory);
		if (!dir.exists())
			QDir().mkdir(directory);

		std::string filename = directory.toStdString() + "/" + obj->getName();
		std::ofstream fs(filename);
		fs << "result" << std::endl;
		for(auto idx : indices)
			fs << idx << std::endl;
		fs.close();*/
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
	else if (_infos->nameCommand == "saveAsSVG") {
		QString filename = (_infos->getParameter<std::string>("filename")).c_str();
		filename.insert(filename.lastIndexOf("."), "_detections");
		saveAsSVG(filename);
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
	else if (_nameCommand == "saveAsSVG") {
		if (_parameters.contains("filename")) {
			std::string val = _parameters["filename"].get<std::string>();
			return poca::core::CommandInfo(false, _nameCommand, "filename", val);
		}
	}
	return poca::core::CommandInfo();
}

poca::core::Command* DetectionSetBasicCommands::copy()
{
	return new DetectionSetBasicCommands(*this);
}

void DetectionSetBasicCommands::saveAsSVG(const QString& _filename) const
{
	poca::core::BoundingBox bbox = m_dset->boundingBox();
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

	const std::vector <float>& xs = m_dset->getData("x");
	const std::vector <float>& ys = m_dset->getData("y");
	const std::vector <float>& zs = m_dset->hasData("z") ? m_dset->getData("z") : std::vector <float>(xs.size(), 0.f);
	poca::core::HistogramInterface* histogram = m_dset->getCurrentHistogram();
	const std::vector<float>& values = histogram->getValues();
	const std::vector<bool>& selection = m_dset->getSelection();
	float minH = histogram->getMin(), maxH = histogram->getMax(), interH = maxH - minH;

	char col[32];
	poca::core::PaletteInterface* pal = m_dset->getPalette();
	for (size_t n = 0; n < xs.size(); n++) {
		if (!selection[n]) continue;
		float valPal = (values[n] - minH) / interH;
		poca::core::Color4uc c = pal->getColor(valPal);
		unsigned char r = c[0], g = c[1], b = c[2];
		poca::core::getColorStringUC(r, g, b, col);
		fs << "<circle fill=\"" << col << "\" cx =\"";
		fs << xs[n] << "\" cy=\"";
		fs << ys[n] << "\" cz=\"";
		fs << zs[n] << "\" r=\"1\"/>\n";
	}
	fs.close();
}