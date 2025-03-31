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
#include <random>
#include <set>

#include <Interfaces/ObjectIndicesFactoryInterface.hpp>
#include <Interfaces/ObjectListInterface.hpp>
#include <Geometry/ObjectLists.hpp>
#include <General/Histogram.hpp>
#include <Geometry/DetectionSet.hpp>
#include <General/Roi.hpp>
#include <General/Engine.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <General/MyData.hpp>
#include <General/Misc.h>
#include <Interfaces/PaletteInterface.hpp>
#include <General/PluginList.hpp>
#include <Objects/MyObject.hpp>
#include <OpenGL/Camera.hpp>

#include "DetectionSetBasicCommands.hpp"
#include "DetectionSetPlugin.hpp"

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
			 poca::core::Engine* engine = poca::core::Engine::instance();
			poca::core::MyObjectInterface* obj = engine->getObject(m_dset);
			if (obj == NULL) return;
			std::vector <poca::core::ROIInterface*> ROIs = obj->getROIs();
			if (ROIs.empty()) return;

			std::vector <bool> selection(m_dset->nbPoints());
			const std::vector <float>& xs = m_dset->getMyData("x")->getData<float>(), ys = m_dset->getMyData("y")->getData<float>(), zs = m_dset->dimension() == 3 ? m_dset->getMyData("z")->getData<float>() : std::vector <float>(xs.size());
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
		poca::core::MyObjectInterface* obj = NULL;
		if (_infos->hasParameter("minNbLocs"))
			minNbLocs = _infos->getParameter<size_t>("minNbLocs");
		if (_infos->hasParameter("maxNbLocs"))
			maxNbLocs = _infos->getParameter<size_t>("maxNbLocs");
		if (_infos->hasParameter("currentScreen"))
			currentScreen = _infos->getParameter<uint32_t>("currentScreen");
		if (_infos->hasParameter("factor"))
			factor = _infos->getParameter<float>("factor");
		if (_infos->hasParameter("object"))
			obj = _infos->getParameterPtr<poca::core::MyObjectInterface>("object");


		 poca::core::Engine* engine = poca::core::Engine::instance();
		if(obj == NULL)
			obj = engine->getObject(m_dset);
		poca::core::BasicComponentInterface* bc = obj->getBasicComponent(selectedComponent);
		if (bc == NULL)
			return;
		const std::vector <bool>& selection = bc->getSelection();
		poca::geometry::ObjectIndicesFactoryInterface* factory = poca::geometry::createObjectIndicesFactory();
		std::vector <uint32_t> indices = factory->createObjects(obj, selection, minNbLocs, maxNbLocs);
		/*std::vector <float> clusterIndices(indices.size());
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
		ofs.close();*/
	}
	else if (_infos->nameCommand == "createObjectsFromHistogram") {
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_dset);
		if (obj == NULL) return;

		size_t minLocs = _infos->hasParameter("minLocs") ? _infos->getParameter<size_t>("minLocs") : 3;
		size_t maxLocs = _infos->hasParameter("maxLocs") ? _infos->getParameter<size_t>("maxLocs") : std::numeric_limits <size_t>::max();
		float minArea = _infos->hasParameter("minArea") ? _infos->getParameter<float>("minArea") : 0.f;
		float maxArea = _infos->hasParameter("maxArea") ? _infos->getParameter<float>("maxArea") : std::numeric_limits < float >::max();
		float cutDistance = _infos->hasParameter("cutDistance") ? _infos->getParameter<float>("cutDistance") : std::numeric_limits < float >::max();

		const std::vector <bool>& selectionLocs = m_dset->getSelection();
		poca::core::Histogram<float>* histogram = dynamic_cast<poca::core::Histogram<float>*>(m_dset->getCurrentHistogram());
		const std::vector <float>& values = histogram->getValues();
		std::vector <uint32_t> selection(m_dset->nbPoints());
		for (size_t n = 0; n < selection.size(); n++) {
			selection[n] = !selectionLocs[n] ? 0 : (uint32_t)values[n];
		}
		poca::geometry::ObjectListFactory factory;
		poca::geometry::ObjectListInterface* objects = factory.createObjectListAlreadyIdentified(obj, selection, cutDistance, minLocs, maxLocs, minArea, maxArea);
		objects->setBoundingBox(m_dset->boundingBox());
		obj->addBasicComponent(objects);
		obj->notify(poca::core::CommandInfo(false, "addCommandToSpecificComponent", "component", (poca::core::BasicComponentInterface*)objects));
	}
	else if (_infos->nameCommand == "computeDensityWithRadius") {
		float radius = 0.f;
		if (_infos->hasParameter("radius"))
			radius = _infos->getParameter<float>("radius");
		poca::geometry::KdTree_DetectionPoint* tree = m_dset->getKdTree();
		const std::vector <float>& xs = m_dset->getMyData("x")->getData<float>(), ys = m_dset->getMyData("y")->getData<float>(), zs = m_dset->dimension() == 3 ? m_dset->getMyData("z")->getData<float>() : std::vector <float>(xs.size());

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
		m_dset->addFeature("density", poca::core::generateDataWithLog(densities));
		float mean = poca::core::mean(densities);
		std::transform(densities.begin(), densities.end(), densities.begin(), [mean](auto& c) { return c / mean; });
		float mean2 = poca::core::mean(densities);
		std::cout << "Final mean = " << mean2 << std::endl;

		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_dset);
		obj->notify("LoadObjCharacteristicsDetectionSetWidget");
}
	else if (_infos->nameCommand == "saveAsSVG") {
		QString filename = (_infos->getParameter<std::string>("filename")).c_str();
		filename.insert(filename.lastIndexOf("."), "_detections");
		saveAsSVG(filename);
	}
	else if (_infos->nameCommand == "saveForGNN") {
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_dset);
		if (obj == NULL) return;
		const std::vector <float>& xs = m_dset->getMyData("x")->getData<float>(), ys = m_dset->getMyData("y")->getData<float>(), zs = m_dset->dimension() == 3 ? m_dset->getMyData("z")->getData<float>() : std::vector <float>(xs.size());
		QString dir = obj->getDir().c_str();
		if (!dir.endsWith("/"))
			dir.append("/");
		QString name = obj->getName().c_str();
		name = dir + name.replace("csv", "txt");

		std::ofstream fs(name.toStdString().data());
		if (!fs) {
			std::cout << "System failed to open " << name.toStdString().data() << std::endl;
			return;
		}
		else
			std::cout << "Saving detections in file " << name.toStdString().data() << std::endl;

		/*QString dirAnnot = dir;
		dirAnnot.append("Annotations");
		if(!QDir(dirAnnot).exists())
			QDir().mkdir(dirAnnot);
		dirAnnot.append("/");*/
		//std::ofstream fsn(QString(dirAnnot + "noise_01.txt").toStdString().data());
		//std::ofstream fsr(QString(dirAnnot + "ring_01.txt").toStdString().data());

		const poca::core::BoundingBox& bbox = m_dset->boundingBox();
		float w = bbox[3] - bbox[0], h = bbox[4] - bbox[1], d = bbox[5] - bbox[2];
		float divisor = 100.f;
		for (auto n = 0; n < xs.size(); n++) {
			float x = (xs[n] - bbox[0] - w / 2.f) / divisor;
			float y = (ys[n] - bbox[1] - h / 2.f) / divisor;
			float z = (zs[n] - bbox[2] - d / 2.f) / divisor;
			fs << x << " " << y << " " << z << std::endl;
			/*if (n < 10)
				fsn << x << " " << y << " " << z << std::endl;
			else
				fsr << x << " " << y << " " << z << std::endl;*/
		}
		fs.close();

		//save infos of rescaling
		name = obj->getName().c_str();
		name = dir + name.replace(".csv", "_scale_info_gnn.txt");
		std::ofstream fs2(name.toStdString().data());
		if (!fs2) {
			std::cout << "System failed to open " << name.toStdString().data() << std::endl;
			return;
		}
		else
			std::cout << "Saving gnn infos in " << name.toStdString().data() << std::endl;
		fs2 << "bbox" << std::endl << bbox[0] << " " << bbox[1] << " " << bbox[2] << " " << bbox[3] << " " << bbox[4] << " " << bbox[5] << " " << std::endl << "divisor" << std::endl << divisor << std::endl;
		fs2.close();
		//fsn.close();
		//fsr.close();
	}
	else if (_infos->nameCommand == "rescaleFromGNN") {
		QString filename = (_infos->getParameter<std::string>("filename")).c_str();
		rescaleFromGNN(filename);
		poca::core::CommandInfo ci = poca::core::CommandInfo(false, "regenerateDisplay");
		m_dset->executeCommand(&ci);
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_dset);
		obj->notifyAll("updateDisplay");
	}
	else if (_infos->nameCommand == "rotatePointCloudXY") {
		float angle = 0.f;
		if (_infos->hasParameter("angle"))
			angle = _infos->getParameter<float>("angle");
		rotateLocsXY(angle);
		m_dset->computeBBoxFromPoints();
		poca::core::CommandInfo ci = poca::core::CommandInfo(false, "regenerateDisplay");
		m_dset->executeCommand(&ci);
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_dset);
		obj->notifyAll("updateDisplay");
	}
	else if (_infos->nameCommand == "createObjectsOnLabels") {
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_dset);
		if (obj == NULL) return;

		if (!m_dset->hasData("label")) {
			QMessageBox msgBox;
			msgBox.setText("Creating objects from labels requires localizations to have a feature called 'id'.");
			msgBox.exec();
			return;
		}

		const std::vector <float>& ids = m_dset->getMyData("label")->getData<float>();
		std::vector<uint32_t> labels = std::vector<uint32_t>(ids.begin(), ids.end());
		for (auto n = 0; n < labels.size(); n++)
			labels[n] = labels[n] == 0 ? std::numeric_limits<uint32_t>::max() : labels[n] - 1;

		poca::geometry::ObjectListFactory factory;
		poca::geometry::ObjectListInterface* objects = factory.createObjectListAlreadyIdentified(obj, labels);

		if (objects == NULL) return;
		objects->setBoundingBox(m_dset->boundingBox());
		DetectionSetPlugin::m_plugins->addCommands(objects);
		if (!obj->hasBasicComponent("ObjectLists")) {
			poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(objects, *_infos, "DetectionSetPlugin");
			DetectionSetPlugin::m_plugins->addCommands(objsList);
			obj->addBasicComponent(objsList);
		}
		else {
			std::string text = _infos->json.dump(4);
			poca::geometry::ObjectLists* objsList = dynamic_cast<poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
			if (objsList)
				objsList->addObjectList(objects, *_infos, "DetectionSetPlugin");
			std::cout << text << std::endl;
		}
		obj->notify("LoadObjCharacteristicsAllWidgets");
	}
	else if (_infos->nameCommand == "saveLocalizations") {
		std::string path = _infos->getParameter<std::string>("path");
		QString filename;

		if (path.empty()) {
			 poca::core::Engine* engine = poca::core::Engine::instance();
			poca::core::MyObjectInterface* obj = engine->getObject(m_dset);
			if (obj == NULL) return;

			std::string dir = obj->getDir(), name = obj->getName();
			filename = QString(dir.c_str());
			if (!filename.endsWith('/'))
				filename.append('/');
			filename.append(name.c_str());
		}
		else
			filename = QString(path.c_str());

		std::ofstream fs(filename.toStdString().data());
		if (!fs) {
			std::cout << "System failed to open " << filename.toLatin1().data() << std::endl;
			return;
		}
		else
			std::cout << "Saving detections in file " << filename.toLatin1().data() << std::endl;

		m_dset->saveDetections(fs);
		fs.close();
	}
	else if (_infos->nameCommand == "keepPercentageOfLocalizations") {
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_dset);
		if (obj == NULL) return;

		float percent = 1.f;
		if (_infos->hasParameter("percent"))
			percent = _infos->getParameter<float>("percent");
		else
			return;

		poca::geometry::DetectionSet* dset = keepPercentageOfLocalizations(percent);
		const std::string& dir = obj->getDir(), name = obj->getName();
		QString newName(name.c_str()), addition = QString("_kept_") + QString::number(percent) + QString("_locs");
		int index = newName.lastIndexOf(".");
		newName.insert(index, addition);
		poca::core::MyObject* wobj = new poca::core::MyObject();
		wobj->setDir(dir.c_str());
		wobj->setName(newName.toLatin1().data());
		wobj->addBasicComponent(dset);
		wobj->setDimension(dset->dimension());
		_infos->addParameter("object", static_cast <poca::core::MyObjectInterface*>(wobj));
	}
}

poca::core::CommandInfo DetectionSetBasicCommands::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	std::cout << __LINE__ << std::endl;
	if (_nameCommand == "selectLocsInROIs" || _nameCommand == "saveForGNN" || _nameCommand == "createObjectsOnLabels") {
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
	else if (_nameCommand == "rotatePointCloudXY") {
		poca::core::CommandInfo ci(false, _nameCommand);
		if (_parameters.contains("angle"))
			ci.addParameter("angle", _parameters["angle"].get<float>());
		return ci;
	}
	else if (_nameCommand == "saveAsSVG" || _nameCommand == "rescaleFromGNN") {
		if (_parameters.contains("filename")) {
			std::string val = _parameters["filename"].get<std::string>();
			return poca::core::CommandInfo(false, _nameCommand, "filename", val);
		}
	}
	else if (_nameCommand == "saveLocalizations") {
		std::string path;
		if (_parameters.contains("path"))
			path = _parameters["path"].get<std::string>();
		return poca::core::CommandInfo(false, _nameCommand, "path", path);
	}
	else if (_nameCommand == "keepPercentageOfLocalizations") {
		poca::core::CommandInfo ci(false, _nameCommand);
		if (_parameters.contains("percent"))
			ci.addParameter("percent", _parameters["percent"].get<float>());
		return ci;
	}
	return poca::core::CommandInfo();
}

poca::core::Command* DetectionSetBasicCommands::copy()
{
	return new DetectionSetBasicCommands(*this);
}

void DetectionSetBasicCommands::saveAsSVG(const QString& _filename) const
{
	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::opengl::CameraInterface* cam = engine->getCamera(m_dset);
	poca::core::Vec3mf direction = poca::core::Vec3mf(cam->getEye().x, cam->getEye().y, cam->getEye().z);

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

	const std::vector <float>& xs = m_dset->getMyData("x")->getData<float>();
	const std::vector <float>& ys = m_dset->getMyData("y")->getData<float>();
	const std::vector <float>& zs = m_dset->hasData("z") ? m_dset->getMyData("z")->getData<float>() : std::vector <float>(xs.size(), 0.f);
	poca::core::Histogram<float>* histogram = dynamic_cast<poca::core::Histogram<float>*>(m_dset->getCurrentHistogram());
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
		glm::vec2 p1 = cam->worldToScreenCoordinates(glm::vec3(xs[n], ys[n], zs[n]));
		fs << "<circle fill=\"" << col << "\" cx =\"";
		fs << p1[0] << "\" cy=\"";
		fs << p1[1] << "\" cz=\"";
		fs << p1[2] << "\" r=\"1\"/>\n";
	}
	fs.close();
}

void DetectionSetBasicCommands::rotateLocsXY(const float _angle)
{
	std::vector <float>& xs = m_dset->getMyData("x")->getData<float>();
	std::vector <float>& ys = m_dset->getMyData("y")->getData<float>();
	const poca::core::BoundingBox& bbox = m_dset->boundingBox();
	float xc = bbox[0] + ((bbox[3] - bbox[0]) / 2), yc = bbox[1] + ((bbox[4] - bbox[1]) / 2);
	for (auto n = 0; n < xs.size(); n++) {
		float currentAngle;
		float d;
		currentAngle = atan2(ys[n] - yc, xs[n] - xc);
		currentAngle += _angle;
		d = poca::geometry::distance(xc, yc, xs[n], ys[n]);
		xs[n] = xc + (d * cos(currentAngle));
		ys[n] = yc + (d * sin(currentAngle));
	}
}

void DetectionSetBasicCommands::rescaleFromGNN(const QString& _filename)
{
	std::ifstream fs(_filename.toStdString().data());
	if (!fs) {
		std::cout << "System failed to open " << _filename.toStdString().data() << std::endl;
		return;
	}
	else
		std::cout << "Opening file " << _filename.toStdString().data() << std::endl;
	std::string s;
	std::getline(fs, s);
	std::getline(fs, s);
	std::istringstream is2(s);
	float x0, y0, z0, x1, y1, z1, divisor;
	is2 >> x0 >> y0 >> z0 >> x1 >> y1 >> z1;
	std::getline(fs, s);
	std::getline(fs, s);
	is2 = std::istringstream(s);
	is2 >> divisor;
	fs.close();

	if (m_dset->dimension() == 2) {
		std::vector <float>& xs = m_dset->getMyData("x")->getData<float>();
		std::vector <float>& ys = m_dset->getMyData("y")->getData<float>();
		float w = x1 - x0, h = y1 - y0;
		for (auto n = 0; n < xs.size(); n++) {
			xs[n] = (xs[n] * divisor) + x0 + (w / 2.f);
			ys[n] = (ys[n] * divisor) + y0 + (h / 2.f);
		}
	}
	else {
		std::vector <float>& xs = m_dset->getMyData("x")->getData<float>();
		std::vector <float>& ys = m_dset->getMyData("y")->getData<float>();
		std::vector <float>& zs = m_dset->getMyData("y")->getData<float>();
		float w = x1 - x0, h = y1 - y0, d = z1 - z0;
		for (auto n = 0; n < xs.size(); n++) {
			xs[n] = (xs[n] * divisor) + x0 + (w / 2.f);
			ys[n] = (ys[n] * divisor) + y0 + (h / 2.f);
			zs[n] = (zs[n] * divisor) + z0 + (d / 2.f);
		}
	}
}

poca::geometry::DetectionSet* DetectionSetBasicCommands::keepPercentageOfLocalizations(const float _percent) const
{
	uint32_t nb_Keept = (uint32_t)((float)m_dset->nbElements() * _percent);
	std::set <uint32_t> unique_selection;
	std::cout << "Targetting " << nb_Keept << " localizations" << std::endl;

	// Initialize the random_device
	std::random_device rd;
	// Seed the engine
	std::mt19937_64 generator(rd());
	// Specify the range of numbers to generate, in this case [min, max]
	std::uniform_int_distribution<uint32_t> dist{ 0, (uint32_t)m_dset->nbElements() };

	/*while (unique_selection.size() < nb_Keept) {
		float tmp = (float)rand() / (float)RAND_MAX;
		uint32_t index = (uint32_t)(tmp * (float)m_dset->nbElements());
		unique_selection.insert(index);
		printf("\rLocs: %u on %u / selected %u - %.9f", unique_selection.size(), nb_Keept, index, tmp);
	}*/
	// Generate the random numbers
	while (unique_selection.size() != nb_Keept)
	{
		auto val = dist(generator);
		unique_selection.insert(val);
		printf("Locs: %u on %u / %u\n", unique_selection.size(), nb_Keept, val);
	}
	std::vector<uint32_t> selection(unique_selection.begin(), unique_selection.end());
	poca::geometry::DetectionSet* dset = m_dset->copySelection(selection);
	return dset;
}