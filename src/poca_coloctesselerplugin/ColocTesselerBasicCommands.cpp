/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ColocTesselerBasicCommands.cpp
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

#include <QtCore/QString>

#include <ctime>
#include <fstream>

#include <DesignPatterns/ListDatasetsSingleton.hpp>
#include <DesignPatterns/StateSoftwareSingleton.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <Geometry/ObjectList.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <General/Misc.h>
#include <General/Roi.hpp>

#include "ColocTesselerBasicCommands.hpp"

ColocTesselerBasicCommands::ColocTesselerBasicCommands(ColocTesseler* _tess) :poca::core::Command("ColocTesselerBasicCommands")
{
	m_colocTesseler = _tess;

	for (size_t n = 0; n < 2; n++) {
		poca::geometry::VoronoiDiagram* voronoi = m_colocTesseler->voronoiAt(n);
		m_borderLocs[n].resize(voronoi->nbElements());
		poca::geometry::DelaunayTriangulationInterface* delau = voronoi->getDelaunay();
		m_trianglesSelectedForCorrection->resize(delau->nbElements());
		m_roiIndexPerLocs[n].resize(voronoi->nbElements());
		m_sortedIndexesInROIs[n].resize(voronoi->nbElements());
	}

	poca::core::StateSoftwareSingleton* sss = poca::core::StateSoftwareSingleton::instance();
	const nlohmann::json& parameters = sss->getParameters();
	addCommandInfo(poca::core::CommandInfo(false, "correction", true));
	addCommandInfo(poca::core::CommandInfo(false, "inROIs", false));
	addCommandInfo(poca::core::CommandInfo(false, "threshold", "color1", 1.f, "color2", 1.f));
	if (parameters.contains(name())) {
		nlohmann::json param = parameters[name()];
		if(param.contains("correction"))
			loadParameters(poca::core::CommandInfo(false, "correction", param["correction"].get<bool>()));
		if (param.contains("inROIs"))
			loadParameters(poca::core::CommandInfo(false, "inROIs", param["inROIs"].get<bool>()));
		if(param.contains("threshold") && param["threshold"].contains("color1") && param["threshold"].contains("color2"))
			loadParameters(poca::core::CommandInfo(false, "threshold", "color1", param["threshold"]["color1"].get<float>(), "color2", param["threshold"]["color2"].get<float>()));
	}
}

ColocTesselerBasicCommands::ColocTesselerBasicCommands(const ColocTesselerBasicCommands& _o) : poca::core::Command(_o)
{
	m_colocTesseler = _o.m_colocTesseler;
}

ColocTesselerBasicCommands::~ColocTesselerBasicCommands()
{
}

void ColocTesselerBasicCommands::execute(poca::core::CommandInfo* _infos)
{
	if (hasCommand(_infos->nameCommand)) {
		loadParameters(*_infos);
	}
	else if (_infos->nameCommand == "computeCoefficients") {
		bool ok, correction = true;
		correction = getParameter<bool>("correction");
		bool inROIs = getParameter<bool>("inROIs");

		poca::core::ListDatasetsSingleton* lds = poca::core::ListDatasetsSingleton::instance();
		poca::core::MyObjectInterface* obj = lds->getObject(m_colocTesseler);
		const std::vector <poca::core::ROIInterface*>& ROIs = inROIs ? obj->getROIs() : std::vector <poca::core::ROIInterface*>();

		if (ROIs.empty()) {
			for (auto n = 0; n < 2; n++) {
				std::fill(m_roiIndexPerLocs[n].begin(), m_roiIndexPerLocs[n].end(), 1);
				std::iota(m_sortedIndexesInROIs[n].begin(), m_sortedIndexesInROIs[n].end(), 0);
			}
		}
		else {
			for (auto n = 0; n < 2; n++) {
				std::fill(m_roiIndexPerLocs[n].begin(), m_roiIndexPerLocs[n].end(), 0);
				poca::geometry::VoronoiDiagram* voronoi = m_colocTesseler->voronoiAt(n);
				const float* xs = voronoi->getXs(), * ys = voronoi->getYs(), * zs = voronoi->getZs();
				for (auto curLoc = 0; curLoc < voronoi->nbElements(); curLoc++) {
					bool found = false;
					m_roiIndexPerLocs[n][curLoc] = 0;
					for (auto curROI = 0; curROI < ROIs.size() && !found; curROI++) {
						found = ROIs[curROI]->inside(xs[curLoc], ys[curLoc], zs != NULL ? zs[curLoc] : 0.f);
						if (found)
							m_roiIndexPerLocs[n][curLoc] = curROI + 1;
					}
				}
				poca::core::sort_indexes(m_roiIndexPerLocs[n], m_sortedIndexesInROIs[n]);
			}
		}

		if (correction)
			computeCorrection();
		classifyLocalizations();
		computeSpearmanRankCorrelation();
		computeMandersCoefficients();

		_infos->addParameters("spearmanColor1", m_spearmans[0],
			"spearmanColor2", m_spearmans[1],
			"mandersColor1", m_manders[0],
			"mandersColor1", m_manders[1],
			"spearman2Color1", m_spearmans2[0],
			"spearman2Color2", m_spearmans2[1],
			"manders2Color1", m_manders2[0],
			"manders2Color2", m_manders2[1]);
	}
	else if (_infos->nameCommand == "savePairDensities") {
		std::string tmp = _infos->getParameter<std::string>("filename");
		QString filename = QString(tmp.c_str());
		for (size_t n = 0; n < 2; n++) {
			QString name(filename);
			int index = name.lastIndexOf(".");
			if (index != -1)
				name = name.left(index - 1);
			name.append(QString("_color0%1.txt").arg(n + 1));

			std::ofstream fs(name.toLatin1().data());
			fs << "id\tdensity color 1\tdensity color 2" << std::endl;

			poca::core::Scatterplot* scatter = m_colocTesseler->scattergramPtrAt(n);
			const std::vector <poca::core::Vec2mf>& values = scatter->getPoints();
			size_t idx = 0;
			for (const poca::core::Vec2mf& val : values) {
				fs << idx++ << "\t" << val[0] << "\t" << val[1] << std::endl;
			}

			fs.close();
		}
	}
}

poca::core::CommandInfo ColocTesselerBasicCommands::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "threshold") {
		float color1, color2;
		bool complete = _parameters.contains("color1");
		if (complete)
			color1 = _parameters["color1"].get<float>();
		complete &= _parameters.contains("color2");
		if (complete) {
			color2 = _parameters["color2"].get<float>();
			return poca::core::CommandInfo(false, _nameCommand, "color1", color1, "color2", color2);
		}
	}
	else if (_nameCommand == "correction") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "computeCoefficients") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	else if (_nameCommand == "savePairDensities") {
		if (_parameters.contains("filename")) {
			std::string val = _parameters["filename"].get<std::string>();
			return poca::core::CommandInfo(false, _nameCommand, "filename", val);
		}
	}
	return poca::core::CommandInfo();
}

poca::core::Command* ColocTesselerBasicCommands::copy()
{
	return new ColocTesselerBasicCommands(*this);
}

void ColocTesselerBasicCommands::computeCorrection()
{
	std::vector<bool> selectedLocs[2];

	bool ok;
	float thresh[2], thresh_high = FLT_MAX;
	thresh[0] = getParameter<float>("threshold", "color1");
	thresh[1] = getParameter<float>("threshold", "color2");

	for (unsigned int current = 0; current < 2; current++) {
		unsigned int other = (current + 1) % 2;
		const poca::core::Scatterplot& vertices = m_colocTesseler->scattergramAt(current);
		double tCurrent = (current == 0) ? thresh[0] : thresh[1], tOther = (current == 0) ? thresh[1] : thresh[0];

		poca::geometry::VoronoiDiagram* currentVoronoi = m_colocTesseler->voronoiAt(current);
		size_t curNbPoints = currentVoronoi->nbElements(), countSelected = 0;
		selectedLocs[current].resize(curNbPoints, false);
		for (unsigned int n = 0; n < curNbPoints; n++) {
			selectedLocs[current][n] = vertices[n][current] > tCurrent;
			if (selectedLocs[current][n]) countSelected++;
		}

		std::cout << "# locs selected = " << countSelected << std::endl;
		//Determination of the border locs
		poca::core::MyArrayUInt32 neighbors = currentVoronoi->getNeighbors();
		const std::vector <uint32_t>& firsts = neighbors.getFirstElements(), & neighs = neighbors.getData();
		bool borderLoc = true;
		for (uint32_t n = 0; n < neighbors.nbElements(); n++) {
			borderLoc = false;
			if (!selectedLocs[current][n]) continue;
			for (uint32_t i = firsts[n]; i < firsts[n + 1] && !borderLoc; i++) {
				uint32_t index = neighs[i];
				borderLoc = (index != std::numeric_limits <uint32_t>::max()) ? !selectedLocs[current][index] : true;
			}
			m_borderLocs[current][n] = borderLoc;
		}

		poca::geometry::DelaunayTriangulationInterface* delau = currentVoronoi->getDelaunay();
		const std::vector <float>& values = delau->dimension() == 3 ? delau->getOriginalHistogram("volume")->getValues() : delau->getOriginalHistogram("area")->getValues();
		const std::vector<uint32_t>& triangles = delau->getTriangles();
		std::fill(m_trianglesSelectedForCorrection[current].begin(), m_trianglesSelectedForCorrection[current].end(), false);

		float averageNbLocs = currentVoronoi->averageMeanNbLocs();
		delau->generateFaceSelectionFromLocSelection(selectedLocs[current], m_trianglesSelectedForCorrection[current]);
		std::vector <float> distribution;
		for (size_t n = 0; n < delau->nbFaces(); n++)
			if (m_trianglesSelectedForCorrection[current][n])
				distribution.push_back(values[n] / averageNbLocs);

		//Determine the cutoff for removing outliers
		if (distribution.empty()) continue;
		std::sort(distribution.begin(), distribution.end());
		float q25 = distribution[int(distribution.size() * 0.25)], q75 = distribution[int(distribution.size() * 0.75)];
		float cutoff = (q75 - q25) * 1.5, high = q75 + cutoff;
		std::cout << "*** Distrib size = " << distribution.size() << ", percentile : 25th = " << q25 << ", 75th = " << q75 << ", high = " << high << std::endl;

		thresh_high = high < thresh_high ? high : thresh_high;
	}

	for (unsigned int current = 0; current < 2; current++) {
		poca::geometry::VoronoiDiagram* currentVoronoi = m_colocTesseler->voronoiAt(current);
		poca::geometry::DelaunayTriangulationInterface* delau = currentVoronoi->getDelaunay();
		const std::vector <float>& values = delau->dimension() == 3 ? delau->getOriginalHistogram("volume")->getValues() : delau->getOriginalHistogram("area")->getValues();
		const std::vector<uint32_t>& triangles = delau->getTriangles();

		float averageNbLocs = currentVoronoi->averageMeanNbLocs();

		//Corection of the triangle sets
		if (delau->dimension() == 2) {
			const std::vector<uint32_t>& triangles = delau->getTriangles();
			for (size_t n = 0; n < delau->nbFaces(); n++) {
				uint32_t i1 = triangles[n * 3], i2 = triangles[n * 3 + 1], i3 = triangles[n * 3 + 2];
				if (m_borderLocs[current][i1] || m_borderLocs[current][i2] || m_borderLocs[current][i3])
					m_trianglesSelectedForCorrection[current][n] = m_trianglesSelectedForCorrection[current][n] && ((values[n] / averageNbLocs) < thresh_high);
			}
		}
		else if (delau->dimension() == 3) {
			const std::vector <uint32_t>& indices = delau->getNeighbors().getFirstElements();
			for (size_t n = 0; n < delau->nbFaces(); n++) {
				uint32_t index = indices[n];
				uint32_t i1 = triangles[3 * index],
					i2 = triangles[3 * index + 3 * 1],
					i3 = triangles[3 * index + 3 * 2],
					i4 = triangles[3 * index + 3 * 3];
				if(m_borderLocs[current][i1] || m_borderLocs[current][i2] || 
					m_borderLocs[current][i3] || m_borderLocs[current][i4])
				m_trianglesSelectedForCorrection[current][n] = (values[n] / averageNbLocs) < thresh_high;
			}
		}
		auto nbT = std::count(m_trianglesSelectedForCorrection[current].begin(), m_trianglesSelectedForCorrection[current].end(), true);
		std::cout << "Nb true for color " << current << " = " << nbT << " / " << m_trianglesSelectedForCorrection[current].size() << std::endl;
		nbT = std::count(m_borderLocs[current].begin(), m_borderLocs[current].end(), true);
		std::cout << "Nb true for border " << current << " = " << nbT << " / " << m_borderLocs[current].size() << std::endl;
	}
}

void ColocTesselerBasicCommands::classifyLocalizations()
{
	bool ok, correction = true;
	float thresh[] = { 1.f, 1.f };
	thresh[0] = getParameter<float>("threshold", "color1");
	thresh[1] = getParameter<float>("threshold", "color2");
	correction = getParameter<bool>("correction");


	for (size_t current = 0; current < 2; current++) {
		size_t other = (current + 1) % 2;
		const poca::core::Scatterplot& vertices = m_colocTesseler->scattergramAt(current);
		const std::vector <uint32_t>& indexTrisOtherColor = m_colocTesseler->indexTriOtherColorAt(current);
		std::vector <unsigned char>& classesLocs = m_colocTesseler->classesLocsAt(current);
		size_t nbBack = 0, nbHD = 0, nbColoc = 0;

		for (size_t n = 0; n < classesLocs.size(); n++) {
			if (m_roiIndexPerLocs[current][n] == 0) {
				classesLocs[n] = ColocTesseler::BACKGROUND;
				continue;
			}
			float dCurrent = vertices[n][current], dOther = vertices[n][other];
			if (dCurrent > thresh[current]) {
				if (dOther > thresh[other]) {
					if (!correction)
						classesLocs[n] = ColocTesseler::COLOC;
					else {
						int indexTriangleInOtherColor = indexTrisOtherColor[n];
						if (indexTriangleInOtherColor != std::numeric_limits<std::uint32_t>::max() && m_trianglesSelectedForCorrection[other][indexTriangleInOtherColor])
							classesLocs[n] = ColocTesseler::COLOC;
						else
							classesLocs[n] = ColocTesseler::HIGH_DENSITY;
					}
				}
				else
					classesLocs[n] = ColocTesseler::HIGH_DENSITY;
			}
			else
				classesLocs[n] = ColocTesseler::BACKGROUND;
			
			uint32_t val = (uint32_t)classesLocs[n];
			switch (val) {
			case 1:
				nbBack++;
				break;
			case 2:
				nbHD++;
				break;
			case 3:
				nbColoc++;
				break;
			}
		}
		std::cout << "# back = " << nbBack << ", # hd = " << nbHD << ", # coloc = " << nbColoc << std::endl;
	}
}

void ColocTesselerBasicCommands::computeSpearmanRankCorrelation()
{
	for (size_t current = 0; current < 2; current++) {
		clock_t t1 = clock(), t2;

		const poca::core::Scatterplot& vertices = m_colocTesseler->scattergramAt(current);
		m_spearmans2[current].clear();
		
		size_t curROI = 1, nextROI = curROI + 1, curIdx = 0;
		while (m_roiIndexPerLocs[current][m_sortedIndexesInROIs[current][curIdx]] != curROI) curIdx++;
		std::vector <double> X, Y;
		for (; curIdx < vertices.size(); curIdx++) {
			auto idx = m_sortedIndexesInROIs[current][curIdx];
			if (m_roiIndexPerLocs[current][idx] == nextROI) {
				double res = poca::geometry::spearmanRankCorrelationCoefficient(X, Y);
				m_spearmans2[current].push_back(res);
				X.clear();
				Y.clear();
				std::cout << "For color " << (current + 1) << " and ROI " << curROI << ", spearman = " << m_spearmans2[current].back() << std::endl;
				curROI = nextROI;
				nextROI++;
			}
			X.push_back(vertices[idx][0]);
			Y.push_back(vertices[idx][1]);
		}
		double res = poca::geometry::spearmanRankCorrelationCoefficient(X, Y);
		m_spearmans2[current].push_back(res);
		std::cout << "For color " << (current + 1) << " and ROI " << curROI << ", spearman = " << m_spearmans2[current].back() << std::endl;

		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		std::cout << "For color " << (current + 1) << ", spearman computed in " << elapsed << std::endl;
	}
}

void ColocTesselerBasicCommands::computeMandersCoefficients()
{
	for (size_t current = 0; current < 2; current++) {
		size_t other = (current + 1) % 2;
		const poca::core::Scatterplot& vertices = m_colocTesseler->scattergramAt(current);
		std::vector <unsigned char>& classesLocs = m_colocTesseler->classesLocsAt(current);
		m_manders2[current].clear();

		size_t curROI = 1, nextROI = curROI + 1, curIdx = 0;
		while (m_roiIndexPerLocs[current][m_sortedIndexesInROIs[current][curIdx]] != curROI) curIdx++;
		size_t locsAboveThresh = 0, locsInROIs = 0;
		float sumNom = 0., sumDenom1 = 0., sumDenom2 = 0.;
		for (; curIdx < vertices.size(); curIdx++) {
			auto idx = m_sortedIndexesInROIs[current][curIdx];
			if (m_roiIndexPerLocs[current][idx] == nextROI) {
				m_manders2[current].push_back(sumNom / sumDenom1);
				std::cout << "For color " << (current + 1) << " and ROI " << curROI << ", manders = " << m_manders2[current].back() << std::endl;
				std::cout << "Locs in ROIs for Manders computation -> " << locsAboveThresh << " / " << locsInROIs << std::endl;
				locsAboveThresh = locsInROIs = 0;
				sumNom = sumDenom1 = sumDenom2 = 0.;
				curROI = nextROI;
				nextROI++;
			}
			if (classesLocs[idx] == ColocTesseler::COLOC || classesLocs[idx] == ColocTesseler::HIGH_DENSITY) {
				double d1 = vertices[idx][current], d2 = vertices[idx][other];
				if (classesLocs[idx] == ColocTesseler::COLOC) {
					sumNom += d1;
					locsAboveThresh++;
				}
				sumDenom1 += d1;
				locsInROIs++;
			}
		}
		m_manders2[current].push_back(sumNom / sumDenom1);
		std::cout << "num = " << sumNom << ", denum = " << sumDenom1 << std::endl;
		std::cout << "For color " << (current + 1) << " and ROI " << curROI << ", manders = " << m_manders2[current].back() << std::endl;
		std::cout << "Locs in ROIs for Manders computation -> " << locsAboveThresh << " / " << locsInROIs << std::endl;
	}
}

