/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ColocTesseler.cpp
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

#include <QtCore/QTime>

#include <Interfaces/HistogramInterface.hpp>
#include <General/MyData.hpp>
#include <General/Misc.h>

#include "ColocTesseler.hpp"

ColocTesseler::ColocTesseler(poca::geometry::VoronoiDiagram* _voro1, poca::geometry::VoronoiDiagram* _voro2) : poca::core::BasicComponent("ColocTesseler")//, m_correction(true)
{
	m_spearmans[0] = m_spearmans[1] = m_manders[0] = m_manders[1] = std::nanf("1");

	QTime time, test;
	time.start();

	m_voronois[0] = _voro1;
	m_voronois[1] = _voro2;

	for (size_t n = 0; n < 2; n++){
		const std::vector <float>& values = m_voronois[n]->getMyData("density")->getData<float>();

		poca::core::BoundingBox bbox = m_voronois[n]->boundingBox();
		float nbs = m_voronois[n]->nbElements(), averageD = 0.f;
		float w = bbox[3] - bbox[0], h = bbox[4] - bbox[1], t = bbox[5] - bbox[2];
		averageD = m_voronois[n]->averageDensity();

		std::vector <float> normValues;
		std::transform(values.begin(), values.end(), std::back_inserter(normValues), [&averageD](auto i) { return  i / averageD; });
		std::string nameFeature = "density_color";
		nameFeature.append(std::to_string(n + 1));
		m_data.insert(std::make_pair(nameFeature, new poca::core::MyData(new poca::core::Histogram<float>(normValues, false), true)));

		uint32_t curNbPoints = m_voronois[n]->nbElements();
		m_scattergram[n].resize(curNbPoints);
		m_scattergramLog[n].resize(curNbPoints);
		m_classesLocs[n].resize(curNbPoints, ColocTesseler::BACKGROUND);
	}

	const std::size_t num_results = 1;
	std::vector<size_t> ret_index(num_results);
	std::vector<double> out_dist_sqr(num_results);
	for (uint32_t currentVorId = 0; currentVorId < 2; currentVorId++) {
		uint32_t otherId = (currentVorId + 1) % 2;
		uint32_t curNbPoints = m_voronois[currentVorId]->nbElements();
		uint32_t curDimemsion = m_voronois[currentVorId]->dimension(), otherDimemsion = m_voronois[otherId]->dimension();

		double nbPs = curNbPoints;
		unsigned int nbForUpdate = nbPs / 100., cptTimer = 0;
		if (nbForUpdate == 0) nbForUpdate = 1;
		std::printf("Creation of the scatterplot for color %i: %.2f %%", currentVorId, (0. / nbPs * 100.));

		std::string currentNameFeature = "density_color" + std::to_string(currentVorId + 1);
		std::string otherNameFeature = "density_color" + std::to_string(otherId + 1);
		const std::vector <float>& currentDensity = getMyData(currentNameFeature)->getData<float>();
		const std::vector <float>& otherDensity = getMyData(otherNameFeature)->getData<float>();
		
		const float* xsCur = m_voronois[currentVorId]->getXs(), * ysCur = m_voronois[currentVorId]->getYs(), * zsCur = m_voronois[currentVorId]->getZs();
		const float* xsOther = m_voronois[otherId]->getXs(), * ysOther = m_voronois[otherId]->getYs(), * zsOther = m_voronois[otherId]->getZs();
		poca::geometry::KdTree_DetectionPoint* otherT = m_voronois[otherId]->getKdTree();
		m_indexTrianglePointsInOtherColor[currentVorId].resize(curNbPoints);
		
		for (unsigned int n = 0; n < curNbPoints; n++) {
			float d1, d2;
			float x = xsCur[n], y = ysCur[n], z = (curDimemsion == 3) ? zsCur[n] : 0.f;
			const double queryPt[3] = { x, y, z };

			if (currentVorId == 0) {
				d1 = currentDensity[n];
				otherT->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
				d2 = otherDensity[ret_index[0]];
			}
			else {
				d2 = currentDensity[n];
				otherT->knnSearch(&queryPt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
				d1 = otherDensity[ret_index[0]];
			}
			m_indexTrianglePointsInOtherColor[currentVorId][n] = m_voronois[otherId]->indexTriangleOfPoint(x, y, z);

			m_scattergram[currentVorId][n].set(d1, d2);
			m_scattergramLog[currentVorId][n].set(log10(d1), log10(d2));
			if (cptTimer++ % nbForUpdate == 0) 	std::printf("\rCreation of the scatterplot for color %i: %.2f %%", currentVorId, ((double)n / nbPs * 100.));
		}
		std::printf("\rCreation of the scatterplot for color %i: 100 %%\n", currentVorId);
		int elapsedTime = time.elapsed();
		time.restart();
		test = QTime();
		test = test.addMSecs(elapsedTime);
		std::cout << "Time elapsed for dataset " << currentVorId << " -> " << test.hour() << " h " << test.minute() << " minutes " << test.second() << " s " << test.msec() << " ms - " << elapsedTime << std::endl << std::endl;
		auto nbT = std::count(m_indexTrianglePointsInOtherColor[currentVorId].begin(), m_indexTrianglePointsInOtherColor[currentVorId].end(), std::numeric_limits<std::uint32_t>::max());
		std::cout << "Nb missed other tri for color " << currentVorId << " = " << nbT << " / " << m_indexTrianglePointsInOtherColor[currentVorId].size() << std::endl;
	}
}

ColocTesseler::~ColocTesseler()
{

}

poca::core::BasicComponentInterface* ColocTesseler::copy()
{
	return new ColocTesseler(*this);
}

void ColocTesseler::executeCommand(poca::core::CommandInfo* _com)
{
	BasicComponent::executeCommand(_com);
	if (_com->nameCommand == "computeCoefficients") {
		if (_com->hasParameter("spearmanColor1"))
			m_spearmans[0] = _com->getParameter<float>("spearmanColor1");
		if (_com->hasParameter("spearmanColor2"))
			m_spearmans[1] = _com->getParameter<float>("spearmanColor2");
		if (_com->hasParameter("mandersColor1"))
			m_manders[0] = _com->getParameter<float>("mandersColor1");
		if (_com->hasParameter("mandersColor2"))
			m_manders[1] = _com->getParameter<float>("mandersColor2");
		if (_com->hasParameter("spearman2Color1"))
			m_spearmans2[0] = _com->getParameter<std::vector<float>>("spearman2Color1");
		if (_com->hasParameter("spearman2Color2"))
			m_spearmans2[1] = _com->getParameter<std::vector<float>>("spearman2Color2");
		if (_com->hasParameter("manders2Color1"))
			m_manders2[0] = _com->getParameter<std::vector<float>>("manders2Color1");
		if (_com->hasParameter("manders2Color2"))
			m_manders2[1] = _com->getParameter<std::vector<float>>("manders2Color2");
	}
}