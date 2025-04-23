/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      KRipley.cpp
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

#include <QtWidgets/QMessageBox>
#include <qmath.h>

#include <Geometry/DetectionSet.hpp>
#include <General/Misc.h>
#include <Geometry/BasicComputation.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <General/Engine.hpp>
#include <General/MyData.hpp>

#include "KRipley.hpp"


KRipleyCommand::KRipleyCommand(poca::geometry::DetectionSet* _ds) :poca::core::Command("KRipleyCommand")
{
	m_dset = _ds;
	m_density = m_dset->averageDensity();
	const poca::core::BoundingBox& bbox = m_dset->boundingBox();
	m_x = bbox[0];
	m_y = bbox[1];
	m_w = bbox[3];
	m_h = bbox[4];

	
	const nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
	addCommandInfo(poca::core::CommandInfo(false, "kripley", "minRadius", 10.f, "maxRadius", 200.f, "step", 10.f));
	if (parameters.contains(name())) {
		nlohmann::json param = parameters[name()];
		if(param.contains("kripley") && param["kripley"].contains("minRadius") && param["kripley"].contains("maxRadius") && param["kripley"].contains("step"))
			loadParameters(poca::core::CommandInfo(false, "kripley",
				"minRadius", param["kripley"]["minRadius"].get<float>(), 
				"maxRadius", param["kripley"]["maxRadius"].get<float>(), 
				"step", param["kripley"]["step"].get<float>()));
	}
}

KRipleyCommand::KRipleyCommand(const KRipleyCommand& _o) : poca::core::Command(_o)
{
	m_dset = _o.m_dset;
}

KRipleyCommand::~KRipleyCommand()
{
}

poca::core::Command* KRipleyCommand::copy()
{
	return new KRipleyCommand(*this);
}

void KRipleyCommand::execute(poca::core::CommandInfo* _infos)
{
	if (hasCommand(_infos->nameCommand)) {
		loadParameters(*_infos);
	}
	if (_infos->nameCommand == "kripley") {
		if (m_dset->dimension() == 3) {
			QMessageBox msgBox;
			msgBox.setText("K-Ripley only works for 2D datasets for now.");
			msgBox.exec();
			return;
		}
		float minRadius = _infos->getParameter <float>("minRadius");
		float maxRadius = _infos->getParameter <float>("maxRadius");
		float step = _infos->getParameter <float>("step");
		computeKRipley(minRadius, maxRadius, step);
	}
	else if (_infos->nameCommand == "getKRipleyResultsKs") {
		if (m_ts.empty()) return;
		_infos->addParameters("nbSteps", m_nbSteps,
			"values", m_ks.data(),
			"ts", m_ts.data(),
			"ls", m_ls.data());
	}
	else if (_infos->nameCommand == "getKRipleyResultsLs") {
		if (m_ts.empty()) return;
		_infos->addParameters("nbSteps", m_nbSteps,
			"values", m_ls.data(),
			"ts", m_ts.data(),
			"ls", m_ls.data());
	}
	else if (_infos->nameCommand == "getKRipleyResults") {
		if (m_ts.empty()) return;
		_infos->addParameters("nbSteps", m_nbSteps,
			"ks", m_ks.data(),
			"ts", m_ts.data(),
			"ls", m_ls.data());
	}
}

poca::core::CommandInfo KRipleyCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "kripley") {
		float minRadius, maxRadius, step;
		bool complete = _parameters.contains("minRadius");
		if (complete)
			minRadius = _parameters["minRadius"].get<float>();
		complete &= _parameters.contains("maxRadius");
		if (complete)
			maxRadius = _parameters["maxRadius"].get<float>();
		complete &= _parameters.contains("step");
		if (complete) {
			step = _parameters["step"].get<float>();
			return poca::core::CommandInfo(false, _nameCommand, "minRadius", minRadius, "maxRadius", maxRadius, "step", step);
		}
	}
	else if (_nameCommand == "displayDBSCAN") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "getKRipleyResultsKs" || _nameCommand == "getKRipleyResultsLs" || _nameCommand == "getKRipleyResults") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	return poca::core::CommandInfo();
}

void KRipleyCommand::computeKRipley(const float _minR, const float  _maxR, const float _stepR)
{
	m_nbSteps = 0;
	for (double r = _minR; r <= _maxR; r += _stepR, m_nbSteps++);
	size_t index = 0;
	m_results.resize(m_nbSteps);
	m_ks.resize(m_nbSteps);
	m_ls.resize(m_nbSteps);
	m_ts.resize(m_nbSteps);

	unsigned int nbForUpdate = m_nbSteps / 100., cpt = 0;
	if (nbForUpdate == 0) nbForUpdate = 1;
	printf("Computing KRipley: %.2f %%", (0. / m_nbSteps * 100.));

	for (double r = _minR; r <= _maxR; r += _stepR, index++) {
		if (cpt++ % nbForUpdate == 0) printf("\rComputing KRipley: %.2f %%", ((double)cpt / m_nbSteps * 100.));
		m_ts[index] = r;
		double val = computeRipleyFunction(r);
		m_ks[index] = val;
		val = sqrt(val / M_PI) - r;
		m_results[index] = val;
		m_ls[index] = val;
	}
	printf("\rComputing KRipley: 100 %%\n");
}

const float KRipleyCommand::computeRipleyFunction(const float _r)
{
	double res = 0., divisor = m_density, r2 = _r * _r;
	double meanNbNeighbors = 0, nbP = m_dset->nbPoints(), trueMean = 0.;
	const std::vector <float>& xs = m_dset->getMyData("x")->getData<float>();
	const std::vector <float>& ys = m_dset->getMyData("y")->getData<float>();
	const double search_radius = static_cast<double>(r2);
	std::vector<std::pair<std::size_t, double> > ret_matches;
	nanoflann::SearchParams params;
	std::size_t nMatches;
	poca::geometry::KdTree_DetectionPoint* tree = m_dset->getKdTree();
	for (int n = 0; n < m_dset->nbPoints(); n++) {
		float x = xs[n], y = ys[n];
		bool crossBorder = (x < (m_x + _r)) || (y < (m_y + _r)) || (x > (m_w - _r)) || (y > (m_h - _r));
		double areaDomain = M_PI * _r * _r;
		double factorArea = (crossBorder) ? edgeCorrection(x, y, _r) : areaDomain;
		factorArea = areaDomain / factorArea;

		double sum = 0.;
		const double queryPt[3] = { x, y, 0. };
		nMatches = tree->radiusSearch(&queryPt[0], search_radius, ret_matches, params);
		sum = nMatches - 1;

		res += (factorArea * sum) / divisor;
		trueMean += (factorArea * sum) / nbP;
		meanNbNeighbors += (double)nMatches / nbP;
	}
	res /= (double)m_dset->nbPoints();
	return res;
}

const float KRipleyCommand::edgeCorrection(const float _x, const float _y, const float _r)
{
	float areaCircle = M_PI * _r * _r;

	std::vector <poca::core::Vec2mf> intersectionPoints;
	poca::geometry::circleLineIntersect(m_x, m_y, m_w, m_y, _x, _y, _r, intersectionPoints);
	poca::geometry::circleLineIntersect(m_w, m_y, m_w, m_h, _x, _y, _r, intersectionPoints);
	poca::geometry::circleLineIntersect(m_x, m_h, m_w, m_h, _x, _y, _r, intersectionPoints);
	poca::geometry::circleLineIntersect(m_x, m_y, m_x, m_h, _x, _y, _r, intersectionPoints);

	if (intersectionPoints.size() != 2 && intersectionPoints.size() != 4) {
		return areaCircle;
	}
	else {
		int nbCornersInsideCircle = 0;
		if (poca::geometry::distance<float>(_x, _y, 0., 0.) < _r)
			nbCornersInsideCircle++;
		if (poca::geometry::distance<float>(_x, _y, 0., m_h) < _r)
			nbCornersInsideCircle++;
		if (poca::geometry::distance<float>(_x, _y, m_w, m_h) < _r)
			nbCornersInsideCircle++;
		if (poca::geometry::distance<float>(_x, _y, m_w, 0.) < _r)
			nbCornersInsideCircle++;

		double areaCircle = M_PI * _r * _r;
		double area = areaCircle;

		for (unsigned int n = 0; n < intersectionPoints.size(); n += 2) {
			poca::core::Vec2mf p1 = intersectionPoints[n], p2 = intersectionPoints[n + 1];
			double areaCircularSegment = poca::geometry::computeAreaCircularSegment(_x, _y, _r, p1, p2);
			double a = poca::geometry::distance<float>(_x, _y, p1.x(), p1.y());
			double b = poca::geometry::distance<float>(_x, _y, p2.x(), p2.y());
			double c = poca::geometry::distance<float>(p1.x(), p1.y(), p2.x(), p2.y());
			double areaTriangle = poca::geometry::computeAreaTriangle(a, b, c);

			if (nbCornersInsideCircle == 0)
				area -= areaCircularSegment;
			else if (nbCornersInsideCircle == 1)
				area -= (areaCircularSegment + areaTriangle);
		}
		return area;
	}
}

