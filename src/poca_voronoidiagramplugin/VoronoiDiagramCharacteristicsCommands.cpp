/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagramCharacteristicsCommands.cpp
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

#include <General/Engine.hpp>
#include <General/Engine.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <Geometry/ObjectLists.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <General/Misc.h>
#include <General/MyData.hpp>

#include "VoronoiDiagramCharacteristicsCommands.hpp"

VoronoiDiagramCharacteristicsCommands::VoronoiDiagramCharacteristicsCommands(poca::geometry::VoronoiDiagram* _voro) :poca::core::Command("VoronoiDiagramCharacteristicsCommands")
{
	m_voronoi = _voro;
	m_monteCarloBinsMeans = m_monteCarloBinsStdDev = NULL;

	m_is3D = m_voronoi->dimension() == 3;
	m_bbox = m_voronoi->boundingBox();
	m_nbFaces = m_voronoi->nbElements();

	
	float env = 0.999;
	uint32_t nbBins = 100, degreePolynome = 3;
	bool onROIs = false;

	const nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
	if (parameters.contains(name())) {
		nlohmann::json param = parameters[name()];
		if (param.contains("voronoiCharacteristics")) {
			if (param["voronoiCharacteristics"].contains("env"))
				env = param["voronoiCharacteristics"]["env"].get<float>();
			if (param["voronoiCharacteristics"].contains("onROIs"))
				onROIs = param["voronoiCharacteristics"]["onROIs"].get<bool>();
			if (param["voronoiCharacteristics"].contains("nbBins"))
				nbBins = param["voronoiCharacteristics"]["nbBins"].get<uint32_t>();
			if (param["voronoiCharacteristics"].contains("degreePolynome"))
				degreePolynome = param["voronoiCharacteristics"]["degreePolynome"].get<uint32_t>();
		}
	}
	addCommandInfo(poca::core::CommandInfo(true, "objectCreationParameters", "env", env, "onROIs", onROIs, "nbBins", nbBins, "degreePolynome", degreePolynome));
}

VoronoiDiagramCharacteristicsCommands::VoronoiDiagramCharacteristicsCommands(const VoronoiDiagramCharacteristicsCommands& _o) : poca::core::Command(_o)
{
	m_voronoi = _o.m_voronoi;
}

VoronoiDiagramCharacteristicsCommands::~VoronoiDiagramCharacteristicsCommands()
{
}

void VoronoiDiagramCharacteristicsCommands::execute(poca::core::CommandInfo* _infos)
{
	if (hasCommand(_infos->nameCommand)) {
		loadParameters(*_infos);
	}
	else if (_infos->nameCommand == "voronoiCharacteristics") {
		 poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* obj = engine->getObject(m_voronoi);
		if (obj == NULL) return;

		float env = _infos->hasParameter("env") ? _infos->getParameter<float>("env") : getParameter<float>("voronoiCharacteristics", "env");
		bool onROIs = _infos->hasParameter("onROIs") ? _infos->getParameter<bool>("onROIs") : getParameter<bool>("voronoiCharacteristics", "onROIs");
		uint32_t nbBins = _infos->hasParameter("nbBins") ? _infos->getParameter<uint32_t>("nbBins") : getParameter<uint32_t>("voronoiCharacteristics", "nbBins");
		uint32_t degreePolynome = _infos->hasParameter("degreePolynome") ? _infos->getParameter<uint32_t>("degreePolynome") : getParameter<uint32_t>("voronoiCharacteristics", "degreePolynome");

		computeCharacteristics(env, onROIs, nbBins, degreePolynome);
		m_threshold = computeThreshold();

		std::cout << "Threshold is " << m_threshold << ", not normalized = " << (m_threshold * m_normalization) << std::endl;

		m_voronoi->executeCommand(false, "histogram", "feature", m_voronoi->dimension() == 2 ? std::string("area") : std::string("volume"), "action", std::string("changeBoundsCustom"), "max", m_threshold * m_normalization);
	}
}

poca::core::CommandInfo VoronoiDiagramCharacteristicsCommands::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
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
	else if (_nameCommand == "voronoiCharacteristics") {
		poca::core::CommandInfo ci(false, _nameCommand);
		if (_parameters.contains("env"))
			ci.addParameter("env", _parameters["env"].get<float>());
		if (_parameters.contains("onROIs"))
			ci.addParameter("onROIs", _parameters["onROIs"].get<bool>());
		if (_parameters.contains("nbBins"))
			ci.addParameter("nbBins", _parameters["nbBins"].get<uint32_t>());
		if (_parameters.contains("degreePolynome"))
			ci.addParameter("degreePolynome", _parameters["degreePolynome"].get<uint32_t>());
		return ci;
	}
	else if (_nameCommand == "createFilteredObjects" || _nameCommand == "invertSelection") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	return poca::core::CommandInfo();
}

poca::core::Command* VoronoiDiagramCharacteristicsCommands::copy()
{
	return new VoronoiDiagramCharacteristicsCommands(*this);
}

void VoronoiDiagramCharacteristicsCommands::computeCharacteristics(const float _env, const bool _onROIs, const uint32_t _nbBins, const uint32_t _degreePoly)
{
	m_env = _env;
	m_nbBins = _nbBins;

	std::vector <float> valTmp;
	if (m_voronoi->hasData("volume"))
		valTmp = m_voronoi->getMyData("volume")->getData<float>();
	if (m_voronoi->hasData("area"))
		valTmp = m_voronoi->getMyData("area")->getData<float>();
	if (valTmp.empty()) return;

	//Then only select the 95% emveloppe of the areas, to discard extrem values from very big cells
	std::sort(valTmp.begin(), valTmp.end());
	unsigned int index = valTmp.size() * m_env;
	valTmp.erase(valTmp.begin() + index, valTmp.end());

	//Normalize the areas
	m_normalization = poca::core::mean(valTmp);
	for (int n = 0; n < valTmp.size(); n++) valTmp[n] /= m_normalization;
	std::cout << "Normalization by " << m_normalization << " versus " << poca::core::median(valTmp) << std::endl;

	//create histogram
	m_hist = poca::core::Histogram<float>(valTmp, false, m_nbBins, true, 0.00000001);

	const std::vector<float>& xsH = m_hist.getTs(), &vals = m_hist.getBins();
	for (unsigned int n = 0; n < m_hist.getNbBins(); n++){
		m_xs.push_back(xsH[n]);
		m_ysPdfExp.push_back(vals[n]);
	}
	m_ysCdfExp.resize(m_ysPdfExp.size());

	m_maxHExp = -DBL_MAX;
	//create cdf experimental vector
	float nbAna = 0, nbExp = 0;
	for (unsigned int n = 0; n < m_ysPdfExp.size(); n++)
		nbExp += m_ysPdfExp[n];
	m_ysCdfExp[0] = m_ysPdfExp[0] / nbExp;
	for (unsigned int n = 1; n < m_ysPdfExp.size(); n++)
		m_ysCdfExp[n] = m_ysCdfExp[n - 1] + m_ysPdfExp[n] / nbExp;
	//Normalize the pdf experimental vector
	float nbData = valTmp.size();
	for (unsigned int n = 0; n < m_xs.size() - 1; n++){
		m_ysPdfExp[n] = m_ysPdfExp[n] / (nbData * (m_xs[n + 1] - m_xs[n]));
		if (m_ysPdfExp[n] > m_maxHExp)
			m_maxHExp = m_ysPdfExp[n];
	}
	m_ysPdfExp[m_xs.size() - 1] = m_ysPdfExp[m_xs.size() - 1] / (nbData * m_hist.getStepX());

	computeAnalyticCurve();
}

void VoronoiDiagramCharacteristicsCommands::computeAnalyticCurve()
{
	if (m_is3D)
		computeAnalyticCurve3D();
	else
		computeAnalyticCurve2D();

	m_maxHAna = 0.;
	for (unsigned int n = 0; n < m_ysPdfAna.size(); n++)
		if (m_ysPdfAna[n] > m_maxHAna) m_maxHAna = m_ysPdfAna[n];
}

void VoronoiDiagramCharacteristicsCommands::computeAnalyticCurve2D()
{
	m_ysPdfAna.resize(m_xs.size());
	m_ysCdfAna.resize(m_xs.size());
	float prec = 0;
	for (unsigned int n = 0; n < m_xs.size(); n++) {
		m_ysPdfAna[n] = (343. / 15.) * sqrt(7. / (2. * 3.14159265359)) * pow(m_xs[n], 2.5) * exp(-3.5 * m_xs[n]);
		m_ysCdfAna[n] = prec + m_ysPdfAna[n];
		prec = m_ysCdfAna[n];
	}
}

void VoronoiDiagramCharacteristicsCommands::computeAnalyticCurve3D()
{
	m_ysPdfAna.resize(m_xs.size());
	m_ysCdfAna.resize(m_xs.size());
	float prec = 0;
	for (unsigned int n = 0; n < m_xs.size(); n++) {
		m_ysPdfAna[n] = (3125. / 24.) * pow(m_xs[n], 4.) * exp(-5. * m_xs[n]);
		m_ysCdfAna[n] = prec + m_ysPdfAna[n];
		prec = m_ysCdfAna[n];
	}
}

const float VoronoiDiagramCharacteristicsCommands::computeThreshold()
{
	float maxHE, maxHA, minY = DBL_MAX;
	const std::vector <float>& xs = m_xs;
	const std::vector <float>& ysExp = getExperimentalCurvePDF(maxHE);
	const std::vector <float>& ysAna = getAnalyticCurvePDF(maxHA);

	const std::vector <float>& ysExpCDF = getExperimentalCurveCDF();
	const std::vector <float>& ysAnaCDF = getAnalyticCurveCDF();

	std::vector<float> y4(xs.size());
	for (unsigned int j = 0; j < xs.size(); j++) {
		y4[j] = ysExp[j] - ysAna[j];
		if (y4[j] < minY) minY = y4[j];
	}

	float ratioLocsClusters = 0, val = ysExp[0] - ysAna[0];
	unsigned int cpt = 1;
	while (val > 0. && cpt < xs.size()) {
		ratioLocsClusters += val;
		val = ysExp[cpt] - ysAna[cpt];
		if (val > 0)
			cpt++;
	}
	if (cpt != xs.size())
		ratioLocsClusters = ysExpCDF[cpt] - ysAnaCDF[cpt];
	float Lr = 1.92 / sqrt((float)m_nbInitialLocs);

	if (cpt != xs.size())
		return xs[cpt];
	else
		return DBL_MAX;
}

const std::vector <float>& VoronoiDiagramCharacteristicsCommands::getExperimentalCurvePDF(float& _maxH)
{
	_maxH = m_maxHExp;
	return m_ysPdfExp;
}

const std::vector <float>& VoronoiDiagramCharacteristicsCommands::getAnalyticCurvePDF(float& _maxH)
{
	_maxH = m_maxHAna;
	return m_ysPdfAna;
}

const std::vector <float>& VoronoiDiagramCharacteristicsCommands::getExperimentalCurveCDF()
{
	return m_ysCdfExp;
}

const std::vector <float>& VoronoiDiagramCharacteristicsCommands::getAnalyticCurveCDF()
{
	return m_ysCdfAna;
}

