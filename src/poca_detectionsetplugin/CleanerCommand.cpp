/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      CleanerCommand.cpp
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

#include <Windows.h>
#include <gl/glew.h>
#include <gl/GL.h>
#include <ctime>

#include <DesignPatterns/ListDatasetsSingleton.hpp>
#include <DesignPatterns/StateSoftwareSingleton.hpp>
#include <General/EquationFit.hpp>
#include <OpenGL/Camera.hpp>
#include <OpenGL/Shader.hpp>
#include <Objects/MyObject.hpp>

#include "CleanerCommand.hpp"

CleanerCommand::CleanerCommand(poca::geometry::DetectionSet* _dset) :poca::core::Command("CleanerCommand"), m_initializeDisplay(false)
{
	m_dset = _dset;
	m_eqnBlinks = m_eqnTOffs = m_eqnTOns = NULL;

	poca::core::StateSoftwareSingleton* sss = poca::core::StateSoftwareSingleton::instance();
	const nlohmann::json& parameters = sss->getParameters();
	if (!parameters.contains(name())) {
		addCommandInfo(poca::core::CommandInfo(false, "clean", "radius", 50.f, "maxDarkTime", (uint32_t)10, "fixedDarkTime", false));
		addCommandInfo(poca::core::CommandInfo(false, "displayCleanedLocs", true));
	}
	else {
		nlohmann::json param = parameters[name()];
		float radius = param["clean"]["radius"].get<float>();
		uint32_t maxDarkTime = param["clean"]["maxDarkTime"].get<uint32_t>();
		bool fixedDarkTime = param["clean"]["fixedDarkTime"].get<bool>();
		addCommandInfo(poca::core::CommandInfo(false, "clean", "radius", radius, "maxDarkTime", maxDarkTime, "fixedDarkTime", fixedDarkTime));
		addCommandInfo(poca::core::CommandInfo(false, "displayCleanedLocs", param["displayCleanedLocs"].get<bool>()));
	}
}

CleanerCommand::CleanerCommand(const CleanerCommand& _o) : poca::core::Command(_o), m_initializeDisplay(_o.m_initializeDisplay)
{
	m_dset = _o.m_dset;
	m_eqnBlinks = m_eqnTOffs = m_eqnTOns = NULL;
}

CleanerCommand::~CleanerCommand() 
{
	if (m_eqnBlinks != NULL)
		delete m_eqnBlinks;
	if (m_eqnTOffs != NULL)
		delete m_eqnTOffs;
	if (m_eqnTOns != NULL)
		delete m_eqnTOns;
}

void CleanerCommand::execute(poca::core::CommandInfo* _infos)
{
	if (hasCommand(_infos->nameCommand))
		loadParameters(*_infos);
	if (_infos->getNameCommand() == "clean") {
		if (!hasParameter("clean", "radius")) return;
		if (!hasParameter("clean", "maxDarkTime")) return;
		if (!hasParameter("clean", "fixedDarkTime")) return;
		float radius = getParameter<float>("clean", "radius");
		uint32_t maxDT = getParameter<uint32_t>("clean", "maxDarkTime");
		bool fixedDT = getParameter<bool>("clean", "fixedDarkTime");
		poca::core::MyObjectInterface* obj = cleanDetectionSet(radius, maxDT, fixedDT);
		_infos->addParameter("object", static_cast <poca::core::MyObjectInterface*>(obj));
	}
	else if (_infos->nameCommand == "display") {
		poca::opengl::Camera* cam = _infos->getParameterPtr<poca::opengl::Camera>("camera");
		bool offscrean = false;
		if (_infos->hasParameter("offscreen"))
			offscrean = _infos->getParameter<bool>("offscreen");
		display(cam, offscrean);
	}
	else if (_infos->nameCommand == "getCleanEquations") {
		if (m_eqnBlinks == NULL) return;
		_infos->addParameters("blinks", m_eqnBlinks,
			"tons", m_eqnTOns, 
			"toffs", m_eqnTOffs,
			"nbEmissionBursts", m_nbEmissionBursts,
			"nbOriginalLocs", m_nbOriginalLocs,
			"nbSupressedLocs", m_nbSuppressedLocs,
			"nbAddedLocs", m_nbCorrectedLocs,
			"nbUncorrectedLocs", m_nbUncorrectedLocs,
			"darkTime", m_darkTime);
	}
}

poca::core::CommandInfo CleanerCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "clean") {
		float radius;
		uint32_t maxDarkTime;
		bool complete = _parameters.contains("radius"), fixedDarkTime;
		if (complete)
			radius = _parameters["radius"].get<float>();
		complete &= _parameters.contains("fixedDarkTime");
		if (complete)
			fixedDarkTime = _parameters["fixedDarkTime"].get<bool>();
		complete &= _parameters.contains("maxDarkTime");
		if (complete) {
			maxDarkTime = _parameters["maxDarkTime"].get<uint32_t>();
			return poca::core::CommandInfo(false, _nameCommand, "radius", radius, "maxDarkTime", maxDarkTime, "fixedDarkTime", fixedDarkTime);
		}
	}
	else if (_nameCommand == "displayCleanedLocs") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "getCleanEquations" || _nameCommand == "getCleanedData") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	return poca::core::CommandInfo();
}

poca::core::Command* CleanerCommand::copy() 
{
	return new CleanerCommand(*this);
}

poca::core::MyObjectInterface* CleanerCommand::cleanDetectionSet(const float _radius, const uint32_t _maxDT, const bool _fixedDT)
{
	m_nbUncorrectedLocs = m_nbCorrectedLocs = m_nbOriginalLocs = m_nbSuppressedLocs = 0;
	size_t nbFrames = m_dset->nbSlices();
	if (nbFrames == 1) return NULL;

	clock_t t1, t2;
	srand((unsigned)time(NULL));
	t1 = clock();

	std::cout << "Nb time -> " << nbFrames << std::endl;
	if (!m_dset->hasData("frame")) return NULL;
	const std::vector <float>& times = m_dset->getData("frame");
	m_nbOriginalLocs = times.size();

	m_darkTime = _maxDT;

	if (m_pointsPerFrame.empty()) {
		float currentTime = times[0];
		uint32_t n = 0;
		m_pointsPerFrame.push_back(n++);
		for (; n < m_dset->nbPoints(); n++) {
			if (times[n] != currentTime) {
				currentTime = times[n];
				m_pointsPerFrame.push_back(n);
			}
		}
		m_pointsPerFrame.push_back(n);
	}

	//For computing emission bursts we fix the DT as 1
	m_mergedPoints.clear();
	m_firstsMerged.clear();
	m_nbBlinks.clear();
	m_nbSequencesOff.clear();
	m_totalOffs.clear();
	m_nbEmissionBursts = computeNbLocsCorrectedSpecifiedDT((uint32_t)1, _radius);
	uint32_t nb = m_nbEmissionBursts;
	std::cout << "# emission bursts: " << m_nbEmissionBursts << std::endl;
	if (!_fixedDT)
		m_darkTime = computeAnalysisParameters(_maxDT, _radius);
	m_mergedPoints.clear();
	m_firstsMerged.clear();
	m_nbBlinks.clear();
	m_nbSequencesOff.clear();
	m_totalOffs.clear();
	nb = computeNbLocsCorrectedSpecifiedDT(m_darkTime, _radius);

	bool hasZ = m_dset->dimension() == 3;
	const std::vector <float>& xs = m_dset->getData("x");
	const std::vector <float>& ys = m_dset->getData("y");
	const std::vector <float>& zs = hasZ ? m_dset->getData("z") : std::vector <float>(xs.size(), 0.f);
	const std::vector <float>& intensities = m_dset->hasData("intensity") ? m_dset->getData("intensity") : std::vector <float>();

	std::vector <float> newXs(nb), newYs(nb), newZs(hasZ ? nb : 0), newTimes(nb), newIntensities(intensities.empty() ? 0 : nb), lengthes(nb), lifetimes(nb), blinks(nb), nbSeqOFF(nb), totalOFFs(nb), nbSeqON(nb);
	for (size_t n = 0; n < m_firstsMerged.size() - 1; n++) {
		poca::core::Vec3mf barycenter;
		float intensity = 0.f, nbLocs = (float)(m_firstsMerged[n + 1] - m_firstsMerged[n]);
		for (uint32_t currentLoc = m_firstsMerged[n]; currentLoc < m_firstsMerged[n + 1]; currentLoc++) {
			barycenter += (poca::core::Vec3mf(xs[m_mergedPoints[currentLoc]], ys[m_mergedPoints[currentLoc]], zs[m_mergedPoints[currentLoc]]) / nbLocs);
			if (!intensities.empty())
				intensity += intensities[currentLoc];
		}
		newXs[n] = barycenter.x();
		newYs[n] = barycenter.y();
		if(hasZ)
			newZs[n] = barycenter.z();
		newTimes[n] = times[m_mergedPoints[m_firstsMerged[n]]];
		lengthes[n] = (float)(m_firstsMerged[n + 1] - m_firstsMerged[n]);
		lifetimes[n] = lengthes[n] == 0 ? 1 : (float)(times[m_mergedPoints[m_firstsMerged[n + 1] - 1]] - times[m_mergedPoints[m_firstsMerged[n]]] + 1.f);
		if (!newIntensities.empty())
			newIntensities[n] = intensity;
		if (lengthes[n] > 1.f) {
			m_nbSuppressedLocs += (uint32_t)lengthes[n];
			m_nbCorrectedLocs++;
		}
		else
			m_nbUncorrectedLocs++;
		blinks[n] = m_nbBlinks[n];
		nbSeqOFF[n] = m_nbSequencesOff[n];
		nbSeqON[n] = 1 + m_nbBlinks[n];
		totalOFFs[n] = m_totalOffs[n];
	}

	std::map <std::string, std::vector <float>> features;

	features["x"] = newXs;
	features["y"] = newYs;
	if (hasZ)
		features["z"] = newZs;
	features["frame"] = newTimes;
	features["total ON"] = lengthes;
	features["lifetime"] = lifetimes;
	features["# seq ON"] = nbSeqON;
	features["# seq OFF"] = nbSeqOFF;
	features["total OFF"] = totalOFFs;
	features["blinks"] = blinks;
	if (!newIntensities.empty())
		features["intensity"] = newIntensities;

	m_featuresCorrectedLocs = features;

	std::cout << "Correction with a fixed dark time of " << m_darkTime << ": " << nb << " (composed of " << m_nbUncorrectedLocs << " uncorrected and " << m_nbCorrectedLocs << " corrected locs)." << std::endl;
	m_initializeDisplay = true;

	poca::geometry::DetectionSet* dset = new poca::geometry::DetectionSet(features);

	poca::core::ListDatasetsSingleton* lds = poca::core::ListDatasetsSingleton::instance();
	poca::core::MyObjectInterface* obj = lds->getObject(m_dset);
	const std::string& dir = obj->getDir(), name = obj->getName();
	QString newName(name.c_str());
	int index = newName.lastIndexOf(".");
	newName.insert(index, "_cleaned");

	poca::core::MyObject* wobj = new poca::core::MyObject();
	wobj->setDir(dir.c_str());
	wobj->setName(newName.toLatin1().data());
	wobj->addBasicComponent(dset);
	wobj->setDimension(dset->dimension());

	m_initializeDisplay = true;
	
	return wobj;
}

size_t CleanerCommand::computeNbLocsCorrectedSpecifiedDT(const uint32_t _maxDarkTime, const float _radius)
{
	const std::vector <float>& xs = m_dset->getData("x");
	const std::vector <float>& ys = m_dset->getData("y");
	const std::vector <float>& zs = m_dset->dimension() == 3 ? m_dset->getData("z") : std::vector <float>(xs.size(), 0.f);
	const std::vector <float>& times = m_dset->getData("frame");
	std::vector <bool> pointsDone(m_dset->nbPoints(), false);
	for (size_t t = 0; t < m_pointsPerFrame.size() - 2; t++) {//We use -2 because m_pointsPerFrame has one value more than the number of frames, and we stop at the avant dernier frame
		for (uint32_t origPoint = m_pointsPerFrame[t]; origPoint < m_pointsPerFrame[t + 1]; origPoint++) {
			if (pointsDone[origPoint]) continue;
			uint32_t currentT = t, currentFrame = times[m_pointsPerFrame[currentT]];
			pointsDone[origPoint] = true;
			m_firstsMerged.push_back(m_mergedPoints.size());
			m_mergedPoints.push_back(origPoint);
			uint32_t nextTime = currentT + 1, currentBlinks = 0, totalTimeOff = 0, currrentTimeOff = times[origPoint], nbSequenceOff = 0;
			bool goesOff = false;
			poca::core::Vec3mf barycenter(xs[origPoint], ys[origPoint], zs[origPoint]);
			while (nextTime < m_pointsPerFrame.size() - 1 && times[m_pointsPerFrame[nextTime]] - currentFrame <= _maxDarkTime) {//times[m_pointsPerFrame[nextTime]] is equivalent to nextFrame
				bool neighFound = false;
				uint32_t index;
				float d = pow(_radius, 2.f);
				for (uint32_t currentPoint = m_pointsPerFrame[nextTime]; currentPoint < m_pointsPerFrame[nextTime + 1]; currentPoint++) {
					if (pointsDone[currentPoint]) continue;
					float x2 = xs[currentPoint] - barycenter.x(), y2 = ys[currentPoint] - barycenter.y(), z2 = zs[currentPoint] - barycenter.z();
					float length = x2 * x2 + y2 * y2 + z2 * z2;
					if (length < d) {
						neighFound = true;
						index = currentPoint;
						d = length;
					}
				}
				if (neighFound) {
					m_mergedPoints.push_back(index);
					barycenter = (barycenter + poca::core::Vec3mf(xs[index], ys[index], zs[index])) / 2.f;
					pointsDone[index] = true;
					if (goesOff) {
						goesOff = false;
						currentBlinks++;
					}
					else if(times[m_pointsPerFrame[nextTime]] - currentFrame > 1) {
						currentBlinks++;
						nbSequenceOff++;
					}
					currentFrame = times[m_pointsPerFrame[nextTime]];
					totalTimeOff += (times[index] - currrentTimeOff) - 1;
					currrentTimeOff = times[index];
				}
				else {
					if (!goesOff)
						nbSequenceOff++;
					goesOff = true;
				}
				nextTime++;
			}
			if(goesOff)
				nbSequenceOff--;
			m_nbBlinks.push_back(currentBlinks); 
			m_nbSequencesOff.push_back(nbSequenceOff);
			m_totalOffs.push_back(totalTimeOff);
		}
	}
	m_firstsMerged.push_back(m_mergedPoints.size());

	// Create a map to store the frequency of each element in vector
	std::map<uint32_t, int> countMap;
	// Iterate over the vector and store the frequency of each element in map
	for (auto& elem : m_mergedPoints)
	{
		auto result = countMap.insert(std::pair<uint32_t, int>(elem, 1));
		if (result.second == false)
			result.first->second++;
	}
	// Iterate over the map
	for (auto& elem : countMap)
	{
		// If frequency count is greater than 1 then its a duplicate element
		if (elem.second > 1)
		{
			std::cout << elem.first << " :: " << elem.second << std::endl;
		}
	}

	return m_firstsMerged.size() - 1;
}

uint32_t CleanerCommand::computeAnalysisParameters(const uint32_t _maxDarkTime, const float _radius)
{
	size_t nbFrames = m_dset->nbSlices();
	/****** Computation of the parameters for the analysis *********/
	std::vector<double> blinks(nbFrames, 0.);
	std::vector<double> tons(nbFrames, 0.);
	std::vector<double> toffs(_maxDarkTime + 1, 0.);

	const std::vector <float>& xs = m_dset->getData("x");
	const std::vector <float>& ys = m_dset->getData("y");
	const std::vector <float>& zs = m_dset->dimension() == 3 ? m_dset->getData("z") : std::vector <float>(xs.size(), 0.f);
	const std::vector <float>& times = m_dset->getData("frame");
	std::vector <bool> pointsDone(m_dset->nbPoints(), false);
	for (size_t t = 0; t < m_pointsPerFrame.size() - 1; t++) {
		for (uint32_t origPoint = m_pointsPerFrame[t]; origPoint < m_pointsPerFrame[t + 1]; origPoint++) {
			if (pointsDone[origPoint]) continue;
			uint32_t currentT = t, currentFrame = times[m_pointsPerFrame[currentT]];
			pointsDone[origPoint] = true;
			uint32_t nextTime = currentT + 1, nextFrame = times[m_pointsPerFrame[nextTime]];
			uint32_t currentDarkTime = nextFrame - currentFrame, nbBlinks = 0, nbOn = 1;
			poca::core::Vec3mf barycenter(xs[origPoint], ys[origPoint], zs[origPoint]);
			while (currentDarkTime <= _maxDarkTime && nextTime < m_pointsPerFrame.size() - 1) {
				bool neighFound = false;
				uint32_t index;
				float d = pow(_radius, 2.f);
				for (uint32_t currentPoint = m_pointsPerFrame[nextTime]; currentPoint < m_pointsPerFrame[nextTime + 1]; currentPoint++) {
					float x2 = xs[currentPoint] - barycenter.x(), y2 = ys[currentPoint] - barycenter.y(), z2 = zs[currentPoint] - barycenter.z();
					float length = x2 * x2 + y2 * y2 + z2 * z2;
					if (length < d) {
						neighFound = true;
						index = currentPoint;
						d = length;
					}
				}
				if (neighFound) {
					barycenter = (barycenter + poca::core::Vec3mf(xs[index], ys[index], zs[index])) / 2.f;
					pointsDone[index] = true;
					currentFrame = nextFrame;
					nbOn++;
					if (currentDarkTime != 0) {
						nbBlinks++;
						toffs[currentDarkTime]++;
					}
				}
				else {
					tons[nbOn]++;
					nbOn = 0;
				}
				nextTime++;
				nextFrame = times[m_pointsPerFrame[nextTime]];
				currentDarkTime = nextFrame - currentFrame;
			}
			if (nbOn != 0 && nbOn < nbFrames)
				tons[nbOn]++;
			blinks[nbBlinks]++;
		}
	}

	std::vector<double> ts(_maxDarkTime);
	for (uint32_t i = 0; i < _maxDarkTime; i++)
		ts[i] = i;//+1;
	/****************************** Fit for blinks *******************************/
	double totalBlinks = 0.;
	for (int i = 0; i < nbFrames; i++)
		if (blinks[i] > 0)
			totalBlinks += blinks[i];
	blinks.resize(_maxDarkTime + 1);
	for (uint32_t n = 0; n < _maxDarkTime + 1; n++)
		blinks[n] = blinks[n] / totalBlinks;// * ( n + 1 );
	m_eqnBlinks = new poca::core::EquationFit(ts, blinks, poca::core::EquationFit::LeeFunction);
	
	for (uint32_t i = 1; i < _maxDarkTime + 1; i++) {
		ts[i - 1] = i;
		tons[i - 1] = tons[i];
	}
	tons.resize(_maxDarkTime);
	m_eqnTOns = new poca::core::EquationFit(ts, tons, poca::core::EquationFit::ExpDecayValue);

	for (uint32_t i = 1; i < _maxDarkTime + 1; i++) {
		ts[i - 1] = i;
		toffs[i - 1] = toffs[i];
	}
	toffs.resize(_maxDarkTime);
	m_eqnTOffs = new poca::core::EquationFit(ts, toffs, poca::core::EquationFit::ExpDecayHalLife);

	return (3 * m_eqnTOffs->getParams()[2] + 0.5);
}

void CleanerCommand::freeGPUMemory()
{
	m_pointBuffer.freeGPUMemory();
	m_lineBuffer.freeGPUMemory();
}

void CleanerCommand::createDisplay()
{
	freeGPUMemory();

	const std::vector <float>& xs = m_featuresCorrectedLocs["x"];
	const std::vector <float>& ys = m_featuresCorrectedLocs["y"];
	const std::vector <float>& zs = m_featuresCorrectedLocs.find("z") != m_featuresCorrectedLocs.end() ? m_featuresCorrectedLocs["z"] : std::vector <float>(xs.size(), 0.f);
	std::vector <poca::core::Vec3mf> points(xs.size());
	for (size_t n = 0; n < xs.size(); n++)
		points[n].set(xs[n], ys[n], zs[n]);

	const std::vector <float>& lengthes = m_featuresCorrectedLocs["lifetime"];
	const std::vector <float>& origXs = m_dset->getData("x");
	const std::vector <float>& origYs = m_dset->getData("y");
	const std::vector <float>& origZs = m_dset->dimension() == 3 ? m_dset->getData("z") : std::vector <float>(origXs.size(), 0.f);
	std::vector <poca::core::Vec3mf> lines;
	for (size_t n = 0; n < xs.size(); n++) {
		poca::core::Vec3mf barycenter(xs[n], ys[n], zs[n]);
		for (uint32_t currentLoc = m_firstsMerged[n]; currentLoc < m_firstsMerged[n + 1]; currentLoc++) {
			poca::core::Vec3mf point(poca::core::Vec3mf(origXs[m_mergedPoints[currentLoc]], origYs[m_mergedPoints[currentLoc]], origZs[m_mergedPoints[currentLoc]]));
			lines.push_back(barycenter);
			lines.push_back(point);
		}
	}

	m_pointBuffer.generateBuffer(points.size(), 512 * 512, 3, GL_FLOAT);
	m_lineBuffer.generateBuffer(lines.size(), 512 * 512, 3, GL_FLOAT);
	m_pointBuffer.updateBuffer(points.data());
	m_lineBuffer.updateBuffer(lines.data());
	m_initializeDisplay = false;
}

void CleanerCommand::display(poca::opengl::Camera* _cam, const bool _offscreen)
{
	if (!m_dset->isSelected()) return;

	if (m_initializeDisplay)
		createDisplay();

	if (m_pointBuffer.empty()) return;

	if (!hasParameter("displayCleanedLocs")) return;
	bool displayClean = getParameter<bool>("displayCleanedLocs");

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	_cam->drawUniformShader<poca::core::Vec3mf>(m_pointBuffer, poca::core::Color4D(0.f, 0.f, 0.f, 1.f));
	_cam->drawUniformShader<poca::core::Vec3mf>(m_lineBuffer, poca::core::Color4D(1.f, 0.f, 0.f, 1.f));
	glDisable(GL_BLEND);
}

