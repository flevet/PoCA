/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      GaussianDisplayCommand.cpp
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
#include <glm/gtx/string_cast.hpp>
#include <algorithm>

#include <QtGui/QOpenGLFramebufferObject>
#include <QtWidgets/QMessageBox>

#include <General/Palette.hpp>
#include <General/MyData.hpp>
#include <General/Histogram.hpp>
#include <OpenGL/Shader.hpp>
#include <OpenGL/Helper.h>
#include <General/Engine.hpp>

#include "GaussianDisplayCommand.hpp"
#include "DetectionSetDisplayCommand.hpp"


GaussianDisplayCommand::GaussianDisplayCommand(poca::geometry::DetectionSet* _ds) :poca::core::Command("GaussianDisplayCommand"), m_dc(NULL)
{
	m_dset = _ds;

	
	const nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
	addCommandInfo(poca::core::CommandInfo(false, "displayGaussian", false));
	addCommandInfo(poca::core::CommandInfo(false, "pointSizeGL", 25u));
	addCommandInfo(poca::core::CommandInfo(false, "alphaGaussian", 0.1f));
	addCommandInfo(poca::core::CommandInfo(false, "fixedSizeGaussian", true));
	if (parameters.contains(name())) {
		nlohmann::json param = parameters[name()];
		if(param.contains("displayGaussian"))
			loadParameters(poca::core::CommandInfo(false, "displayGaussian", param["displayGaussian"].get<bool>()));
		if (param.contains("pointSizeGL"))
			loadParameters(poca::core::CommandInfo(false, "pointSizeGL", param["pointSizeGL"].get<uint32_t>()));
		if (param.contains("alphaGaussian"))
			loadParameters(poca::core::CommandInfo(false, "alphaGaussian", param["alphaGaussian"].get<float>()));
		if (param.contains("fixedSizeGaussian"))
			loadParameters(poca::core::CommandInfo(false, "fixedSizeGaussian", param["fixedSizeGaussian"].get<bool>()));
	}
}

GaussianDisplayCommand::GaussianDisplayCommand(const GaussianDisplayCommand& _o) : poca::core::Command(_o)
{
	m_dset = _o.m_dset;
}

GaussianDisplayCommand::~GaussianDisplayCommand()
{
	freeGPUMemory();
}

poca::core::Command* GaussianDisplayCommand::copy()
{
	return new GaussianDisplayCommand(*this);
}

void GaussianDisplayCommand::execute(poca::core::CommandInfo* _infos)
{
	if ((_infos->nameCommand == "histogram" || _infos->nameCommand == "updateFeature") && !m_dc) {
		generateFeatureBuffer();
	}
	else if (_infos->nameCommand == "display") {
		poca::opengl::Camera* cam = _infos->getParameterPtr<poca::opengl::Camera>("camera");
		bool offscrean = false;
		if (_infos->hasParameter("offscreen"))
			offscrean = _infos->getParameter<bool>("offscreen");
		display(cam, offscrean);
	}
	else if (hasCommand(_infos->nameCommand)) {
		loadParameters(*_infos);
	}
	else if (_infos->nameCommand == "sortWRTCameraPosition" && !m_dc) {
		glm::vec3 cameraPosition = _infos->getParameter<glm::vec3>("cameraPosition");
		glm::vec3 cameraForward = _infos->getParameter<glm::vec3>("cameraForward");
		sortWrtCameraPosition(cameraPosition, cameraForward);
	}
	else if (_infos->nameCommand == "changeLUT")
		m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(m_dset->getPalette());
}

poca::core::CommandInfo GaussianDisplayCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "displayGaussian") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "pointSizeGL" || _nameCommand == "alphaGaussian") {
		float val = _parameters.get<float>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	return poca::core::CommandInfo();
}

void GaussianDisplayCommand::createDisplay()
{
	freeGPUMemory();

	try {
		m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(m_dset->getPalette());


		m_dc = m_dset->getCommand<DetectionSetDisplayCommand>();
		if (!m_dc) {
			const std::vector <float>& xs = m_dset->getMyData("x")->getData<float>(), & ys = m_dset->getMyData("y")->getData<float>();
			std::vector <poca::core::Vec3mf> points(xs.size());
			if (m_dset->hasData("z")) {
				const std::vector <float>& zs = m_dset->getMyData("z")->getData<float>();
				for (size_t n = 0; n < xs.size(); n++)
					points[n].set(xs[n], ys[n], zs[n]);
			}
			else
				for (size_t n = 0; n < xs.size(); n++)
					points[n].set(xs[n], ys[n], 0.f);

			std::vector <poca::core::Vec3mf> sigmas;
			if (m_dset->hasData(m_nameSigmaXY)) {
				const std::vector <float>& sigXY = m_dset->getMyData(m_nameSigmaXY)->getData<float>();
				const std::vector <float>& sigZ = m_dset->dimension() == 3 && m_dset->hasData(m_nameSigmaZ) ? m_dset->getMyData(m_nameSigmaZ)->getData<float>() : std::vector <float>();
				sigmas.resize(sigXY.size());
				for (size_t n = 0; n < xs.size(); n++)
					sigmas[n].set(3.f * sigXY[n], 3.f * sigXY[n], sigZ.empty() ? 3.f * sigXY[n] : 3.f * sigZ[n]);
			}
	
			m_pointBuffer.generateBuffer(points.size(), 3, GL_FLOAT);
			m_featureBuffer.generateBuffer(points.size(), 1, GL_FLOAT);
			m_pointBuffer.updateBuffer(points.data());

			if (!sigmas.empty()) {
				m_uncertaintiesBuffer.generateBuffer(sigmas.size(), 3, GL_FLOAT);
				m_uncertaintiesBuffer.updateBuffer(sigmas.data());
			}

			poca::core::HistogramInterface* hist = m_dset->getCurrentHistogram();
			generateFeatureBuffer(hist);
		}
		GL_CHECK_ERRORS();
	}
	catch (std::runtime_error const& e) {
		std::string mess("Error: creating display for command " + name() + " of component " + m_dset->getName() + " failed with error message: " + e.what());
		QMessageBox msgBox;
		msgBox.setText(mess.c_str());
		msgBox.exec();
	}
}

void GaussianDisplayCommand::freeGPUMemory()
{
	m_pointBuffer.freeGPUMemory();
	m_featureBuffer.freeGPUMemory();
	m_uncertaintiesBuffer.freeGPUMemory();
}

void GaussianDisplayCommand::display(poca::opengl::Camera* _cam, const bool _offscreen)
{
	if (!m_dset->isSelected()) return;

	if (m_pointBuffer.empty() && m_dc == NULL)
		createDisplay();

	bool displayGaussian = getParameter<bool>("displayGaussian");
	uint32_t pointSize = getParameter<uint32_t>("pointSizeGL");
	float alpha = getParameter<float>("alphaGaussian");
	bool fixedSize = getParameter<bool>("fixedSizeGaussian");

	glDisable(GL_CULL_FACE);

	if (!displayGaussian) return;

	if(!m_dc)
		_cam->draw2DGaussianRendering<poca::core::Vec3mf, float>(m_textureLutID, m_pointBuffer, m_uncertaintiesBuffer, m_featureBuffer, m_minOriginalFeature, m_maxOriginalFeature, alpha, 3 * pointSize, fixedSize);
	else
		_cam->draw2DGaussianRendering<poca::core::Vec3mf, float>(m_textureLutID, m_dc->getPointBuffer(), m_dc->getUncertaintiesBuffer(), m_dc->getFeatureBuffer(), m_dc->getMinOriginalFeature(), m_dc->getMaxOriginalFeature(), alpha, 3 * pointSize, fixedSize);

	GL_CHECK_ERRORS(); 
}

void GaussianDisplayCommand::generateFeatureBuffer(poca::core::HistogramInterface* _histogram)
{
	if (_histogram == NULL)
		_histogram = m_dset->getCurrentHistogram();
	poca::core::Histogram<float>* histogram = dynamic_cast <poca::core::Histogram<float>*>(_histogram);
	const std::vector<float>& values = histogram->getValues();
	const std::vector<bool>& selection = m_dset->getSelection();
	m_minOriginalFeature = histogram->getMin();
	m_maxOriginalFeature = histogram->getMax();
	m_actualValueFeature = m_maxOriginalFeature;

	std::vector <float> feature(values.size());
	for (size_t n = 0; n < values.size(); n++)
		feature[n] = selection[n] ? values[n] : -10000.f;// MIN_VALUE_FEATURE_SHADER;
	m_featureBuffer.updateBuffer(feature);
}

void GaussianDisplayCommand::sortWrtCameraPosition(const glm::vec3& _cameraPosition, const glm::vec3& _cameraForwardVec)
{
	try {
		if (m_dset->dimension() == 2) return;
		const std::vector <float>& xs = m_dset->getMyData("x")->getData<float>(), & ys = m_dset->getMyData("y")->getData<float>(), & zs = m_dset->getMyData("z")->getData<float>();
		std::vector <uint32_t> indices(xs.size());
		std::iota(std::begin(indices), std::end(indices), 0);

		//Compute a vector of distances of the points to the camera position
		std::vector <float> distances(xs.size());
		for (size_t n = 0; n < xs.size(); n++)
			distances[n] = glm::dot(glm::vec3(xs[n], ys[n], zs[n]) - _cameraPosition, _cameraForwardVec);
		//Sort wrt the distance to the camera position
		std::sort(indices.begin(), indices.end(),
			[&](int A, int B) -> bool {
				return distances[A] < distances[B];
			});
		m_pointBuffer.updateIndices(indices);
	}
	catch (std::runtime_error const& e) {
		std::string mess("Error: sorting localizations with respect to the camera position for component " + m_dset->getName() + " failed with error message: " + e.what());
		QMessageBox msgBox;
		msgBox.setText(mess.c_str());
		msgBox.exec();
	}
}

