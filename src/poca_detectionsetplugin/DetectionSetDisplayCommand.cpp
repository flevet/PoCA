/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DetectionSetDisplayCommand.cpp
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
#include <QtGui/QOpenGLFramebufferObject>
#include <QtGui/QImage>
#include <QtGui/QOpenGLContextGroup>
#include <QtWidgets/QMessageBox>

#include <General/Palette.hpp>
#include <General/Histogram.hpp>
#include <General/MyData.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <Interfaces/CameraInterface.hpp>
#include <OpenGL/Shader.hpp>
#include <OpenGL/Helper.h>
#include <DesignPatterns/StateSoftwareSingleton.hpp>
#include <Cuda/Misc.h>

#include "DetectionSetDisplayCommand.hpp"


DetectionSetDisplayCommand::DetectionSetDisplayCommand(poca::geometry::DetectionSet* _ds) : poca::opengl::BasicDisplayCommand(_ds, "DetectionSetDisplayCommand"), m_textureLutID(0)
{
	m_dset = _ds;
	const poca::core::BoundingBox bbox = m_dset->boundingBox();

	poca::core::StateSoftwareSingleton* sss = poca::core::StateSoftwareSingleton::instance();
	const nlohmann::json& parameters = sss->getParameters();
	addCommandInfo(poca::core::CommandInfo(false, "pointRendering", true));
	addCommandInfo(poca::core::CommandInfo(false, "pointSizeGL", 6u));
	if (parameters.contains(name())) {
		nlohmann::json param = parameters[name()];
		if(param.contains("pointRendering"))
			loadParameters(poca::core::CommandInfo(false, "pointRendering", param["pointRendering"].get<bool>()));
		if (param.contains("pointSizeGL"))
			loadParameters(poca::core::CommandInfo(false, "pointSizeGL", param["pointSizeGL"].get<uint32_t>()));
	}
}

DetectionSetDisplayCommand::DetectionSetDisplayCommand(const DetectionSetDisplayCommand& _o) : poca::opengl::BasicDisplayCommand(_o)
{
	m_dset = _o.m_dset;
}

DetectionSetDisplayCommand::~DetectionSetDisplayCommand()
{
}

void DetectionSetDisplayCommand::execute(poca::core::CommandInfo* _infos)
{
	poca::opengl::BasicDisplayCommand::execute(_infos);
	if (_infos->nameCommand == "histogram" || _infos->nameCommand == "updateFeature") {
		generateFeatureBuffer();
	}
	if (_infos->nameCommand == "display") {
		poca::opengl::Camera* cam = _infos->getParameterPtr<poca::opengl::Camera>("camera");
		bool offscrean = false, ssao = false;
		if (_infos->hasParameter("offscreen"))
			offscrean = _infos->getParameter<bool>("offscreen");
		if (_infos->hasParameter("ssao"))
			ssao = _infos->getParameter<bool>("ssao");
		display(cam, offscrean, ssao);
	}
	else if (_infos->nameCommand == "changeLUT") {
		m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(m_dset->getPalette());
		if (_infos->hasParameter("regenerateFeatureBuffer"))
			generateFeatureBuffer();
	}
	if (_infos->nameCommand == "pick") {
		if (!m_dset->isSelected()) return;
		QString infos = getInfosLocalization(m_idSelection);
		if (infos.isEmpty()) return;
		poca::core::stringList listInfos = _infos->getParameter<poca::core::stringList>("infos");
		listInfos.push_back(infos.toLatin1().data());
		_infos->addParameter("infos", listInfos);
		if (m_idSelection >= 0 && _infos->hasParameter("pickedPoints")) {
			std::vector <poca::core::Vec3mf> pickedPoints = _infos->getParameter< std::vector <poca::core::Vec3mf>>("pickedPoints");
			float x = m_dset->getOriginalHistogram("x")->getValues()[m_idSelection];
			float y = m_dset->getOriginalHistogram("y")->getValues()[m_idSelection];
			float z = m_dset->dimension() == 3 ? m_dset->getOriginalHistogram("z")->getValues()[m_idSelection] : 0.f;
			pickedPoints.push_back(poca::core::Vec3mf(x, y, z));
			_infos->addParameter("pickedPoints", pickedPoints);
		}
	}
	else if (hasParameter(_infos->nameCommand)) {
		loadParameters(*_infos);
	}
	else if (_infos->nameCommand == "freeGPU") {
		freeGPUMemory();
	}
	else if (_infos->nameCommand == "sortWRTCameraPosition") {
		glm::vec3 cameraPosition = _infos->getParameter<glm::vec3>("cameraPosition");
		glm::vec3 cameraForward = _infos->getParameter<glm::vec3>("cameraForward");
		sortWrtCameraPosition(cameraPosition, cameraForward);
	}
	else if (_infos->nameCommand == "addNormals") {
		std::vector <poca::core::Vec3mf>* normals = _infos->getParameterPtr<std::vector <poca::core::Vec3mf>>("addNormals");
		m_normalBuffer.generateBuffer(normals->size(), 3, GL_FLOAT);
		m_normalBuffer.updateBuffer(normals->data());

		//Adding normals to the DetectionSet
		std::vector <float> nx(normals->size()), ny(normals->size()), nz(normals->size());
#pragma omp parallel for
		for (int n = 0; n < normals->size(); n++) {
			nx[n] = (*normals)[n].x();
			ny[n] = (*normals)[n].y();
			nz[n] = (*normals)[n].z();
		}
		m_dset->addFeature("nx", new poca::core::MyData(nx));
		m_dset->addFeature("ny", new poca::core::MyData(ny));
		m_dset->addFeature("nz", new poca::core::MyData(nz));

#ifdef DEBUG_NORMAL
		const std::vector<float> xs = m_dset->getOriginalHistogram("x")->getValues();
		const std::vector<float> ys = m_dset->getOriginalHistogram("y")->getValues();
		const std::vector<float> zs = m_dset->getOriginalHistogram("z")->getValues();

		std::vector <poca::core::Vec3mf> normalsDebug;
		for (size_t n = 0; n < xs.size(); n++) {
			poca::core::Vec3mf pt(xs[n], ys[n], zs[n]);
			normalsDebug.push_back(pt);
			normalsDebug.push_back(pt + 150.f * (*normals)[n]);
		}
		m_normalDebugBuffer.generateBuffer(normalsDebug.size(), 3, GL_FLOAT);
		m_normalDebugBuffer.updateBuffer(normalsDebug.data());
#endif
	}
}

poca::core::Command* DetectionSetDisplayCommand::copy()
{
	return new DetectionSetDisplayCommand(*this);
}

void DetectionSetDisplayCommand::display(poca::opengl::Camera* _cam, const bool _offscreen, const bool _ssao)
{
	if (!m_dset->isSelected()) return;

	bool pointRendering = getParameter<bool>("pointRendering");
	if (!pointRendering) return;

	drawElements(_cam, _ssao);
	if (!_offscreen)
		drawPicking(_cam); 
}

void DetectionSetDisplayCommand::drawElements(poca::opengl::Camera* _cam, const bool _ssao)
{
	GL_CHECK_ERRORS();
	if (m_pointBuffer.empty())
		createDisplay();

	GL_CHECK_ERRORS();
	glEnable(GL_DEPTH_TEST);
	GL_CHECK_ERRORS();
	glCullFace(GL_FRONT);
	GL_CHECK_ERRORS();

	uint32_t pointSize = getParameter<uint32_t>("pointSizeGL");
	if(m_normalBuffer.empty())
		_cam->drawSphereRendering<poca::core::Vec3mf, float>(m_textureLutID, m_pointBuffer, m_featureBuffer, m_minOriginalFeature, m_maxOriginalFeature, pointSize, _ssao);
	else
		_cam->drawSphereRendering<poca::core::Vec3mf, float>(m_textureLutID, m_pointBuffer, m_normalBuffer, m_featureBuffer, m_minOriginalFeature, m_maxOriginalFeature, pointSize, _ssao);

#ifdef DEBUG_NORMAL
	if (!m_normalDebugBuffer.empty())
		_cam->drawUniformShader(m_normalDebugBuffer, poca::core::Color4D(0.f, 0.f, 0.f, 1.f));
#endif
}

void DetectionSetDisplayCommand::drawPicking(poca::opengl::Camera* _cam)
{
	if (m_pickFBO == NULL) return;

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glDisable(GL_BLEND);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_CULL_FACE);

	GLfloat bkColor[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	bool success = m_pickFBO->bind();
	if (!success) std::cout << "Problem with binding" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	_cam->drawPickingShader<poca::core::Vec3mf, float>(m_pointBuffer, m_idBuffer, m_featureBuffer, m_minOriginalFeature);
	success = m_pickFBO->release();
	if (!success) std::cout << "Problem with releasing" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, _cam->defaultFramebufferObject());
	glClearColor(bkColor[0], bkColor[1], bkColor[2], bkColor[3]);
	GL_CHECK_ERRORS();
}

void DetectionSetDisplayCommand::createDisplay()
{
	GL_CHECK_ERRORS();
	freeGPUMemory();
	GL_CHECK_ERRORS();

	try {
		GL_CHECK_ERRORS();
		const std::vector <float>& xs = m_dset->getMyData("x")->getOriginalData(), & ys = m_dset->getMyData("y")->getOriginalData();
		m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(m_dset->getPalette());

		std::vector <poca::core::Vec3mf> points(xs.size());
		if (m_dset->hasData("z")) {
			const std::vector <float>& zs = m_dset->getMyData("z")->getOriginalData();
			for (size_t n = 0; n < xs.size(); n++)
				points[n].set(xs[n], ys[n], zs[n]);
		}
		else
			for (size_t n = 0; n < xs.size(); n++)
				points[n].set(xs[n], ys[n], 0.f);
		std::vector <float> ids(xs.size());
		std::iota(std::begin(ids), std::end(ids), 1);

		if (m_dset->hasData("nx") && m_dset->hasData("ny") && m_dset->hasData("nz")) {
			const std::vector <float>& nxs = m_dset->getMyData("nx")->getOriginalData(), & nys = m_dset->getMyData("ny")->getOriginalData(), & nzs = m_dset->getMyData("nz")->getOriginalData();
			std::vector <poca::core::Vec3mf> normals(nxs.size());
			for (size_t n = 0; n < nxs.size(); n++)
				normals[n].set(nxs[n], nys[n], nzs[n]);
			m_normalBuffer.generateBuffer(normals.size(), 3, GL_FLOAT);
			m_normalBuffer.updateBuffer(normals.data());
		}

		m_pointBuffer.generateBuffer(points.size(), 3, GL_FLOAT);
		m_idBuffer.generateBuffer(points.size(), 1, GL_FLOAT);
		m_featureBuffer.generateBuffer(points.size(), 1, GL_FLOAT);

		m_pointBuffer.updateBuffer(points.data());
		m_idBuffer.updateBuffer(ids.data());

		poca::core::HistogramInterface* hist = m_dset->getCurrentHistogram();
		generateFeatureBuffer(hist);
		GL_CHECK_ERRORS();
	}
	catch (std::runtime_error const& e) {
		std::string mess("Error: creating display for command " + name() + " of component " + m_dset->getName() + " failed with error message: " + e.what());
		QMessageBox msgBox;
		msgBox.setText(mess.c_str());
		msgBox.exec();
	}
}

void DetectionSetDisplayCommand::freeGPUMemory()
{
	if (m_textureLutID == 0) return;

	m_pointBuffer.freeGPUMemory();
	m_normalBuffer.freeGPUMemory();
	m_uncertaintiesBuffer.freeGPUMemory();
	m_idBuffer.freeGPUMemory();
	m_featureBuffer.freeGPUMemory();

	m_textureLutID = 0;
}

void DetectionSetDisplayCommand::generateFeatureBuffer(poca::core::HistogramInterface* _histogram)
{
	if (_histogram == NULL)
		_histogram = m_dset->getCurrentHistogram();
	const std::vector<float>& values = _histogram->getValues();
	const std::vector<bool>& selection = m_dset->getSelection();
	m_minOriginalFeature = _histogram->getMin();
	m_maxOriginalFeature = _histogram->getMax();
	m_actualValueFeature = m_maxOriginalFeature;

	std::vector <float> feature(values.size());
	for(size_t n = 0; n < values.size(); n++)
		feature[n] = selection[n] ? values[n] : -10000.f;
	m_featureBuffer.updateBuffer(feature);
}

QString DetectionSetDisplayCommand::getInfosLocalization(const int _id) const
{
	QString text;
	if (_id >= 0) {
		poca::core::stringList nameData = m_dset->getNameData();
		float x = m_dset->getOriginalHistogram("x")->getValues()[_id];
		float y = m_dset->getOriginalHistogram("y")->getValues()[_id];
		text.append(QString("Localization id: %1\n").arg(_id));
		text.append(QString("Coords: [x=%1,y=%2").arg(x).arg(y));
		if (m_dset->hasData("z")) {
			float z = m_dset->getOriginalHistogram("z")->getValues()[_id];
			text.append(QString(",z=%1").arg(z));
		}
		text.append("]");
		for (std::string type : nameData) {
			if (type == "x" || type == "y" || type == "z") continue;
			float val = m_dset->getOriginalHistogram(type)->getValues()[_id];
			text.append(QString("\n%1: %2").arg(type.c_str()).arg(val));
		}
	}
	return text;
}

void DetectionSetDisplayCommand::sortWrtCameraPosition(const glm::vec3& _cameraPosition, const glm::vec3& _cameraForwardVec)
{
	try {
		if (m_dset->dimension() == 2) return;

		const std::vector <float>& xs = m_dset->getMyData("x")->getOriginalData(), & ys = m_dset->getMyData("y")->getOriginalData(), & zs = m_dset->getMyData("z")->getOriginalData();
		std::vector <uint32_t> indices(xs.size());
		std::iota(std::begin(indices), std::end(indices), 0);

		//Compute a vector of distances of the points to the camera position
		std::vector <float> distances(xs.size());

#pragma omp parallel for
		for (int n = 0; n < xs.size(); n++)
			distances[n] = glm::dot(glm::vec3(xs[n], ys[n], zs[n]) - _cameraPosition, _cameraForwardVec);

		sortArrayWRTKeys(distances, indices);

		m_pointBuffer.updateIndices(indices);
		if (!m_normalBuffer.empty())
			m_normalBuffer.updateIndices(indices);
	}
	catch (std::runtime_error const& e) {
		std::string mess("Error: sorting localizations with respect to the camera position for component " + m_dset->getName() + " failed with error message: " + e.what());
		QMessageBox msgBox;
		msgBox.setText(mess.c_str());
		msgBox.exec();
	}
}

