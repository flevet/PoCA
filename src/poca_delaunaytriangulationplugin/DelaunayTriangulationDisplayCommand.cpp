/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DelaunayTriangulationDisplayCommand.cpp
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

#include <General/Palette.hpp>
#include <General/Histogram.hpp>
#include <General/MyData.hpp>
#include <General/Misc.h>
#include <Interfaces/HistogramInterface.hpp>
#include <Interfaces/CameraInterface.hpp>
#include <OpenGL/Shader.hpp>
#include <General/Engine.hpp>
#include <OpenGL/Helper.h>

#include "DelaunayTriangulationDisplayCommand.hpp"


DelaunayTriangulationDisplayCommand::DelaunayTriangulationDisplayCommand(poca::geometry::DelaunayTriangulationInterface* _delau) : poca::opengl::BasicDisplayCommand(_delau, "DelaunayTriangulationDisplayCommand"), m_textureLutID(0)
{
	m_delaunay = _delau;

	
	const nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
	addCommandInfo(poca::core::CommandInfo(false, "fill", true));
	addCommandInfo(poca::core::CommandInfo(false, "bboxSelection", true));
	if (parameters.contains(name())) {
		nlohmann::json param = parameters[name()];
		if(param.contains("fill"))
			loadParameters(poca::core::CommandInfo(false, "fill", param["fill"].get<bool>()));
		if (param.contains("bboxSelection"))
			loadParameters(poca::core::CommandInfo(false, "bboxSelection", param["bboxSelection"].get<bool>()));
	}
}


DelaunayTriangulationDisplayCommand::DelaunayTriangulationDisplayCommand(const DelaunayTriangulationDisplayCommand& _o) : poca::opengl::BasicDisplayCommand(_o)
{
	m_delaunay = _o.m_delaunay;
}

DelaunayTriangulationDisplayCommand::~DelaunayTriangulationDisplayCommand()
{
}

void DelaunayTriangulationDisplayCommand::execute(poca::core::CommandInfo* _infos)
{
	poca::opengl::BasicDisplayCommand::execute(_infos);
	if (_infos->nameCommand == "histogram" || _infos->nameCommand == "updateFeature") {
		generateFeatureBuffer();
	}
	else if (_infos->nameCommand == "display") {
		poca::opengl::Camera* cam = _infos->getParameterPtr<poca::opengl::Camera>("camera");
		bool offscrean = false;
		if (_infos->hasParameter("offscreen"))
			offscrean = _infos->getParameter<bool>("offscreen");
		display(cam, offscrean);
	}
	else if (_infos->nameCommand == "changeLUT") {
		m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(m_delaunay->getPalette());
		if (_infos->hasParameter("regenerateFeatureBuffer"))
			generateFeatureBuffer();
	}
	if (_infos->nameCommand == "pick") {
		if (!m_delaunay->isSelected()) return;
		QString infos = getInfosTriangle(m_idSelection);
		if (infos.isEmpty()) return;
		poca::core::stringList listInfos = _infos->getParameter< poca::core::stringList>("infos");
		listInfos.push_back(infos.toLatin1().data());
		_infos->addParameter("infos", listInfos);
		if (m_idSelection >= 0) {
			generateBoundingBoxSelection(m_idSelection);
			if (_infos->hasParameter("pickedPoints")) {
				std::vector <poca::core::Vec3mf> pickedPoints = _infos->getParameter< std::vector <poca::core::Vec3mf>>("pickedPoints");
				pickedPoints.push_back(m_delaunay->computeBarycenterElement(m_idSelection));
				_infos->addParameter("pickedPoints", pickedPoints);
			}
		}
	}
	else if (hasParameter(_infos->nameCommand)) {
		loadParameters(*_infos);
	}
	else if (_infos->nameCommand == "freeGPU") {
		freeGPUMemory();
	}
	else if (_infos->nameCommand == "selectedBorderTriangles") {
		std::vector <bool> selection(m_delaunay->nbFaces(), false);
		const poca::core::MyArrayUInt32& neighbors = m_delaunay->getNeighbors();
		for (auto n = 0; n < m_delaunay->nbFaces(); n++) {
			for (size_t i = 0; i < neighbors.nbElementsObject(n) && !selection[n]; i++) {
				uint32_t indexNeigh = neighbors.elementIObject(n, i);
				selection[n] = indexNeigh == std::numeric_limits<std::uint32_t>::max();
			}
		}
		m_delaunay->setSelection(selection);
		m_delaunay->executeCommand(false, "updateFeature");
	}
}

poca::core::CommandInfo DelaunayTriangulationDisplayCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "selectedBorderTriangles") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	return poca::core::CommandInfo();
}

poca::core::Command* DelaunayTriangulationDisplayCommand::copy()
{
	return new DelaunayTriangulationDisplayCommand(*this);
}

void DelaunayTriangulationDisplayCommand::display(poca::opengl::Camera* _cam, const bool _offscreen)
{
	if (!m_delaunay->isSelected()) return;

	drawElements(_cam);
	if (!_offscreen)
		drawPicking(_cam);
}

void DelaunayTriangulationDisplayCommand::drawElements(poca::opengl::Camera* _cam)
{
	if (m_triangleBuffer.empty())
		createDisplay();
	if (m_pickFBO == NULL)
		updatePickingFBO(_cam->getWidth(), _cam->getHeight());

	bool ok;
	bool fill = getParameter<bool>("fill");
	bool displayBboxSelection = getParameter<bool>("bboxSelection");
	bool cullFaceActivated = _cam->cullFaceActivated();

	glPolygonMode(GL_FRONT_AND_BACK, fill ? GL_FILL : GL_LINE);
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	if (cullFaceActivated)
		glEnable(GL_CULL_FACE);
	else
		glDisable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	_cam->drawSimpleShader<poca::core::Vec3mf, float>(m_textureLutID, m_triangleBuffer, m_featureBuffer, m_minOriginalFeature, m_maxOriginalFeature);
	GL_CHECK_ERRORS();

	glDisable(GL_DEPTH_TEST);
	if (m_idSelection >= 0 && displayBboxSelection) {
		GLfloat bkColor[4];
		glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
		poca::core::Color4D color = poca::core::contrastColor(poca::core::Color4D(bkColor[0] * 255.f, bkColor[1] * 255.f, bkColor[2] * 255.f, bkColor[3] * 255.f));
		_cam->drawUniformShader<poca::core::Vec3mf>(m_boundingBoxSelection, color);
	}
	GL_CHECK_ERRORS();
}

void DelaunayTriangulationDisplayCommand::drawPicking(poca::opengl::Camera* _cam)
{
	if (m_pickFBO == NULL)
		updatePickingFBO(_cam->getWidth(), _cam->getHeight());

	if (m_pickFBO == NULL) return;

	glEnable(GL_DEPTH_TEST);
	GLfloat bkColor[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	bool success = m_pickFBO->bind();
	if (!success) std::cout << "Problem with binding" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	_cam->drawPickingShader<poca::core::Vec3mf, float>(m_triangleBuffer, m_idBuffer, m_featureBuffer, m_minOriginalFeature);

	success = m_pickFBO->release();
	if (!success) std::cout << "Problem with releasing" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, _cam->defaultFramebufferObject());
	glClearColor(bkColor[0], bkColor[1], bkColor[2], bkColor[3]);
	GL_CHECK_ERRORS();
}

void DelaunayTriangulationDisplayCommand::createDisplay()
{
	freeGPUMemory();

	m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(m_delaunay->getPalette());

	std::vector <poca::core::Vec3mf> triangles;
	m_delaunay->generateTriangles(triangles);

	std::vector <float> ids;
	m_delaunay->generatePickingIndices(ids);

	m_triangleBuffer.generateBuffer(triangles.size(), 512 * 512, 3, GL_FLOAT);
	m_idBuffer.generateBuffer(triangles.size(), 512 * 512, 1, GL_FLOAT);
	m_featureBuffer.generateBuffer(triangles.size(), 512 * 512, 1, GL_FLOAT);

	m_triangleBuffer.updateBuffer(triangles.data());
	m_idBuffer.updateBuffer(ids.data());

	m_boundingBoxSelection.generateBuffer(24, 512 * 512, 3, GL_FLOAT);

	poca::core::HistogramInterface* hist = m_delaunay->getCurrentHistogram();
	generateFeatureBuffer(hist);
}

void DelaunayTriangulationDisplayCommand::freeGPUMemory()
{
	m_triangleBuffer.freeGPUMemory();
	m_idBuffer.freeGPUMemory();
	m_featureBuffer.freeGPUMemory();

	m_boundingBoxSelection.freeGPUMemory();
	m_neighborsSelection.freeGPUMemory();

	m_textureLutID = 0;
}

void DelaunayTriangulationDisplayCommand::generateFeatureBuffer(poca::core::HistogramInterface* _histogram)
{
	if (_histogram == NULL)
		_histogram = m_delaunay->getCurrentHistogram();
	poca::core::Histogram<float>* histogram = dynamic_cast<poca::core::Histogram<float>*>(_histogram);
	const std::vector<float>& values = histogram->getValues();
	const std::vector<bool>& selection = m_delaunay->getSelection();
	m_minOriginalFeature = _histogram->getMin();
	m_maxOriginalFeature = _histogram->getMax();
	m_actualValueFeature = m_maxOriginalFeature;

	std::vector <float> featureValues;
	m_delaunay->getFeatureInSelection(featureValues, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);

	m_featureBuffer.updateBuffer(featureValues.data());
}

QString DelaunayTriangulationDisplayCommand::getInfosTriangle(const int _id) const
{
	QString text;
	if (_id >= 0) {
		text.append(QString("Delaunay triangulation\n"));
		text.append(QString("Triangle id: %1").arg(_id));
		poca::core::stringList nameData = m_delaunay->getNameData();
		for (std::string type : nameData) {
			float val = m_delaunay->getMyData(type)->getData<float>()[_id];
			text.append(QString("\n%1: %2").arg(type.c_str()).arg(val));
		}
	}
	return text;
}

void DelaunayTriangulationDisplayCommand::generateBoundingBoxSelection(const int _idx)
{
	if (_idx < 0) return;
	poca::core::BoundingBox bbox = m_delaunay->computeBoundingBoxElement(_idx);

	std::vector <poca::core::Vec3mf> cube(24);
	poca::geometry::createCubeFromVector(cube, bbox);
	m_boundingBoxSelection.updateBuffer(cube.data());

	const float* xs = m_delaunay->getXs();
	const float* ys = m_delaunay->getYs();
	const float* zs = m_delaunay->getZs();
	const poca::core::MyArrayUInt32& neighbors = m_delaunay->getNeighbors();

	std::vector <poca::core::Vec3mf> lines;
	poca::core::Vec3mf centroid = m_delaunay->computeBarycenterElement(_idx);
	uint32_t realNbNeighs = 0;
	for (size_t i = 0; i < neighbors.nbElementsObject(_idx); i++) {
		uint32_t indexNeigh = neighbors.elementIObject(_idx, i);
		if (indexNeigh == std::numeric_limits<std::uint32_t>::max()) continue;
		poca::core::Vec3mf centroidOther = m_delaunay->computeBarycenterElement(indexNeigh);

		lines.push_back(centroid);
		lines.push_back(centroidOther);
		realNbNeighs++;
	}
	m_neighborsSelection.freeGPUMemory();
	m_neighborsSelection.generateBuffer(2 * realNbNeighs, 512 * 512, 3, GL_FLOAT);
	m_neighborsSelection.updateBuffer(lines.data());
}

