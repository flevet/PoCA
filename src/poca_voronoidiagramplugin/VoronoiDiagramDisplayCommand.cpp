/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagramDisplayCommand.cpp
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
#include <DesignPatterns/StateSoftwareSingleton.hpp>
#include <OpenGL/Helper.h>

#include "VoronoiDiagramDisplayCommand.hpp"


VoronoiDiagramDisplayCommand::VoronoiDiagramDisplayCommand(poca::geometry::VoronoiDiagram* _voro) : poca::opengl::BasicDisplayCommand(_voro, "VoronoiDiagramDisplayCommand"), m_displayTriangleSelection(false), m_textureLutID(0)//, m_fill(true), m_locsDisplay(true), m_polytopesDisplay(true)
{
	m_voronoi = _voro;

	poca::core::StateSoftwareSingleton* sss = poca::core::StateSoftwareSingleton::instance();
	const nlohmann::json& parameters = sss->getParameters();
	addCommandInfo(poca::core::CommandInfo(false, "fill", true));
	addCommandInfo(poca::core::CommandInfo(false, "pointRendering", true));
	addCommandInfo(poca::core::CommandInfo(false, "polytopeRendering", true));
	addCommandInfo(poca::core::CommandInfo(false, "bboxSelection", true));
	addCommandInfo(poca::core::CommandInfo(false, "pointSizeGL", 6u));
	if (parameters.contains(name())) {
		nlohmann::json param = parameters[name()];
		if(param.contains("fill"))
			loadParameters(poca::core::CommandInfo(false, "fill", param["fill"].get<bool>()));
		if (param.contains("pointRendering"))
			loadParameters(poca::core::CommandInfo(false, "pointRendering", param["pointRendering"].get<bool>()));
		if (param.contains("polytopeRendering"))
			loadParameters(poca::core::CommandInfo(false, "polytopeRendering", param["polytopeRendering"].get<bool>()));
		if (param.contains("bboxSelection"))
			loadParameters(poca::core::CommandInfo(false, "bboxSelection", param["bboxSelection"].get<bool>()));
		if (param.contains("pointSizeGL"))
			loadParameters(poca::core::CommandInfo(false, "pointSizeGL", param["pointSizeGL"].get<uint32_t>()));
	}
}

VoronoiDiagramDisplayCommand::VoronoiDiagramDisplayCommand(const VoronoiDiagramDisplayCommand& _o) : poca::opengl::BasicDisplayCommand(_o)//, m_fill(_o.m_fill)
{
	m_voronoi = _o.m_voronoi;
}

VoronoiDiagramDisplayCommand::~VoronoiDiagramDisplayCommand()
{
}

void VoronoiDiagramDisplayCommand::execute(poca::core::CommandInfo* _infos)
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
		m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(m_voronoi->getPalette());
		if (_infos->hasParameter("regenerateFeatureBuffer"))
			generateFeatureBuffer();
	}
	else if (_infos->nameCommand == "pick") {
		if (!m_voronoi->isSelected()) return;
		QString infos = getInfosTriangle(m_idSelection);
		if (infos.isEmpty()) return;
		poca::core::stringList listInfos = _infos->getParameter<poca::core::stringList>("infos");
		listInfos.push_back(infos.toLatin1().data());
		_infos->addParameter("infos", listInfos);
		if (m_idSelection >= 0) {
			generateBoundingBoxSelection(m_idSelection);
			if (_infos->hasParameter("pickedPoints") && m_voronoi->hasCells()) {
				std::vector <poca::core::Vec3mf> pickedPoints = _infos->getParameter< std::vector <poca::core::Vec3mf>>("pickedPoints");
				pickedPoints.push_back(m_voronoi->computeBarycenterElement(m_idSelection));
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
	else if (_infos->nameCommand == "determineTrianglesLinkedToPoint") {
		if (!m_voronoi->isSelected()) {
			m_displayTriangleSelection = false;
			return;
		}
		float x = _infos->getParameter<float>("x");
		float y = _infos->getParameter<float>("y");
		float z = _infos->getParameter<float>("z");

		std::vector <poca::core::Vec3mf> triangle;

		size_t indexTriangle = m_voronoi->indexTriangleOfPoint(x, y, z);
		if (indexTriangle == std::numeric_limits<std::size_t>::max()) {
			m_displayTriangleSelection = false;
			return;
		}
	}
	else if (_infos->nameCommand == "explodeCells") {
		float factor = 1.f;
		if (_infos->hasParameter("factor"))
			factor = _infos->getParameter<float>("factor");
		explodeDiagram(factor);
	}
}

poca::core::CommandInfo VoronoiDiagramDisplayCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "fill" || _nameCommand == "pointRendering" || _nameCommand == "polytopeRendering" || _nameCommand == "bboxSelection") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "determineTrianglesLinkedToPoint") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	else if (_nameCommand == "explodeCells") {
		float factor = 1.f;
		poca::core::CommandInfo ci(false, _nameCommand);
		ci.addParameter("factor", _parameters.contains("factor") ? _parameters["factor"].get<float>() : 1.f);
		return ci;
	}
	return poca::opengl::BasicDisplayCommand::createCommand(_nameCommand, _parameters);
}

poca::core::Command* VoronoiDiagramDisplayCommand::copy()
{
	return new VoronoiDiagramDisplayCommand(*this);
}

void VoronoiDiagramDisplayCommand::display(poca::opengl::Camera* _cam, const bool _offscreen)
{
	if (!m_voronoi->isSelected()) return;

	drawElements(_cam);
	if (!_offscreen)
		drawPicking(_cam);
}

void VoronoiDiagramDisplayCommand::drawElements(poca::opengl::Camera* _cam)
{
	GL_CHECK_ERRORS();
	if (m_pointBuffer.empty())
		createDisplay();
	GL_CHECK_ERRORS();

	bool pointRendering = getParameter<bool>("pointRendering");
	bool polytopeRendering = getParameter<bool>("polytopeRendering");
	bool fill = getParameter<bool>("fill");
	bool displayBboxSelection = getParameter<bool>("bboxSelection");
	uint32_t pointSize = getParameter<uint32_t>("pointSizeGL");
	bool cullFaceActivated = _cam->cullFaceActivated();
	GL_CHECK_ERRORS();

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	GL_CHECK_ERRORS();

	if (pointRendering) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		_cam->drawSphereRendering<poca::core::Vec3mf, float>(m_textureLutID, m_pointBuffer, m_locsFeatureBuffer, m_minOriginalFeature, m_maxOriginalFeature, pointSize, false);
	}
	GL_CHECK_ERRORS();
	if (polytopeRendering && m_voronoi->hasCells()) {
		glPolygonMode(GL_FRONT_AND_BACK, fill ? GL_FILL : GL_LINE);
		if (cullFaceActivated)
			glEnable(GL_CULL_FACE);
		else
			glDisable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		if (m_voronoi->dimension() == 3)
			_cam->drawSimpleShader<poca::core::Vec3mf, float>(m_textureLutID, m_triangleBuffer, m_triangleFeatureBuffer, m_minOriginalFeature, m_maxOriginalFeature);
		else {
			if (fill) {
				_cam->drawSimpleShader<poca::core::Vec3mf, float>(m_textureLutID, m_triangleBuffer, m_triangleFeatureBuffer, m_minOriginalFeature, m_maxOriginalFeature);
				//glDisable(GL_DEPTH_TEST);
				//_cam->drawLineShader<poca::core::Vec3mf>(m_lineBuffer, m_lineFeatureBuffer, poca::core::Color4D(0.f, 0.f, 0.f, 1.f), m_lineNormalBuffer, m_minOriginalFeature);
			}
			else {
				glDisable(GL_DEPTH_TEST);
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				_cam->drawLineShader<poca::core::Vec3mf, float>(m_textureLutID, m_lineBuffer, m_lineFeatureBuffer, m_lineNormalBuffer, m_minOriginalFeature, m_maxOriginalFeature, 1.f);
			}
		}
	}
	GL_CHECK_ERRORS();

	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	
	GL_CHECK_ERRORS();
	if (m_idSelection >= 0 && polytopeRendering && m_voronoi->hasCells() && displayBboxSelection) {
		GLfloat bkColor[4];
		glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
		poca::core::Color4D color = poca::core::contrastColor(poca::core::Color4D(bkColor[0] * 255.f, bkColor[1] * 255.f, bkColor[2] * 255.f, bkColor[3] * 255.f));
		_cam->drawUniformShader<poca::core::Vec3mf>(m_boundingBoxSelection, color);
	}
	GL_CHECK_ERRORS();

	if (m_displayTriangleSelection) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		_cam->drawUniformShader<poca::core::Vec3mf>(m_triangleSelection, poca::core::Color4D(1.f, 0.f, 0.f, 1.f));
	}
	GL_CHECK_ERRORS();
}

void VoronoiDiagramDisplayCommand::drawPicking(poca::opengl::Camera* _cam)
{
	GL_CHECK_ERRORS();
	if (m_pickFBO == NULL)
		updatePickingFBO(_cam->getWidth(), _cam->getHeight());
	GL_CHECK_ERRORS();

	if (m_pickFBO == NULL) return;
	GL_CHECK_ERRORS();

	glEnable(GL_DEPTH_TEST);
	GLfloat bkColor[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	bool success = m_pickFBO->bind();
	if (!success) std::cout << "Problem with binding" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (m_voronoi->hasCells())
		_cam->drawPickingShader<poca::core::Vec3mf, float>(m_triangleBuffer, m_idPolytopeBuffer, m_triangleFeatureBuffer, m_minOriginalFeature);
	else
		_cam->drawPickingShader<poca::core::Vec3mf, float>(m_pointBuffer, m_idLocsBuffer, m_locsFeatureBuffer, m_minOriginalFeature);

	success = m_pickFBO->release();
	if (!success) std::cout << "Problem with releasing" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, _cam->defaultFramebufferObject());
	glClearColor(bkColor[0], bkColor[1], bkColor[2], bkColor[3]);
	GL_CHECK_ERRORS();
}

void VoronoiDiagramDisplayCommand::createDisplay()
{
	freeGPUMemory();

	m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(m_voronoi->getPalette());

	//For locs
	std::vector <poca::core::Vec3mf> points(m_voronoi->nbFaces());
	const float* xs = m_voronoi->getXs(), * ys = m_voronoi->getYs(), * zs = m_voronoi->getZs();
	if (zs != NULL)
		for (size_t n = 0; n < m_voronoi->nbFaces(); n++)
			points[n].set(xs[n], ys[n], zs[n]);
	else
		for (size_t n = 0; n < m_voronoi->nbFaces(); n++)
			points[n].set(xs[n], ys[n], 0.f);
	std::vector <float> ids(m_voronoi->nbFaces());
	std::iota(std::begin(ids), std::end(ids), 1);
	m_pointBuffer.generateBuffer(points.size(), 3, GL_FLOAT);
	m_idLocsBuffer.generateBuffer(points.size(), 1, GL_FLOAT);
	m_locsFeatureBuffer.generateBuffer(points.size(), 1, GL_FLOAT);
	m_pointBuffer.updateBuffer(points.data());
	m_idLocsBuffer.updateBuffer(ids.data());

	//For Voronoi polytopes
	if (m_voronoi->hasCells()) {
		std::vector <poca::core::Vec3mf> triangles;
		m_voronoi->generateTriangles(triangles);
		std::vector <float> ids;
		m_voronoi->generatePickingIndices(ids);
		m_triangleBuffer.generateBuffer(triangles.size(), 512 * 512, 3, GL_FLOAT);
		m_triangleFeatureBuffer.generateBuffer(triangles.size(), 512 * 512, 1, GL_FLOAT);
		m_idPolytopeBuffer.generateBuffer(triangles.size(), 512 * 512, 1, GL_FLOAT);
		m_triangleBuffer.updateBuffer(triangles.data());
		m_idPolytopeBuffer.updateBuffer(ids.data());

		if (m_voronoi->dimension() == 2) {
			std::vector <poca::core::Vec3mf> lines, normals;
			m_voronoi->generateLines(lines);
			m_voronoi->generateLinesNormals(normals);
			m_lineBuffer.generateBuffer(lines.size(), 3, GL_FLOAT);
			m_lineFeatureBuffer.generateBuffer(lines.size(), 1, GL_FLOAT);
			m_lineBuffer.updateBuffer(lines.data());
			if (!normals.empty()) {
				m_lineNormalBuffer.generateBuffer(normals.size(), 3, GL_FLOAT);
				m_lineNormalBuffer.updateBuffer(normals.data());
			}
		}

		m_boundingBoxSelection.generateBuffer(24, 512 * 512, 3, GL_FLOAT);
	}

	m_triangleSelection.generateBuffer(3, 512 * 512, 3, GL_FLOAT);

	poca::core::HistogramInterface* hist = m_voronoi->getCurrentHistogram();
	generateFeatureBuffer(hist);
	GL_CHECK_ERRORS();
}

void VoronoiDiagramDisplayCommand::freeGPUMemory()
{
	if (m_textureLutID == 0) return;

	m_pointBuffer.freeGPUMemory();
	m_idLocsBuffer.freeGPUMemory();
	m_locsFeatureBuffer.freeGPUMemory();

	m_triangleBuffer.freeGPUMemory();
	m_lineBuffer.freeGPUMemory();
	m_idPolytopeBuffer.freeGPUMemory();
	m_triangleFeatureBuffer.freeGPUMemory();
	m_lineFeatureBuffer.freeGPUMemory();

	m_boundingBoxSelection.freeGPUMemory();
	m_triangleSelection.freeGPUMemory();

	m_textureLutID = 0;
}

void VoronoiDiagramDisplayCommand::generateFeatureBuffer(poca::core::HistogramInterface* _histogram)
{
	if (m_locsFeatureBuffer.empty()) return;

	if (_histogram == NULL)
		_histogram = m_voronoi->getCurrentHistogram();
	const std::vector<float>& values = _histogram->getValues();
	const std::vector<bool>& selection = m_voronoi->getSelection();
	m_minOriginalFeature = _histogram->getMin();
	m_maxOriginalFeature = _histogram->getMax();
	m_actualValueFeature = m_maxOriginalFeature;

	std::vector<float> featureLocs(m_voronoi->nbFaces());
	for (size_t n = 0; n < m_voronoi->nbFaces(); n++)
		featureLocs[n] = selection[n] ? values[n] : poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER;
	m_locsFeatureBuffer.updateBuffer(featureLocs.data());

	if (m_voronoi->hasCells()) {
		std::vector <float> featureValues;
		m_voronoi->getFeatureInSelection(featureValues, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER, false);
		m_triangleFeatureBuffer.updateBuffer(featureValues.data());
		if (m_voronoi->dimension() == 2) {
			m_voronoi->getFeatureInSelection(featureValues, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER, true);
			m_lineFeatureBuffer.updateBuffer(featureValues.data());
		}
	}
}

QString VoronoiDiagramDisplayCommand::getInfosTriangle(const int _id) const
{
	QString text;
	if (_id >= 0) {
		text.append(QString("Voronoi diagram id: %1").arg(_id));
		poca::core::stringList nameData = m_voronoi->getNameData();
		for (std::string type : nameData) {
			float val = m_voronoi->getOriginalHistogram(type)->getValues()[_id];
			text.append(QString("\n%1: %2").arg(type.c_str()).arg(val));
		}
	}
	return text;
}

void VoronoiDiagramDisplayCommand::generateBoundingBoxSelection(const int _idx)
{
	if (_idx < 0 || !m_voronoi->hasCells()) return;
	poca::core::BoundingBox bbox = m_voronoi->computeBoundingBoxElement(_idx);

	std::vector <poca::core::Vec3mf> cube(24);
	poca::geometry::createCubeFromVector(cube, bbox);
	m_boundingBoxSelection.updateBuffer(cube.data());
}

void VoronoiDiagramDisplayCommand::explodeDiagram(const float _factor)
{
	const poca::core::MyArrayUInt32& neighbors = m_voronoi->getNeighbors();
	const poca::core::BoundingBox& bboxV = m_voronoi->boundingBox();
	poca::core::Vec3mf centroidDiagram(bboxV[0] + (bboxV[3] - bboxV[0]) / 2.f, bboxV[1] + (bboxV[4] - bboxV[1]) / 2.f, bboxV[2] + (bboxV[5] - bboxV[2]) / 2.f);

	poca::geometry::VoronoiDiagram2D* v2D = dynamic_cast <poca::geometry::VoronoiDiagram2D*>(m_voronoi);
	if (v2D) {

	}
	poca::geometry::VoronoiDiagram3D* v3D = dynamic_cast <poca::geometry::VoronoiDiagram3D*>(m_voronoi);
	if (v3D) {
		if (!v3D->hasCells()) return;
		const std::vector <poca::core::Vec3mf>& cells = v3D->getCells();
		const std::vector <uint32_t>& firsts = v3D->getFirstCells();

		std::vector <poca::core::Vec3mf> explodedCells(cells.size());

		for (auto idx = 0; idx < v3D->nbFaces(); idx++) {
			poca::core::Vec3mf centroid(0.f, 0.f, 0.f);
			float nbs = firsts[idx + 1] - firsts[idx];
			for (unsigned int n = firsts[idx]; n < firsts[idx + 1]; n++)
				centroid += cells[n] / nbs;
			poca::core::Vec3mf vector = (_factor - 1.f) * (centroid - centroidDiagram);
			for (unsigned int n = firsts[idx]; n < firsts[idx + 1]; n++) {
				explodedCells[n] = cells[n] + vector;
			}
		}
		m_triangleBuffer.updateBuffer(explodedCells.data());
	}
}

