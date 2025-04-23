/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectColocalizationCommands.cpp
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

#include <Windows.h>
#include <gl/glew.h>
#include <gl/GL.h>
#include <QtGui/QOpenGLFramebufferObject>

#include <Geometry/ObjectLists.hpp>
#include <Geometry/DelaunayTriangulation.hpp>
#include <OpenGL/Shader.hpp>
#include <General/Engine.hpp>
#include <General/Misc.h>
#include <General/Histogram.hpp>
#include <OpenGL/Helper.h>

#include "ObjectColocalizationCommands.hpp"
#include "ObjectColocalization.hpp"

ObjectColocalizationCommands::ObjectColocalizationCommands(ObjectColocalization* _coloc) :poca::opengl::BasicDisplayCommand(_coloc, "ObjectColocalizationCommands")
{
	m_colocalization = _coloc;

	
	const nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
	addCommandInfo(poca::core::CommandInfo(false, "pointRendering", true));
	addCommandInfo(poca::core::CommandInfo(false, "shapeRendering", true));
	addCommandInfo(poca::core::CommandInfo(false, "bboxSelection", true));
	addCommandInfo(poca::core::CommandInfo(false, "fill", true));

	if (parameters.contains(name())) {
		nlohmann::json param = parameters[name()];
		if (param.contains("fill"))
			loadParameters(poca::core::CommandInfo(false, "fill", param["fill"].get<bool>()));
		if (param.contains("pointRendering"))
			loadParameters(poca::core::CommandInfo(false, "pointRendering", param["pointRendering"].get<bool>()));
		if (param.contains("shapeRendering"))
			loadParameters(poca::core::CommandInfo(false, "shapeRendering", param["shapeRendering"].get<bool>()));
		if (param.contains("bboxSelection"))
			loadParameters(poca::core::CommandInfo(false, "bboxSelection", param["bboxSelection"].get<bool>()));
	}

}

ObjectColocalizationCommands::ObjectColocalizationCommands(const ObjectColocalizationCommands& _o) : poca::opengl::BasicDisplayCommand(_o)
{
	m_colocalization = _o.m_colocalization;
}

ObjectColocalizationCommands::~ObjectColocalizationCommands()
{
}

void ObjectColocalizationCommands::execute(poca::core::CommandInfo* _infos)
{
	poca::opengl::BasicDisplayCommand::execute(_infos);
	if (_infos->nameCommand == "display") {
		poca::opengl::Camera* cam = _infos->getParameterPtr<poca::opengl::Camera>("camera");
		bool offscrean = false;
		if (_infos->hasParameter("offscreen"))
			offscrean = _infos->getParameter<bool>("offscreen");
		display(cam, offscrean);
	}
	else if (_infos->nameCommand == "histogram" || _infos->nameCommand == "updateFeature") {
		//updateDisplay();
		generateFeatureBuffer();
	}
	else if (hasParameter(_infos->nameCommand)) {
		loadParameters(*_infos);
	}
	else if (_infos->nameCommand == "freeGPU") {
		freeGPUMemory();
	}
	else if (_infos->nameCommand == "pick") {
		if (!m_colocalization->isSelected()) return;
		QString infos = getInfosTriangle(m_idSelection);
		if (infos.isEmpty()) return;
		poca::core::stringList listInfos = _infos->getParameter<poca::core::stringList>("infos");
		listInfos.push_back(infos.toLatin1().data());
		_infos->addParameter("infos", listInfos);
		if (m_idSelection >= 0) {
			generateBoundingBoxSelection(m_idSelection);
		}
	}
	else if (_infos->nameCommand == "doubleClickCamera") {
		size_t idSelection = m_idSelection;
		//if (_infos->hasParameter("objectID"))
		//	idSelection = _infos->getParameter<size_t>("objectID");
		if (m_colocalization->getObjectsOverlap() == NULL) return;
		if (idSelection >= 0 && idSelection < m_colocalization->getObjectsOverlap()->nbElements()) {
			poca::core::BoundingBox bbox = m_colocalization->computeBoundingBoxElement(idSelection);
			if (_infos->hasParameter("bbox")) {
				poca::core::BoundingBox bbox2 = _infos->getParameter<poca::core::BoundingBox>("bbox");
				for (size_t n = 0; n < 3; n++)
					bbox[n] = bbox[n] < bbox2[n] ? bbox[n] : bbox2[n];
				for (size_t n = 3; n < 6; n++)
					bbox[n] = bbox[n] > bbox2[n] ? bbox[n] : bbox2[n];
			}
			_infos->addParameter("bbox", bbox);
		}
	}
	else if (_infos->nameCommand == "changeLUT") {
		//generateLutTexture();
		m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(m_colocalization->getObjectsOverlap()->getPalette());
		if (_infos->hasParameter("regenerateFeatureBuffer"))
			generateFeatureBuffer();
	}
	else if (_infos->nameCommand == "getInfoObjectCurrentlyPicked") {
		if (m_idSelection != -1) {
			QString infos = getInfosTriangle(m_idSelection);
			if (infos.isEmpty()) return;
			_infos->addParameter("infos", std::string(infos.toLatin1().data()));
			if (m_idSelection >= 0)
				generateBoundingBoxSelection(m_idSelection);
		}
	}
	else if (_infos->nameCommand == "getObjectPickedID") {
		if (m_idSelection != -1 && m_pickingEnabled)
			_infos->addParameter("id", m_idSelection);
	}
}

poca::core::CommandInfo ObjectColocalizationCommands::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "pointRendering" || _nameCommand == "shapeRendering" || _nameCommand == "bboxSelection" || _nameCommand == "fill") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "histogram" || _nameCommand == "updateFeature" || _nameCommand == "freeGPU") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	return poca::core::CommandInfo();
}

poca::core::Command* ObjectColocalizationCommands::copy()
{
	return new ObjectColocalizationCommands(*this);
}

void ObjectColocalizationCommands::freeGPUMemory()
{
	m_triangleBuffer.freeGPUMemory();
	m_idBuffer.freeGPUMemory();
	m_triangleFeatureBuffer.freeGPUMemory();
	m_pointBuffer.freeGPUMemory();
	m_boundingBoxSelection.freeGPUMemory();
}

void ObjectColocalizationCommands::createDisplay()
{
	freeGPUMemory();

	poca::geometry::ObjectListInterface* obj = m_colocalization->getObjectsOverlap();
	poca::geometry::DelaunayTriangulationInterface* delau = m_colocalization->getDelaunay();
	std::vector <poca::core::Vec3mf> triangles;

	if (obj != NULL) {
		m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(obj->getPalette());

		obj->generateTriangles(triangles);
		std::vector <float> ids;
		obj->generatePickingIndices(ids);
		if (!triangles.empty() && !ids.empty()) {
			m_triangleBuffer.generateBuffer(triangles.size(), 3, GL_FLOAT);
			m_triangleBuffer.updateBuffer(triangles.data());
			m_idBuffer.generateBuffer(ids.size(), 1, GL_FLOAT);
			m_idBuffer.updateBuffer(ids.data());
			m_triangleFeatureBuffer.generateBuffer(triangles.size(), 1, GL_FLOAT);
		}
	}

	/*std::vector <poca::core::Vec3mf> points;
	const std::vector <float> xs = m_colocalization->getXs(), ys = m_colocalization->getYs(), zs = m_colocalization->getZs();
	for (size_t n = 0; n < xs.size(); n++) {
		points.push_back(poca::core::Vec3mf(xs[n], ys[n], zs.empty() ? 0.f : zs[n]));
	}*/
	const std::vector <poca::core::Vec3mf>& points = m_colocalization->getLocsOverlapObjects().getData();
	if (!points.empty()) {
		m_pointBuffer.generateBuffer(points.size(), 3, GL_FLOAT);
		m_pointBuffer.updateBuffer(points.data());
	}

	m_boundingBoxSelection.generateBuffer(24, 3, GL_FLOAT);

	if (obj == NULL) return;
	poca::core::HistogramInterface* hist = obj->getCurrentHistogram();
	if (hist == NULL) return;
	generateFeatureBuffer(hist);
}

void ObjectColocalizationCommands::display(poca::opengl::Camera* _cam, const bool _offscreen)
{
	if (!m_colocalization->isSelected()) return;

	drawElements(_cam);
	if (!_offscreen)
		drawPicking(_cam);
}

void ObjectColocalizationCommands::drawElements(poca::opengl::Camera* _cam)
{
	if (m_triangleBuffer.empty())
		createDisplay();
	if (m_pickFBO == NULL)
		updatePickingFBO(_cam->getWidth(), _cam->getHeight());

	bool pointRendering = hasParameter("pointRendering") ? getParameter<bool>("pointRendering") : false;
	bool shapeRendering = hasParameter("shapeRendering") ? getParameter<bool>("shapeRendering") : false;
	bool displayBboxSelection = hasParameter("bboxSelection") ? getParameter<bool>("bboxSelection") : true;
	bool fill = hasParameter("fill") ? getParameter<bool>("fill") : true;

	//std::cout << "In DelaunayTriangulationDisplayCommand::display " << std::endl;
	poca::opengl::Shader* shader = _cam->getShader("simpleShader");

	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	if(m_colocalization->dimension() == 3)
		glEnable(GL_CULL_FACE);
	else
		glDisable(GL_CULL_FACE);

	glCullFace(GL_FRONT);

	glPolygonMode(GL_FRONT_AND_BACK, fill ? GL_FILL : GL_LINE);
	if(fill)
		glEnable(GL_BLEND);
	else
		glDisable(GL_BLEND);
	//glDisable(GL_CULL_FACE);
	if(shapeRendering)
		//_cam->drawUniformShader<poca::core::Vec3mf>(m_completeTriangleBuffer, poca::core::Color4D(0.f, 0.f, 0.F, 0.2f));
		_cam->drawSimpleShader<poca::core::Vec3mf>(m_textureLutID, m_triangleBuffer, m_triangleFeatureBuffer, m_minOriginalFeature, m_maxOriginalFeature, 0.4f);

	glDisable(GL_CULL_FACE);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//glEnable(GL_POINT_SPRITE);
	//glEnable(GL_PROGRAM_POINT_SIZE);
	glDisable(GL_BLEND);
	if (pointRendering)
		_cam->drawUniformShader<poca::core::Vec3mf>(m_pointBuffer, poca::core::Color4D(1.f, 0.f, 1.F, 1.f));

	glDisable(GL_BLEND);
	if (m_idSelection >= 0 && shapeRendering && displayBboxSelection) {
		GLfloat bkColor[4];
		glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
		poca::core::Color4D color = poca::core::contrastColor(poca::core::Color4D(bkColor[0] * 255.f, bkColor[1] * 255.f, bkColor[2] * 255.f, bkColor[3] * 255.f));
		_cam->drawUniformShader<poca::core::Vec3mf>(m_boundingBoxSelection, color);
	}

	/*glDisable(GL_BLEND);
	if (m_idSelection >= 0 && shapeRendering && displayBboxSelection) {
		GLfloat bkColor[4];
		glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
		poca::core::Color4D color = poca::core::contrastColor(poca::core::Color4D(bkColor[0] * 255.f, bkColor[1] * 255.f, bkColor[2] * 255.f, bkColor[3] * 255.f));
		_cam->drawUniformShader<poca::core::Vec3mf>(m_boundingBoxSelection, color);
	}*/

	GL_CHECK_ERRORS(); 
}

void ObjectColocalizationCommands::drawPicking(poca::opengl::Camera* _cam)
{
	if (m_pickFBO == NULL)
		updatePickingFBO(_cam->getWidth(), _cam->getHeight());

	if (m_pickFBO == NULL) return;

	bool pointRendering = getParameter<bool>("pointRendering");
	bool shapeRendering = getParameter<bool>("shapeRendering");

	glEnable(GL_DEPTH_TEST);
	GLfloat bkColor[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	bool success = m_pickFBO->bind();
	if (!success) std::cout << "Problem with binding" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDisable(GL_BLEND);
	_cam->drawPickingShader<poca::core::Vec3mf, float>(m_triangleBuffer, m_idBuffer);

	success = m_pickFBO->release();
	if (!success) std::cout << "Problem with releasing" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, _cam->defaultFramebufferObject());
	glClearColor(bkColor[0], bkColor[1], bkColor[2], bkColor[3]);
	GL_CHECK_ERRORS();
}

QString ObjectColocalizationCommands::getInfosTriangle(const int _id) const
{
	const uint32_t INVALID = std::numeric_limits<std::uint32_t>::max();

	const std::vector <std::pair<uint32_t, uint32_t>>& infos = m_colocalization->getInfosColoc();
	QString text;
	if (_id >= 0) {
		uint32_t idC1 = infos[_id].first, idC2 = infos[_id].second;
		text.append(QString("Colocalization object id: %1\n").arg(_id + 1));
		text.append(QString("Overlap objects [%1,%2]\n").arg(idC1 == INVALID ? INVALID : idC1 + 1).arg(idC2 == INVALID ? INVALID : idC2 + 1));

		float areaIntersection = m_colocalization->getAreaObjectIntersection(_id);
		float areaObjColor1 = idC1 == INVALID ? 0.f : m_colocalization->getAreaObjectColor(0, infos[_id].first);
		float areaColor2 = idC2 == INVALID ?  0.f : m_colocalization->getAreaObjectColor(1, infos[_id].second);
		text.append(QString("Ratio overlap [%1%,%2%]").arg((areaIntersection / areaObjColor1) * 100.f).arg((areaIntersection / areaColor2) * 100.f));
	}
	return text;
}

void ObjectColocalizationCommands::generateFeatureBuffer(poca::core::HistogramInterface* _histogram)
{
	poca::geometry::ObjectListInterface* objects = m_colocalization->getObjectsOverlap();
	if (_histogram == NULL)
		_histogram = objects->getCurrentHistogram();
	poca::core::Histogram<float>* histogram = dynamic_cast <poca::core::Histogram<float>*>(_histogram);
	const std::vector<float>& values = histogram->getValues();
	const std::vector<bool>& selection = objects->getSelection();
	if (values.empty() || selection.empty()) return;
	m_minOriginalFeature = histogram->getMin();
	m_maxOriginalFeature = histogram->getMax();
	m_actualValueFeature = m_maxOriginalFeature;

	std::vector <float> featureLocs, featureValues, featureOutlines;
	if (objects->isHiLow()) {
		float inter = m_maxOriginalFeature - m_minOriginalFeature;
		float selectedValue = m_minOriginalFeature + inter / 4.f;
		float notSelectedValue = m_minOriginalFeature + inter * (3.f / 4.f);
		objects->getFeatureInSelectionHiLow(featureValues, selection, selectedValue, notSelectedValue);
	}
	else {
		objects->getFeatureInSelection(featureValues, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
	}
	m_triangleFeatureBuffer.updateBuffer(featureValues.data());
}

void ObjectColocalizationCommands::generateBoundingBoxSelection(const int _idx)
{
	if (_idx < 0) return;
	bool shapeRendering = getParameter<bool>("shapeRendering");
	if (!shapeRendering) return;

	poca::core::BoundingBox bbox = m_colocalization->computeBoundingBoxElement(_idx);
	//std::cout << "bbox " << bbox << std::endl;

	std::vector <poca::core::Vec3mf> cube(24);
	poca::geometry::createCubeFromVector(cube, bbox);
	m_boundingBoxSelection.updateBuffer(cube.data());
}

/*void ObjectColocalizationCommands::saveCommands(nlohmann::json& _json)
{
}*/

