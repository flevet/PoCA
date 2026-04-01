/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiMultiObjectDisplayCommand.cpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#include <Windows.h>
#include <algorithm>
#include <limits>

#include <gl/glew.h>
#include <gl/GL.h>

#include <QtGui/QOpenGLFramebufferObject>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Objects/MyMultipleObject.hpp>
#include <Objects/MyObject.hpp>
#include <General/Engine.hpp>
#include <General/Histogram.hpp>
#include <General/Misc.h>
#include <General/MyData.hpp>
#include <Geometry/VoronoiDiagram.hpp>
#include <OpenGL/Camera.hpp>
#include <OpenGL/Helper.h>
#include <OpenGL/RenderCommandContext.hpp>
#include <OpenGL/Shader.hpp>

#include "VoronoiDiagramDisplayCommand.hpp"
#include "VoronoiMultiObjectDisplayCommand.hpp"

namespace {
	poca::core::Vec3mf transformPosition(const glm::mat4& _model, const poca::core::Vec3mf& _pos)
	{
		const glm::vec4 transformed = _model * glm::vec4(_pos.x(), _pos.y(), _pos.z(), 1.f);
		return poca::core::Vec3mf(transformed.x, transformed.y, transformed.z);
	}

	poca::core::Vec3mf transformDirection(const glm::mat4& _model, const poca::core::Vec3mf& _dir)
	{
		const glm::vec4 transformed = _model * glm::vec4(_dir.x(), _dir.y(), _dir.z(), 0.f);
		return poca::core::Vec3mf(transformed.x, transformed.y, transformed.z);
	}
}

VoronoiMultiObjectDisplayCommand::VoronoiMultiObjectDisplayCommand(MyMultipleObject* _object)
	: poca::opengl::BasicDisplayCommand(nullptr, "VoronoiMultiObjectDisplayCommand"),
	m_object(_object), m_hasCells(false), m_is3D(false), m_textureLutID(0),
	m_minOriginalFeature(0.f), m_maxOriginalFeature(1.f)
{
}

VoronoiMultiObjectDisplayCommand::VoronoiMultiObjectDisplayCommand(const VoronoiMultiObjectDisplayCommand& _o)
	: poca::opengl::BasicDisplayCommand(_o), m_object(_o.m_object), m_hasCells(false), m_is3D(false), m_textureLutID(0),
	m_minOriginalFeature(0.f), m_maxOriginalFeature(1.f)
{
}

VoronoiMultiObjectDisplayCommand::~VoronoiMultiObjectDisplayCommand()
{
	freeGPUMemory();
}

std::vector<poca::core::CommandSpec> VoronoiMultiObjectDisplayCommand::commandSpecs() const
{
	return poca::opengl::BasicDisplayCommand::commandSpecs();
}

void VoronoiMultiObjectDisplayCommand::execute(poca::core::CommandInfo* _infos)
{
	poca::core::CommandExecutionContext context;
	poca::core::CommandExecutionResult result;
	execute(_infos, context, result);
}

void VoronoiMultiObjectDisplayCommand::execute(poca::core::CommandInfo* _infos, const poca::core::CommandExecutionContext& _context, poca::core::CommandExecutionResult& _result)
{
	poca::opengl::BasicDisplayCommand::execute(_infos, _context, _result);

	if (_infos->nameCommand == "display") {
		poca::opengl::Camera* cam = nullptr;
		if (_context.has<poca::opengl::ActiveCamera>())
			cam = _context.get<poca::opengl::ActiveCamera>().camera;
		if (!cam) return;
		bool offscreen = _infos->hasParameter("offscreen") && _infos->getParameter<bool>("offscreen");
		display(cam, offscreen, _result);
	}
	else if (_infos->nameCommand == "pick") {
		if (!canBatch()) return;
		QString infos = getInfosTriangle(m_idSelection);
		if (!infos.isEmpty()) {
			poca::core::stringList listInfos;
			if (_result.has<poca::opengl::PickedInfoListResult>())
				listInfos = _result.get<poca::opengl::PickedInfoListResult>().infos;
			listInfos.push_back(infos.toLatin1().data());
			_result.set(poca::opengl::PickedInfoListResult{ listInfos });
		}
		if (m_hasCells && m_idSelection >= 0 && (size_t)m_idSelection < m_trianglePickMap.size()) {
			generateBoundingBoxSelection(m_idSelection);
			const poca::opengl::PickMappingEntry& picked = m_trianglePickMap[m_idSelection];
			poca::core::MyObjectInterface* child = m_object->getObject(picked.objectIndex);
			poca::geometry::VoronoiDiagram* voro = dynamic_cast<poca::geometry::VoronoiDiagram*>(child->getBasicComponent("VoronoiDiagram"));
			if (voro) {
				std::vector<poca::core::Vec3mf> pickedPoints;
				if (_result.has<poca::opengl::PickedPointsResult>())
					pickedPoints = _result.get<poca::opengl::PickedPointsResult>().points;
				const glm::mat4 model = glm::inverse(m_object->getModelMatrix()) * child->getModelMatrix();
				pickedPoints.push_back(transformPosition(model, voro->computeBarycenterElement((int)picked.localIndex)));
				_result.set(poca::opengl::PickedPointsResult{ pickedPoints });
			}
		}
		_result.set(poca::opengl::ChildObjectRenderingHandled{ true });
	}
	else if (_infos->nameCommand == "changeLUT" || _infos->nameCommand == "histogram" || _infos->nameCommand == "updateFeature") {
		freeGPUMemory();
	}
	else if (_infos->nameCommand == "freeGPU") {
		freeGPUMemory();
	}
}

poca::core::Command* VoronoiMultiObjectDisplayCommand::copy()
{
	return new VoronoiMultiObjectDisplayCommand(*this);
}

bool VoronoiMultiObjectDisplayCommand::canBatch() const
{
	if (m_object == nullptr || m_object->nbColors() <= 1)
		return false;

	bool first = true;
	bool hasCells = false;
	uint32_t dimension = 0;
	bool hasSelectedChild = false;
	for (size_t n = 0; n < m_object->nbColors(); n++) {
		poca::core::MyObjectInterface* child = m_object->getObject(n);
		poca::geometry::VoronoiDiagram* voro = dynamic_cast<poca::geometry::VoronoiDiagram*>(child->getBasicComponent("VoronoiDiagram"));
		if (voro == nullptr || !voro->isSelected())
			continue;

		hasSelectedChild = true;
		if (first) {
			hasCells = voro->hasCells();
			dimension = voro->dimension();
			first = false;
		}
		else if (hasCells != voro->hasCells() || dimension != voro->dimension()) {
			return false;
		}
	}

	return hasSelectedChild;
}

VoronoiDiagramDisplayCommand* VoronoiMultiObjectDisplayCommand::referenceDisplayCommand() const
{
	if (m_object == nullptr) return nullptr;
	for (size_t n = 0; n < m_object->nbColors(); n++) {
		poca::core::MyObjectInterface* child = m_object->getObject(n);
		poca::geometry::VoronoiDiagram* voro = dynamic_cast<poca::geometry::VoronoiDiagram*>(child->getBasicComponent("VoronoiDiagram"));
		if (voro == nullptr || !voro->isSelected())
			continue;
		return voro->getCommand<VoronoiDiagramDisplayCommand>();
	}
	return nullptr;
}

bool VoronoiMultiObjectDisplayCommand::rebuild()
{
	freeGPUMemory();
	if (!canBatch())
		return false;

	VoronoiDiagramDisplayCommand* referenceCommand = referenceDisplayCommand();
	if (referenceCommand == nullptr)
		return false;

	poca::opengl::MultiPointData<poca::core::Vec3mf> pointData;
	poca::opengl::MultiTriangleData<poca::core::Vec3mf> triangleData;
	poca::opengl::MultiLineData<poca::core::Vec3mf> lineData;

	m_hasCells = false;
	m_is3D = false;
	m_minOriginalFeature = std::numeric_limits<float>::max();
	m_maxOriginalFeature = std::numeric_limits<float>::lowest();

	for (size_t objectIndex = 0; objectIndex < m_object->nbColors(); objectIndex++) {
		poca::core::MyObjectInterface* child = m_object->getObject(objectIndex);
		poca::geometry::VoronoiDiagram* voro = dynamic_cast<poca::geometry::VoronoiDiagram*>(child->getBasicComponent("VoronoiDiagram"));
		if (voro == nullptr || !voro->isSelected())
			continue;

		poca::core::HistogramInterface* histInterface = voro->getCurrentHistogram();
		poca::core::Histogram<float>* histogram = dynamic_cast<poca::core::Histogram<float>*>(histInterface);
		if (histogram == nullptr)
			continue;

		m_hasCells = voro->hasCells();
		m_is3D = voro->dimension() == 3;
		m_minOriginalFeature = std::min(m_minOriginalFeature, histInterface->getMin());
		m_maxOriginalFeature = std::max(m_maxOriginalFeature, histInterface->getMax());

		const glm::mat4 model = glm::inverse(m_object->getModelMatrix()) * child->getModelMatrix();
		const std::vector<float>& values = histogram->getValues();
		const std::vector<bool>& selection = voro->getSelection();
		const std::vector<bool>& borderLocs = voro->borderLocalizations();

		const size_t pointBase = pointData.pickMap.size();
		for (size_t idx = 0; idx < voro->nbFaces(); idx++) {
			const float z = voro->getZs() != nullptr ? voro->getZs()[idx] : 0.f;
			pointData.vertices.push_back(transformPosition(model, poca::core::Vec3mf(voro->getXs()[idx], voro->getYs()[idx], z)));
			pointData.ids.push_back((float)(pointBase + idx + 1));
			pointData.features.push_back(selection[idx] && !borderLocs[idx] ? values[idx] : poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
			pointData.pickMap.push_back({ (uint32_t)objectIndex, (uint32_t)idx });
		}

		if (!voro->hasCells())
			continue;

		const size_t triangleBase = triangleData.pickMap.size();
		for (size_t idx = 0; idx < voro->nbFaces(); idx++)
			triangleData.pickMap.push_back({ (uint32_t)objectIndex, (uint32_t)idx });

		std::vector<poca::core::Vec3mf> triangles;
		voro->generateTriangles(triangles);
		for (const auto& vertex : triangles)
			triangleData.vertices.push_back(transformPosition(model, vertex));

		std::vector<float> triangleIds;
		voro->generatePickingIndices(triangleIds);
		for (float id : triangleIds) {
			const uint32_t localIndex = id > 0.f ? (uint32_t)(id - 1.f) : 0u;
			triangleData.ids.push_back((float)(triangleBase + localIndex + 1));
		}

		std::vector<float> triangleFeatures;
		voro->getFeatureInSelection(triangleFeatures, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER, false);
		triangleData.features.insert(triangleData.features.end(), triangleFeatures.begin(), triangleFeatures.end());

		if (voro->dimension() == 2) {
			std::vector<poca::core::Vec3mf> lines, normals;
			voro->generateLines(lines);
			for (const auto& vertex : lines)
				lineData.vertices.push_back(transformPosition(model, vertex));

			voro->generateLinesNormals(normals);
			for (const auto& normal : normals)
				lineData.normals.push_back(transformDirection(model, normal));

			std::vector<float> lineFeatures;
			voro->getFeatureInSelection(lineFeatures, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER, true);
			lineData.features.insert(lineData.features.end(), lineFeatures.begin(), lineFeatures.end());
		}
	}

	if (m_minOriginalFeature == std::numeric_limits<float>::max()) {
		m_minOriginalFeature = 0.f;
		m_maxOriginalFeature = 1.f;
	}
	if (m_maxOriginalFeature <= m_minOriginalFeature)
		m_maxOriginalFeature = m_minOriginalFeature + 1.f;

	poca::geometry::VoronoiDiagram* referenceVoronoi = nullptr;
	for (size_t n = 0; n < m_object->nbColors() && referenceVoronoi == nullptr; n++) {
		poca::core::MyObjectInterface* child = m_object->getObject(n);
		referenceVoronoi = dynamic_cast<poca::geometry::VoronoiDiagram*>(child->getBasicComponent("VoronoiDiagram"));
	}
	if (referenceVoronoi == nullptr)
		return false;

	m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(referenceVoronoi->getPalette());

	if (!pointData.vertices.empty()) {
		m_pointBuffer.generateBuffer(pointData.vertices.size(), 3, GL_FLOAT);
		m_idLocsBuffer.generateBuffer(pointData.ids.size(), 1, GL_FLOAT);
		m_locsFeatureBuffer.generateBuffer(pointData.features.size(), 1, GL_FLOAT);
		m_pointBuffer.updateBuffer(pointData.vertices.data());
		m_idLocsBuffer.updateBuffer(pointData.ids.data());
		m_locsFeatureBuffer.updateBuffer(pointData.features.data());
		m_pointPickMap = std::move(pointData.pickMap);
	}

	if (!triangleData.vertices.empty()) {
		m_triangleBuffer.generateBuffer(triangleData.vertices.size(), 512 * 512, 3, GL_FLOAT);
		m_triangleFeatureBuffer.generateBuffer(triangleData.features.size(), 512 * 512, 1, GL_FLOAT);
		m_idPolytopeBuffer.generateBuffer(triangleData.ids.size(), 512 * 512, 1, GL_FLOAT);
		m_triangleBuffer.updateBuffer(triangleData.vertices.data());
		m_triangleFeatureBuffer.updateBuffer(triangleData.features.data());
		m_idPolytopeBuffer.updateBuffer(triangleData.ids.data());
		m_trianglePickMap = std::move(triangleData.pickMap);
		m_boundingBoxSelection.generateBuffer(24, 512 * 512, 3, GL_FLOAT);
	}

	if (!lineData.vertices.empty()) {
		m_lineBuffer.generateBuffer(lineData.vertices.size(), 3, GL_FLOAT);
		m_lineFeatureBuffer.generateBuffer(lineData.features.size(), 1, GL_FLOAT);
		m_lineBuffer.updateBuffer(lineData.vertices.data());
		m_lineFeatureBuffer.updateBuffer(lineData.features.data());
		if (!lineData.normals.empty()) {
			m_lineNormalBuffer.generateBuffer(lineData.normals.size(), 3, GL_FLOAT);
			m_lineNormalBuffer.updateBuffer(lineData.normals.data());
		}
	}

	return !m_pointBuffer.empty() || !m_triangleBuffer.empty() || !m_lineBuffer.empty();
}

void VoronoiMultiObjectDisplayCommand::display(poca::opengl::Camera* _cam, const bool _offscreen, poca::core::CommandExecutionResult& _result)
{
	if (!canBatch())
		return;
	if (m_pointBuffer.empty() && !rebuild())
		return;

	VoronoiDiagramDisplayCommand* referenceCommand = referenceDisplayCommand();
	if (referenceCommand == nullptr)
		return;

	drawElements(_cam, referenceCommand);
	if (!_offscreen)
		drawPicking(_cam);
	_result.set(poca::opengl::ChildObjectRenderingHandled{ true });
}

void VoronoiMultiObjectDisplayCommand::drawElements(poca::opengl::Camera* _cam, VoronoiDiagramDisplayCommand* _referenceCommand)
{
	const bool pointRendering = _referenceCommand->getParameter<bool>("pointRendering");
	const bool polytopeRendering = _referenceCommand->getParameter<bool>("polytopeRendering");
	const bool fill = _referenceCommand->getParameter<bool>("fill");
	const bool displayBboxSelection = _referenceCommand->getParameter<bool>("bboxSelection");
	const uint32_t pointSize = _referenceCommand->getParameter<uint32_t>("pointSizeGL");
	const bool cullFaceActivated = _cam->cullFaceActivated();

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);

	if (pointRendering && !m_pointBuffer.empty()) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		_cam->drawSphereRendering<poca::core::Vec3mf, float>(m_textureLutID, m_pointBuffer, m_locsFeatureBuffer,
			m_minOriginalFeature, m_maxOriginalFeature, m_minOriginalFeature, m_maxOriginalFeature, false, pointSize, false);
	}

	if (polytopeRendering && m_hasCells && !m_triangleBuffer.empty()) {
		glPolygonMode(GL_FRONT_AND_BACK, fill ? GL_FILL : GL_LINE);
		if (cullFaceActivated)
			glEnable(GL_CULL_FACE);
		else
			glDisable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		if (m_is3D) {
			glEnable(GL_BLEND);
			_cam->drawSimpleShader<poca::core::Vec3mf, float>(m_textureLutID, m_triangleBuffer, m_triangleFeatureBuffer,
				m_minOriginalFeature, m_maxOriginalFeature, 0.3f);
		}
		else if (fill) {
			_cam->drawSimpleShader<poca::core::Vec3mf, float>(m_textureLutID, m_triangleBuffer, m_triangleFeatureBuffer,
				m_minOriginalFeature, m_maxOriginalFeature);
		}
		else {
			glDisable(GL_DEPTH_TEST);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			if (m_lineNormalBuffer.empty())
				glDisable(GL_CULL_FACE);
			_cam->drawLineShader<poca::core::Vec3mf, float>(m_textureLutID, m_lineBuffer, m_lineFeatureBuffer, m_lineNormalBuffer,
				m_minOriginalFeature, m_maxOriginalFeature, 1.f);
		}
	}

	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);

	if (m_idSelection >= 0 && displayBboxSelection && m_hasCells && !m_boundingBoxSelection.empty()) {
		GLfloat bkColor[4];
		glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
		const poca::core::Color4D color = poca::core::contrastColor(
			poca::core::Color4D(bkColor[0] * 255.f, bkColor[1] * 255.f, bkColor[2] * 255.f, bkColor[3] * 255.f));
		_cam->drawUniformShader<poca::core::Vec3mf>(m_boundingBoxSelection, color);
	}
}

void VoronoiMultiObjectDisplayCommand::drawPicking(poca::opengl::Camera* _cam)
{
	if (m_pickFBO == nullptr)
		updatePickingFBO(_cam->getWidth(), _cam->getHeight());
	if (m_pickFBO == nullptr)
		return;

	glEnable(GL_DEPTH_TEST);
	GLfloat bkColor[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	const bool successBind = m_pickFBO->bind();
	if (!successBind) std::cout << "Problem with binding" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (m_hasCells)
		_cam->drawPickingShader<poca::core::Vec3mf, float>(m_triangleBuffer, m_idPolytopeBuffer, m_triangleFeatureBuffer, m_minOriginalFeature);
	else
		_cam->drawPickingShader<poca::core::Vec3mf, float>(m_pointBuffer, m_idLocsBuffer, m_locsFeatureBuffer, m_minOriginalFeature);

	const bool successRelease = m_pickFBO->release();
	if (!successRelease) std::cout << "Problem with releasing" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, _cam->defaultFramebufferObject());
	glClearColor(bkColor[0], bkColor[1], bkColor[2], bkColor[3]);
}

QString VoronoiMultiObjectDisplayCommand::getInfosTriangle(const int _id) const
{
	if (_id < 0)
		return QString();

	const std::vector<poca::opengl::PickMappingEntry>& pickMap = m_hasCells ? m_trianglePickMap : m_pointPickMap;
	if ((size_t)_id >= pickMap.size())
		return QString();

	const poca::opengl::PickMappingEntry& picked = pickMap[_id];
	poca::core::MyObjectInterface* child = m_object->getObject(picked.objectIndex);
	poca::geometry::VoronoiDiagram* voro = dynamic_cast<poca::geometry::VoronoiDiagram*>(child->getBasicComponent("VoronoiDiagram"));
	if (voro == nullptr)
		return QString();

	QString text;
	text.append(QString("Voronoi diagram id: %1").arg(picked.localIndex));
	for (const std::string& type : voro->getNameData()) {
		const std::vector<float>& values = voro->getMyData(type)->getData<float>();
		if (picked.localIndex < values.size())
			text.append(QString("\n%1: %2").arg(type.c_str()).arg(values[picked.localIndex]));
	}
	return text;
}

void VoronoiMultiObjectDisplayCommand::generateBoundingBoxSelection(const int _idx)
{
	if (_idx < 0 || (size_t)_idx >= m_trianglePickMap.size())
		return;

	const poca::opengl::PickMappingEntry& picked = m_trianglePickMap[_idx];
	poca::core::MyObjectInterface* child = m_object->getObject(picked.objectIndex);
	poca::geometry::VoronoiDiagram* voro = dynamic_cast<poca::geometry::VoronoiDiagram*>(child->getBasicComponent("VoronoiDiagram"));
	if (voro == nullptr || !voro->hasCells())
		return;

	poca::core::BoundingBox bbox = voro->computeBoundingBoxElement((int)picked.localIndex);
	std::vector<poca::core::Vec3mf> cube(24);
	poca::geometry::createCubeFromVector(cube, bbox);
	const glm::mat4 model = glm::inverse(m_object->getModelMatrix()) * child->getModelMatrix();
	for (auto& point : cube)
		point = transformPosition(model, point);
	m_boundingBoxSelection.updateBuffer(cube.data());
}

void VoronoiMultiObjectDisplayCommand::freeGPUMemory()
{
	m_pointBuffer.freeGPUMemory();
	m_idLocsBuffer.freeGPUMemory();
	m_locsFeatureBuffer.freeGPUMemory();
	m_triangleBuffer.freeGPUMemory();
	m_idPolytopeBuffer.freeGPUMemory();
	m_triangleFeatureBuffer.freeGPUMemory();
	m_lineBuffer.freeGPUMemory();
	m_lineNormalBuffer.freeGPUMemory();
	m_lineFeatureBuffer.freeGPUMemory();
	m_boundingBoxSelection.freeGPUMemory();
	m_pointPickMap.clear();
	m_trianglePickMap.clear();
	m_textureLutID = 0;
}
