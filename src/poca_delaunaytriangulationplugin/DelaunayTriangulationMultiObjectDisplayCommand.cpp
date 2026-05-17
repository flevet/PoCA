/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DelaunayTriangulationMultiObjectDisplayCommand.cpp
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

#include <Objects/MyMultipleObject.hpp>
#include <General/Histogram.hpp>
#include <General/Misc.h>
#include <Interfaces/DelaunayTriangulationInterface.hpp>
#include <OpenGL/Camera.hpp>
#include <OpenGL/Helper.h>
#include <OpenGL/RenderCommandContext.hpp>
#include <OpenGL/Shader.hpp>

#include "DelaunayTriangulationDisplayCommand.hpp"
#include "DelaunayTriangulationMultiObjectDisplayCommand.hpp"

namespace {
	const std::string kRenderedComponentName = "DelaunayTriangulation";

	void markComponentFamilyHandled(poca::core::CommandExecutionResult& _result)
	{
		poca::opengl::RenderedComponentFamilies families;
		if (_result.has<poca::opengl::RenderedComponentFamilies>())
			families = _result.get<poca::opengl::RenderedComponentFamilies>();
		if (std::find(families.componentNames.begin(), families.componentNames.end(), kRenderedComponentName) == families.componentNames.end())
			families.componentNames.push_back(kRenderedComponentName);
		_result.set(families);
		_result.set(poca::opengl::ChildObjectRenderingHandled{ true });
	}

	poca::core::Vec3mf transformPosition(const glm::mat4& _model, const poca::core::Vec3mf& _pos)
	{
		const glm::vec4 transformed = _model * glm::vec4(_pos.x(), _pos.y(), _pos.z(), 1.f);
		return poca::core::Vec3mf(transformed.x, transformed.y, transformed.z);
	}

	QString globalObjectInfo(MyMultipleObject* _object, const size_t _objectIndex)
	{
		if (_object == nullptr || _objectIndex >= _object->nbColors())
			return QString();

		poca::core::MyObjectInterface* child = _object->getObject(_objectIndex);
		if (child == nullptr)
			return QString();

		QString text;
		text.append(QString("Global object index: %1").arg(_objectIndex));
		text.append(QString("\nGlobal object id: %1").arg(child->currentInternalId()));
		if (!child->getName().empty())
			text.append(QString("\nName: %1").arg(child->getName().c_str()));
		if (!child->getDir().empty())
			text.append(QString("\nDir: %1").arg(child->getDir().c_str()));
		return text;
	}

	void appendGlobalObjectInfo(poca::core::CommandExecutionResult& _result, MyMultipleObject* _object, const size_t _objectIndex)
	{
		const QString infos = globalObjectInfo(_object, _objectIndex);
		if (infos.isEmpty())
			return;

		poca::core::stringList listInfos;
		if (_result.has<poca::opengl::PickedInfoListResult>())
			listInfos = _result.get<poca::opengl::PickedInfoListResult>().infos;
		listInfos.push_back(infos.toLatin1().data());
		_result.set(poca::opengl::PickedInfoListResult{ listInfos });
	}

	void handlePickedObjectSelection(poca::core::CommandInfo* _infos, MyMultipleObject* _object, const size_t _objectIndex)
	{
		if (_object == nullptr || !_infos->hasParameter("click"))
			return;

		const std::string click = _infos->getParameter<std::string>("click");
		if (click != "left" || _objectIndex >= _object->nbColors())
			return;

		_object->setCurrentObject(_objectIndex);
		_object->setSelectedObjectIndices({ _objectIndex });
		_object->notify("LoadObjCharacteristicsAllWidgets");
		_object->notifyAll("updateDisplay");
	}
}

DelaunayTriangulationMultiObjectDisplayCommand::DelaunayTriangulationMultiObjectDisplayCommand(MyMultipleObject* _object)
	: poca::opengl::BasicDisplayCommand(nullptr, "DelaunayTriangulationMultiObjectDisplayCommand"),
	m_object(_object), m_textureLutID(0), m_minOriginalFeature(0.f), m_maxOriginalFeature(1.f), m_actualValueFeature(1.f)
{
}

DelaunayTriangulationMultiObjectDisplayCommand::DelaunayTriangulationMultiObjectDisplayCommand(const DelaunayTriangulationMultiObjectDisplayCommand& _o)
	: poca::opengl::BasicDisplayCommand(_o), m_object(_o.m_object), m_textureLutID(0), m_minOriginalFeature(0.f), m_maxOriginalFeature(1.f), m_actualValueFeature(1.f)
{
}

DelaunayTriangulationMultiObjectDisplayCommand::~DelaunayTriangulationMultiObjectDisplayCommand()
{
	freeGPUMemory();
}

std::vector<poca::core::CommandSpec> DelaunayTriangulationMultiObjectDisplayCommand::commandSpecs() const
{
	return poca::opengl::BasicDisplayCommand::commandSpecs();
}

void DelaunayTriangulationMultiObjectDisplayCommand::execute(poca::core::CommandInfo* _infos)
{
	poca::core::CommandExecutionContext context;
	poca::core::CommandExecutionResult result;
	execute(_infos, context, result);
}

void DelaunayTriangulationMultiObjectDisplayCommand::execute(poca::core::CommandInfo* _infos, const poca::core::CommandExecutionContext& _context, poca::core::CommandExecutionResult& _result)
{
	if (_infos == nullptr)
		return;
	if (_infos->nameCommand != "pick" && _infos->nameCommand != "updatePickingBuffer")
		poca::opengl::BasicDisplayCommand::execute(_infos);

	if (_infos->nameCommand == "updatePickingBuffer") {
		markComponentFamilyHandled(_result);
	}
	else if (_infos->nameCommand == "display") {
		poca::opengl::Camera* cam = nullptr;
		if (_context.has<poca::opengl::ActiveCamera>())
			cam = _context.get<poca::opengl::ActiveCamera>().camera;
		if (!cam) return;
		const bool offscreen = _infos->hasParameter("offscreen") && _infos->getParameter<bool>("offscreen");
		display(cam, offscreen, _result);
	}
	else if (_infos->nameCommand == "pick") {
		if (!canBatch()) return;
		if (!pickObjectBoundingBox(_infos, _context, m_object, m_idSelection))
			return;
		if (m_idSelection >= 0 && (size_t)m_idSelection < m_object->nbColors()) {
			generateBoundingBoxSelection(m_idSelection);
			const size_t objectIndex = (size_t)m_idSelection;
			appendGlobalObjectInfo(_result, m_object, objectIndex);
			_result.set(poca::opengl::PickedObjectIdResult{ m_idSelection, true });
			handlePickedObjectSelection(_infos, m_object, objectIndex);
			poca::core::MyObjectInterface* child = m_object->getObject(objectIndex);
			if (child != nullptr) {
				std::vector<poca::core::Vec3mf> pickedPoints;
				if (_result.has<poca::opengl::PickedPointsResult>())
					pickedPoints = _result.get<poca::opengl::PickedPointsResult>().points;
				pickedPoints.push_back(childObjectBoundingBox(m_object, child).centroid());
				_result.set(poca::opengl::PickedPointsResult{ pickedPoints });
			}
		}
		markComponentFamilyHandled(_result);
	}
	else if (_infos->nameCommand == "changeLUT") {
		freeGPUMemory();
	}
	else if (_infos->nameCommand == "histogram" || _infos->nameCommand == "updateFeature") {
		if (!updateFeatureBuffer())
			freeGPUMemory();
	}
	else if (_infos->nameCommand == "freeGPU") {
		freeGPUMemory();
	}
}

poca::core::Command* DelaunayTriangulationMultiObjectDisplayCommand::copy()
{
	return new DelaunayTriangulationMultiObjectDisplayCommand(*this);
}

bool DelaunayTriangulationMultiObjectDisplayCommand::canBatch() const
{
	if (m_object == nullptr || m_object->nbColors() <= 1)
		return false;

	bool hasDelaunayChild = false, first = true;
	uint32_t dimension = 0;
	for (size_t n = 0; n < m_object->nbColors(); n++) {
		poca::core::MyObjectInterface* child = m_object->getObject(n);
		poca::geometry::DelaunayTriangulationInterface* delau = dynamic_cast<poca::geometry::DelaunayTriangulationInterface*>(child->getBasicComponent("DelaunayTriangulation"));
		if (delau == nullptr)
			continue;
		hasDelaunayChild = true;
		if (first) {
			dimension = delau->dimension();
			first = false;
		}
		else if (dimension != delau->dimension())
			return false;
	}
	return hasDelaunayChild;
}

DelaunayTriangulationDisplayCommand* DelaunayTriangulationMultiObjectDisplayCommand::referenceDisplayCommand() const
{
	if (m_object == nullptr) return nullptr;
	for (size_t n = 0; n < m_object->nbColors(); n++) {
		poca::core::MyObjectInterface* child = m_object->getObject(n);
		poca::geometry::DelaunayTriangulationInterface* delau = dynamic_cast<poca::geometry::DelaunayTriangulationInterface*>(child->getBasicComponent("DelaunayTriangulation"));
		if (delau == nullptr)
			continue;
		return delau->getCommand<DelaunayTriangulationDisplayCommand>();
	}
	return nullptr;
}

bool DelaunayTriangulationMultiObjectDisplayCommand::rebuild()
{
	freeGPUMemory();
	if (!canBatch())
		return false;

	std::vector<poca::core::Vec3mf> triangles;
	std::vector<float> features;
	m_minOriginalFeature = std::numeric_limits<float>::max();
	m_maxOriginalFeature = std::numeric_limits<float>::lowest();

	for (size_t objectIndex = 0; objectIndex < m_object->nbColors(); objectIndex++) {
		poca::core::MyObjectInterface* child = m_object->getObject(objectIndex);
		poca::geometry::DelaunayTriangulationInterface* delau = dynamic_cast<poca::geometry::DelaunayTriangulationInterface*>(child->getBasicComponent("DelaunayTriangulation"));
		if (delau == nullptr)
			continue;

		poca::core::HistogramInterface* histInterface = delau->getCurrentHistogram();
		poca::core::Histogram<float>* histogram = dynamic_cast<poca::core::Histogram<float>*>(histInterface);
		if (histogram == nullptr)
			continue;

		m_minOriginalFeature = std::min(m_minOriginalFeature, histInterface->getMin());
		m_maxOriginalFeature = std::max(m_maxOriginalFeature, histInterface->getMax());
		const glm::mat4 model = glm::inverse(m_object->getModelMatrix()) * child->getModelMatrix();

		std::vector<poca::core::Vec3mf> localTriangles;
		delau->generateTriangles(localTriangles);
		for (const auto& vertex : localTriangles)
			triangles.push_back(transformPosition(model, vertex));

		std::vector<float> localFeatures;
		delau->getFeatureInSelection(localFeatures, histogram->getValues(), delau->getSelection(), poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		if (!delau->isSelected())
			std::fill(localFeatures.begin(), localFeatures.end(), poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		features.insert(features.end(), localFeatures.begin(), localFeatures.end());
	}

	if (triangles.empty())
		return false;
	if (m_minOriginalFeature == std::numeric_limits<float>::max()) {
		m_minOriginalFeature = 0.f;
		m_maxOriginalFeature = 1.f;
	}
	m_actualValueFeature = m_maxOriginalFeature;

	poca::geometry::DelaunayTriangulationInterface* reference = nullptr;
	for (size_t n = 0; n < m_object->nbColors() && reference == nullptr; n++) {
		poca::core::MyObjectInterface* child = m_object->getObject(n);
		reference = dynamic_cast<poca::geometry::DelaunayTriangulationInterface*>(child->getBasicComponent("DelaunayTriangulation"));
	}
	if (reference == nullptr)
		return false;

	m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(reference->getPalette());
	m_triangleBuffer.generateBuffer(triangles.size(), 512 * 512, 3, GL_FLOAT);
	m_featureBuffer.generateBuffer(features.size(), 512 * 512, 1, GL_FLOAT);
	m_triangleBuffer.updateBuffer(triangles.data());
	m_featureBuffer.updateBuffer(features.data());
	m_boundingBoxSelection.generateBuffer(24, 512 * 512, 3, GL_FLOAT);
	return true;
}

void DelaunayTriangulationMultiObjectDisplayCommand::display(poca::opengl::Camera* _cam, const bool _offscreen, poca::core::CommandExecutionResult& _result)
{
	(void)_offscreen;
	if (!canBatch())
		return;
	if (m_triangleBuffer.empty() && !rebuild())
		return;

	DelaunayTriangulationDisplayCommand* referenceCommand = referenceDisplayCommand();
	if (referenceCommand == nullptr)
		return;

	drawElements(_cam, referenceCommand);
	markComponentFamilyHandled(_result);
}

bool DelaunayTriangulationMultiObjectDisplayCommand::updateFeatureBuffer()
{
	if (!canBatch() || m_featureBuffer.empty())
		return false;

	std::vector<float> features;
	m_minOriginalFeature = std::numeric_limits<float>::max();
	m_maxOriginalFeature = std::numeric_limits<float>::lowest();

	for (size_t objectIndex = 0; objectIndex < m_object->nbColors(); objectIndex++) {
		poca::core::MyObjectInterface* child = m_object->getObject(objectIndex);
		poca::geometry::DelaunayTriangulationInterface* delau = dynamic_cast<poca::geometry::DelaunayTriangulationInterface*>(child->getBasicComponent("DelaunayTriangulation"));
		if (delau == nullptr)
			continue;

		poca::core::HistogramInterface* histInterface = delau->getCurrentHistogram();
		poca::core::Histogram<float>* histogram = dynamic_cast<poca::core::Histogram<float>*>(histInterface);
		if (histogram == nullptr)
			return false;

		m_minOriginalFeature = std::min(m_minOriginalFeature, histInterface->getMin());
		m_maxOriginalFeature = std::max(m_maxOriginalFeature, histInterface->getMax());
		std::vector<float> localFeatures;
		delau->getFeatureInSelection(localFeatures, histogram->getValues(), delau->getSelection(), poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		if (!delau->isSelected())
			std::fill(localFeatures.begin(), localFeatures.end(), poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		features.insert(features.end(), localFeatures.begin(), localFeatures.end());
	}

	if (m_minOriginalFeature == std::numeric_limits<float>::max()) {
		m_minOriginalFeature = 0.f;
		m_maxOriginalFeature = 1.f;
	}
	m_actualValueFeature = m_maxOriginalFeature;
	m_featureBuffer.updateBuffer(features);
	return true;
}

void DelaunayTriangulationMultiObjectDisplayCommand::drawElements(poca::opengl::Camera* _cam, DelaunayTriangulationDisplayCommand* _referenceCommand)
{
	const bool fill = _referenceCommand->getParameter<bool>("fill");
	const bool displayBboxSelection = _referenceCommand->getParameter<bool>("bboxSelection");
	const bool cullFaceActivated = _cam->cullFaceActivated();

	glPolygonMode(GL_FRONT_AND_BACK, fill ? GL_FILL : GL_LINE);
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	if (cullFaceActivated)
		glEnable(GL_CULL_FACE);
	else
		glDisable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	_cam->drawSimpleShader<poca::core::Vec3mf, float>(m_textureLutID, m_triangleBuffer, m_featureBuffer, m_minOriginalFeature, m_maxOriginalFeature);

	glDisable(GL_DEPTH_TEST);
	if (m_idSelection >= 0 && displayBboxSelection) {
		GLfloat bkColor[4];
		glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
		const poca::core::Color4D color = poca::core::contrastColor(
			poca::core::Color4D(bkColor[0] * 255.f, bkColor[1] * 255.f, bkColor[2] * 255.f, bkColor[3] * 255.f));
		_cam->drawUniformShader<poca::core::Vec3mf>(m_boundingBoxSelection, color);
	}
}

void DelaunayTriangulationMultiObjectDisplayCommand::generateBoundingBoxSelection(const int _idx)
{
	if (_idx < 0 || (size_t)_idx >= m_object->nbColors())
		return;
	poca::core::MyObjectInterface* child = m_object->getObject((size_t)_idx);
	if (child == nullptr)
		return;

	std::vector<poca::core::Vec3mf> cube(24);
	poca::geometry::createCubeFromVector(cube, childObjectBoundingBox(m_object, child));
	if (m_boundingBoxSelection.empty())
		m_boundingBoxSelection.generateBuffer(24, 512 * 512, 3, GL_FLOAT);
	m_boundingBoxSelection.updateBuffer(cube.data());
}

void DelaunayTriangulationMultiObjectDisplayCommand::freeGPUMemory()
{
	m_triangleBuffer.freeGPUMemory();
	m_featureBuffer.freeGPUMemory();
	m_boundingBoxSelection.freeGPUMemory();
	m_textureLutID = 0;
}
