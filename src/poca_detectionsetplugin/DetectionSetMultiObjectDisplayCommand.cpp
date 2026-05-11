/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DetectionSetMultiObjectDisplayCommand.cpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#include <Windows.h>
#include <algorithm>
#include <limits>
#include <map>

#include <gl/glew.h>
#include <gl/GL.h>

#include <QtGui/QOpenGLFramebufferObject>

#include <glm/gtc/matrix_inverse.hpp>

#include <Objects/MyMultipleObject.hpp>
#include <Objects/ObjectCommandContext.hpp>
#include <General/Engine.hpp>
#include <General/Histogram.hpp>
#include <General/MyData.hpp>
#include <Geometry/DetectionSet.hpp>
#include <OpenGL/Camera.hpp>
#include <OpenGL/Helper.h>
#include <OpenGL/RenderCommandContext.hpp>
#include <OpenGL/Shader.hpp>

#include "DetectionSetDisplayCommand.hpp"
#include "DetectionSetMultiObjectDisplayCommand.hpp"

namespace {
	const std::string kRenderedComponentName = "DetectionSet";

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

	poca::core::Vec3mf transformDirection(const glm::mat4& _model, const poca::core::Vec3mf& _dir)
	{
		const glm::vec4 transformed = _model * glm::vec4(_dir.x(), _dir.y(), _dir.z(), 0.f);
		return poca::core::Vec3mf(transformed.x, transformed.y, transformed.z);
	}

	glm::mat4 currentObjectModel(MyMultipleObject* _object)
	{
		poca::core::MyObjectInterface* child = _object != nullptr ? _object->currentObject() : nullptr;
		return child != nullptr ? glm::inverse(_object->getModelMatrix()) * child->getModelMatrix() : glm::mat4(1.f);
	}

	std::string transformedObjectName(poca::core::MyObjectInterface* _source)
	{
		const std::string sourceName = _source != nullptr ? _source->getName() : std::string();
		return sourceName.empty() ? "transformed_object" : sourceName + "_transformed";
	}

	poca::core::MyObjectInterface* createdObject(poca::core::CommandExecutionResult& _result)
	{
		if (!_result.has<poca::core::CreatedObjectContext>())
			return nullptr;
		return _result.get<poca::core::CreatedObjectContext>().object;
	}

	poca::core::MyObjectInterface* addComponentToTransformedObject(poca::core::CommandExecutionResult& _result, poca::core::MyObjectInterface* _source, poca::core::BasicComponentInterface* _component)
	{
		if (_source == nullptr || _component == nullptr)
			return nullptr;

		poca::core::Engine* engine = poca::core::Engine::instance();
		poca::core::MyObjectInterface* object = createdObject(_result);
		if (object != nullptr) {
			if (engine->addComponentToObject(object, _component))
				return object;
			return nullptr;
		}

		object = engine->createObject(_source->getDir(), transformedObjectName(_source), _component);
		if (object != nullptr)
			_result.set(poca::core::CreatedObjectContext{ object });
		return object;
	}

	poca::geometry::DetectionSet* transformedDetectionSet(poca::geometry::DetectionSet* _source, const glm::mat4& _model)
	{
		if (_source == nullptr || !_source->hasData("x") || !_source->hasData("y"))
			return nullptr;

		std::map<std::string, std::vector<float>> data;
		for (const auto& name : _source->getNameData())
			data[name] = _source->getMyData(name)->getOriginalData<float>();

		std::vector<float>& xs = data["x"];
		std::vector<float>& ys = data["y"];
		const bool hasZ = data.find("z") != data.end();
		std::vector<float>* zs = hasZ ? &data["z"] : nullptr;
		for (size_t n = 0; n < xs.size(); n++) {
			const float z = zs != nullptr ? (*zs)[n] : 0.f;
			const poca::core::Vec3mf transformed = transformPosition(_model, poca::core::Vec3mf(xs[n], ys[n], z));
			xs[n] = transformed.x();
			ys[n] = transformed.y();
			if (zs != nullptr)
				(*zs)[n] = transformed.z();
		}

		if (data.find("nx") != data.end() && data.find("ny") != data.end() && data.find("nz") != data.end()) {
			std::vector<float>& nxs = data["nx"];
			std::vector<float>& nys = data["ny"];
			std::vector<float>& nzs = data["nz"];
			for (size_t n = 0; n < nxs.size(); n++) {
				const poca::core::Vec3mf transformed = transformDirection(_model, poca::core::Vec3mf(nxs[n], nys[n], nzs[n]));
				nxs[n] = transformed.x();
				nys[n] = transformed.y();
				nzs[n] = transformed.z();
			}
		}

		poca::geometry::DetectionSet* transformed = new poca::geometry::DetectionSet(data);
		if (transformed->nbElements() == _source->nbElements())
			transformed->setSelection(_source->getSelection());
		const std::string currentHistogram = _source->currentHistogramType();
		if (!currentHistogram.empty() && transformed->hasData(currentHistogram))
			transformed->setCurrentHistogramType(currentHistogram);
		transformed->setSelected(_source->isSelected());
		transformed->setHiLow(_source->isHiLow());
		return transformed;
	}

	bool createTransformedDetectionSet(MyMultipleObject* _object, poca::core::CommandExecutionResult& _result)
	{
		poca::core::MyObjectInterface* child = _object != nullptr ? _object->currentObject() : nullptr;
		if (child == nullptr || !child->hasBasicComponent("DetectionSet"))
			return false;

		poca::core::MyObjectInterface* object = createdObject(_result);
		if (object != nullptr && object->hasBasicComponent("DetectionSet"))
			return true;

		poca::geometry::DetectionSet* source = dynamic_cast<poca::geometry::DetectionSet*>(child->getBasicComponent("DetectionSet"));
		poca::geometry::DetectionSet* transformed = transformedDetectionSet(source, currentObjectModel(_object));
		if (transformed == nullptr)
			return false;

		return addComponentToTransformedObject(_result, child, transformed) != nullptr;
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
		_object->notify("LoadObjCharacteristicsAllWidgets");
		_object->notifyAll("updateDisplay");
	}
}

DetectionSetMultiObjectDisplayCommand::DetectionSetMultiObjectDisplayCommand(MyMultipleObject* _object)
	: poca::opengl::BasicDisplayCommand(nullptr, "DetectionSetMultiObjectDisplayCommand"), m_object(_object),
	m_textureLutID(0), m_minOriginalFeature(0.f), m_maxOriginalFeature(1.f), m_currentMinOriginalFeature(0.f),
	m_currentMaxOriginalFeature(1.f), m_actualValueFeature(1.f), m_isScaleLUT(false)
{
}

DetectionSetMultiObjectDisplayCommand::DetectionSetMultiObjectDisplayCommand(const DetectionSetMultiObjectDisplayCommand& _o)
	: poca::opengl::BasicDisplayCommand(_o), m_object(_o.m_object), m_textureLutID(0), m_minOriginalFeature(0.f), m_maxOriginalFeature(1.f),
	m_currentMinOriginalFeature(0.f), m_currentMaxOriginalFeature(1.f), m_actualValueFeature(1.f), m_isScaleLUT(false)
{
}

DetectionSetMultiObjectDisplayCommand::~DetectionSetMultiObjectDisplayCommand()
{
	freeGPUMemory();
}

void DetectionSetMultiObjectDisplayCommand::execute(poca::core::CommandInfo* _infos)
{
	poca::core::CommandExecutionContext context;
	poca::core::CommandExecutionResult result;
	execute(_infos, context, result);
}

void DetectionSetMultiObjectDisplayCommand::execute(poca::core::CommandInfo* _infos, const poca::core::CommandExecutionContext& _context, poca::core::CommandExecutionResult& _result)
{
	poca::opengl::BasicDisplayCommand::execute(_infos);

	if (_infos->nameCommand == "display") {
		poca::opengl::Camera* cam = nullptr;
		if (_context.has<poca::opengl::ActiveCamera>())
			cam = _context.get<poca::opengl::ActiveCamera>().camera;
		if (!cam) return;
		const bool offscreen = _infos->hasParameter("offscreen") && _infos->getParameter<bool>("offscreen");
		const bool ssao = _infos->hasParameter("ssao") && _infos->getParameter<bool>("ssao");
		display(cam, offscreen, ssao, _result);
	}
	else if (_infos->nameCommand == "pick") {
		if (!canBatch()) return;
		const QString infos = getInfosLocalization(m_idSelection);
		if (!infos.isEmpty()) {
			poca::core::stringList listInfos;
			if (_result.has<poca::opengl::PickedInfoListResult>())
				listInfos = _result.get<poca::opengl::PickedInfoListResult>().infos;
			listInfos.push_back(infos.toLatin1().data());
			_result.set(poca::opengl::PickedInfoListResult{ listInfos });
		}
		if (m_idSelection >= 0 && (size_t)m_idSelection < m_pickMap.size()) {
			const auto& picked = m_pickMap[m_idSelection];
			appendGlobalObjectInfo(_result, m_object, picked.objectIndex);
			handlePickedObjectSelection(_infos, m_object, picked.objectIndex);
			poca::core::MyObjectInterface* child = m_object->getObject(picked.objectIndex);
			poca::geometry::DetectionSet* dset = dynamic_cast<poca::geometry::DetectionSet*>(child->getBasicComponent("DetectionSet"));
			if (dset) {
				const glm::mat4 model = glm::inverse(m_object->getModelMatrix()) * child->getModelMatrix();
				float x = dset->getMyData("x")->getData<float>()[picked.localIndex];
				float y = dset->getMyData("y")->getData<float>()[picked.localIndex];
				float z = dset->dimension() == 3 ? dset->getMyData("z")->getData<float>()[picked.localIndex] : 0.f;
				std::vector<poca::core::Vec3mf> pickedPoints;
				if (_result.has<poca::opengl::PickedPointsResult>())
					pickedPoints = _result.get<poca::opengl::PickedPointsResult>().points;
				pickedPoints.push_back(transformPosition(model, poca::core::Vec3mf(x, y, z)));
				_result.set(poca::opengl::PickedPointsResult{ pickedPoints });
			}
		}
		markComponentFamilyHandled(_result);
	}
	else if (_infos->nameCommand == "changeLUT" || _infos->nameCommand == "regenerateDisplay") {
		freeGPUMemory();
	}
	else if (_infos->nameCommand == "histogram" || _infos->nameCommand == "updateFeature") {
		if (!updateFeatureBuffer())
			freeGPUMemory();
	}
	else if (_infos->nameCommand == "updateTransform") {
		if (!refreshTransformBuffers())
			freeGPUMemory();
	}
	else if (_infos->nameCommand == "createObjectFromCurrentTransform") {
		createTransformedDetectionSet(m_object, _result);
	}
	else if (_infos->nameCommand == "freeGPU") {
		freeGPUMemory();
	}
	else if (_infos->nameCommand == "rebuildDisplay") {
		rebuild();
	}
}

poca::core::Command* DetectionSetMultiObjectDisplayCommand::copy()
{
	return new DetectionSetMultiObjectDisplayCommand(*this);
}

bool DetectionSetMultiObjectDisplayCommand::canBatch() const
{
	if (m_object == nullptr || m_object->nbColors() <= 1)
		return false;

	bool hasDetectionSetChild = false, hasNormals = false, hasColors = false, first = true;
	for (size_t n = 0; n < m_object->nbColors(); n++) {
		poca::core::MyObjectInterface* child = m_object->getObject(n);
		poca::geometry::DetectionSet* dset = dynamic_cast<poca::geometry::DetectionSet*>(child->getBasicComponent("DetectionSet"));
		if (dset == nullptr)
			continue;

		hasDetectionSetChild = true;
		const bool childHasNormals = dset->hasData("nx") && dset->hasData("ny") && dset->hasData("nz");
		const bool childHasColors = dset->hasData("r") && dset->hasData("g") && dset->hasData("b");
		if (first) {
			hasNormals = childHasNormals;
			hasColors = childHasColors;
			first = false;
		}
		else if (hasNormals != childHasNormals || hasColors != childHasColors)
			return false;
	}
	return hasDetectionSetChild;
}

DetectionSetDisplayCommand* DetectionSetMultiObjectDisplayCommand::referenceDisplayCommand() const
{
	if (m_object == nullptr) return nullptr;
	for (size_t n = 0; n < m_object->nbColors(); n++) {
		poca::core::MyObjectInterface* child = m_object->getObject(n);
		poca::geometry::DetectionSet* dset = dynamic_cast<poca::geometry::DetectionSet*>(child->getBasicComponent("DetectionSet"));
		if (dset == nullptr)
			continue;
		return dset->getCommand<DetectionSetDisplayCommand>();
	}
	return nullptr;
}

bool DetectionSetMultiObjectDisplayCommand::rebuild()
{
	freeGPUMemory();
	if (!canBatch())
		return false;

	DetectionSetDisplayCommand* referenceCommand = referenceDisplayCommand();
	if (referenceCommand == nullptr)
		return false;

	std::vector<poca::core::Vec3mf> points, normals;
	std::vector<float> ids, features;
	std::vector<poca::core::Color4D> colors;
	m_pickMap.clear();

	m_minOriginalFeature = std::numeric_limits<float>::max();
	m_maxOriginalFeature = std::numeric_limits<float>::lowest();
	m_currentMinOriginalFeature = std::numeric_limits<float>::max();
	m_currentMaxOriginalFeature = std::numeric_limits<float>::lowest();
	m_isScaleLUT = false;

	for (size_t objectIndex = 0; objectIndex < m_object->nbColors(); objectIndex++) {
		poca::core::MyObjectInterface* child = m_object->getObject(objectIndex);
		poca::geometry::DetectionSet* dset = dynamic_cast<poca::geometry::DetectionSet*>(child->getBasicComponent("DetectionSet"));
		if (dset == nullptr)
			continue;

		poca::core::HistogramInterface* histInterface = dset->getCurrentHistogram();
		poca::core::Histogram<float>* histogram = dynamic_cast<poca::core::Histogram<float>*>(histInterface);
		if (histogram == nullptr)
			continue;

		const glm::mat4 model = glm::inverse(m_object->getModelMatrix()) * child->getModelMatrix();
		const std::vector<float>& xs = dset->getMyData("x")->getData<float>();
		const std::vector<float>& ys = dset->getMyData("y")->getData<float>();
		const std::vector<float>* zs = dset->hasData("z") ? &dset->getMyData("z")->getData<float>() : nullptr;
		const std::vector<float>& values = histogram->getValues();
		const std::vector<bool>& selection = dset->getSelection();
		const size_t base = m_pickMap.size();

		m_minOriginalFeature = std::min(m_minOriginalFeature, histInterface->getMin());
		m_maxOriginalFeature = std::max(m_maxOriginalFeature, histInterface->getMax());
		m_currentMinOriginalFeature = std::min(m_currentMinOriginalFeature, histInterface->getCurrentMin());
		m_currentMaxOriginalFeature = std::max(m_currentMaxOriginalFeature, histInterface->getCurrentMax());
		m_isScaleLUT = m_isScaleLUT || histInterface->scaleLUT();

		for (size_t idx = 0; idx < xs.size(); idx++) {
			const float z = zs != nullptr ? (*zs)[idx] : 0.f;
			points.push_back(transformPosition(model, poca::core::Vec3mf(xs[idx], ys[idx], z)));
			ids.push_back((float)(base + idx + 1));
			features.push_back(dset->isSelected() && selection[idx] ? values[idx] : -10000.f);
			m_pickMap.push_back({ (uint32_t)objectIndex, (uint32_t)idx });
		}

		if (dset->hasData("nx") && dset->hasData("ny") && dset->hasData("nz")) {
			const std::vector<float>& nxs = dset->getMyData("nx")->getData<float>();
			const std::vector<float>& nys = dset->getMyData("ny")->getData<float>();
			const std::vector<float>& nzs = dset->getMyData("nz")->getData<float>();
			for (size_t idx = 0; idx < nxs.size(); idx++)
				normals.push_back(transformDirection(model, poca::core::Vec3mf(nxs[idx], nys[idx], nzs[idx])));
		}

		if (dset->hasData("r") && dset->hasData("g") && dset->hasData("b")) {
			const std::vector<float>& rs = dset->getMyData("r")->getData<float>();
			const std::vector<float>& gs = dset->getMyData("g")->getData<float>();
			const std::vector<float>& bs = dset->getMyData("b")->getData<float>();
			for (size_t idx = 0; idx < rs.size(); idx++)
				colors.emplace_back(rs[idx] / 255.f, gs[idx] / 255.f, bs[idx] / 255.f, 1.f);
		}
	}

	if (points.empty())
		return false;
	if (m_minOriginalFeature == std::numeric_limits<float>::max()) {
		m_minOriginalFeature = 0.f;
		m_maxOriginalFeature = 1.f;
		m_currentMinOriginalFeature = 0.f;
		m_currentMaxOriginalFeature = 1.f;
	}
	m_actualValueFeature = m_maxOriginalFeature;

	poca::geometry::DetectionSet* referenceDset = nullptr;
	for (size_t n = 0; n < m_object->nbColors() && referenceDset == nullptr; n++) {
		poca::core::MyObjectInterface* child = m_object->getObject(n);
		referenceDset = dynamic_cast<poca::geometry::DetectionSet*>(child->getBasicComponent("DetectionSet"));
	}
	if (referenceDset == nullptr)
		return false;

	m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(referenceDset->getPalette());
	m_pointBuffer.generateBuffer(points.size(), 3, GL_FLOAT);
	m_idBuffer.generateBuffer(ids.size(), 1, GL_FLOAT);
	m_featureBuffer.generateBuffer(features.size(), 1, GL_FLOAT);
	m_pointBuffer.updateBuffer(points.data());
	m_idBuffer.updateBuffer(ids.data());
	m_featureBuffer.updateBuffer(features.data());

	if (!normals.empty()) {
		m_normalBuffer.generateBuffer(normals.size(), 3, GL_FLOAT);
		m_normalBuffer.updateBuffer(normals.data());
	}
	if (!colors.empty()) {
		m_colorBuffer.generateBuffer(colors.size(), 4, GL_FLOAT);
		m_colorBuffer.updateBuffer(colors.data());
	}
	return true;
}

bool DetectionSetMultiObjectDisplayCommand::refreshTransformBuffers()
{
	if (!canBatch())
		return false;
	if (m_pointBuffer.empty())
		return true;

	std::vector<poca::core::Vec3mf> points, normals;
	const bool updateNormals = !m_normalBuffer.empty();
	const glm::mat4 parentInvModel = glm::inverse(m_object->getModelMatrix());

	for (size_t objectIndex = 0; objectIndex < m_object->nbColors(); objectIndex++) {
		poca::core::MyObjectInterface* child = m_object->getObject(objectIndex);
		poca::geometry::DetectionSet* dset = dynamic_cast<poca::geometry::DetectionSet*>(child->getBasicComponent("DetectionSet"));
		if (dset == nullptr)
			continue;

		const glm::mat4 model = parentInvModel * child->getModelMatrix();
		const std::vector<float>& xs = dset->getMyData("x")->getData<float>();
		const std::vector<float>& ys = dset->getMyData("y")->getData<float>();
		const std::vector<float>* zs = dset->hasData("z") ? &dset->getMyData("z")->getData<float>() : nullptr;
		for (size_t idx = 0; idx < xs.size(); idx++) {
			const float z = zs != nullptr ? (*zs)[idx] : 0.f;
			points.push_back(transformPosition(model, poca::core::Vec3mf(xs[idx], ys[idx], z)));
		}

		if (updateNormals) {
			if (!dset->hasData("nx") || !dset->hasData("ny") || !dset->hasData("nz"))
				return false;
			const std::vector<float>& nxs = dset->getMyData("nx")->getData<float>();
			const std::vector<float>& nys = dset->getMyData("ny")->getData<float>();
			const std::vector<float>& nzs = dset->getMyData("nz")->getData<float>();
			for (size_t idx = 0; idx < nxs.size(); idx++)
				normals.push_back(transformDirection(model, poca::core::Vec3mf(nxs[idx], nys[idx], nzs[idx])));
		}
	}

	if (points.size() != m_pointBuffer.getNbElements())
		return false;
	m_pointBuffer.updateBuffer(points);

	if (updateNormals) {
		if (normals.size() != m_normalBuffer.getNbElements())
			return false;
		m_normalBuffer.updateBuffer(normals);
	}
	return true;
}

void DetectionSetMultiObjectDisplayCommand::display(poca::opengl::Camera* _cam, const bool _offscreen, const bool _ssao, poca::core::CommandExecutionResult& _result)
{
	if (!canBatch())
		return;
	if (m_pointBuffer.empty() && !rebuild())
		return;
	// Keep batched multi-object vertices synchronized with per-child model matrices.
	// Gizmo transforms update the child object model; if the command notification is
	// skipped by a component-specific display path, refreshing here still applies the
	// current transform before drawing.
	if (!m_pointBuffer.empty() && !refreshTransformBuffers()) {
		freeGPUMemory();
		if (!rebuild())
			return;
	}

	DetectionSetDisplayCommand* referenceCommand = referenceDisplayCommand();
	if (referenceCommand == nullptr)
		return;

	const bool pointRendering = referenceCommand->getParameter<bool>("pointRendering");
	if (!pointRendering)
		return;

	drawElements(_cam, _ssao, referenceCommand);
	if (!_offscreen)
		drawPicking(_cam);
	markComponentFamilyHandled(_result);
}

bool DetectionSetMultiObjectDisplayCommand::updateFeatureBuffer()
{
	if (!canBatch() || m_featureBuffer.empty())
		return false;

	std::vector<float> features;
	m_minOriginalFeature = std::numeric_limits<float>::max();
	m_maxOriginalFeature = std::numeric_limits<float>::lowest();
	m_currentMinOriginalFeature = std::numeric_limits<float>::max();
	m_currentMaxOriginalFeature = std::numeric_limits<float>::lowest();
	m_isScaleLUT = false;

	for (size_t objectIndex = 0; objectIndex < m_object->nbColors(); objectIndex++) {
		poca::core::MyObjectInterface* child = m_object->getObject(objectIndex);
		poca::geometry::DetectionSet* dset = dynamic_cast<poca::geometry::DetectionSet*>(child->getBasicComponent("DetectionSet"));
		if (dset == nullptr)
			continue;

		poca::core::HistogramInterface* histInterface = dset->getCurrentHistogram();
		poca::core::Histogram<float>* histogram = dynamic_cast<poca::core::Histogram<float>*>(histInterface);
		if (histogram == nullptr)
			return false;

		m_minOriginalFeature = std::min(m_minOriginalFeature, histInterface->getMin());
		m_maxOriginalFeature = std::max(m_maxOriginalFeature, histInterface->getMax());
		m_currentMinOriginalFeature = std::min(m_currentMinOriginalFeature, histInterface->getCurrentMin());
		m_currentMaxOriginalFeature = std::max(m_currentMaxOriginalFeature, histInterface->getCurrentMax());
		m_isScaleLUT = m_isScaleLUT || histInterface->scaleLUT();

		const std::vector<float>& values = histogram->getValues();
		const std::vector<bool>& selection = dset->getSelection();
		for (size_t idx = 0; idx < values.size(); idx++)
			features.push_back(dset->isSelected() && selection[idx] ? values[idx] : -10000.f);
	}

	if (m_minOriginalFeature == std::numeric_limits<float>::max()) {
		m_minOriginalFeature = 0.f;
		m_maxOriginalFeature = 1.f;
		m_currentMinOriginalFeature = 0.f;
		m_currentMaxOriginalFeature = 1.f;
	}
	m_actualValueFeature = m_maxOriginalFeature;
	m_featureBuffer.updateBuffer(features);
	return true;
}

void DetectionSetMultiObjectDisplayCommand::drawElements(poca::opengl::Camera* _cam, const bool _ssao, DetectionSetDisplayCommand* _referenceCommand)
{
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glCullFace(GL_FRONT);

	const uint32_t pointSize = _referenceCommand->getParameter<uint32_t>("pointSizeGL");
	const bool screenCoordinates = _referenceCommand->getParameter<bool>("screenCoordinates");
	if (!m_colorBuffer.empty())
		_cam->drawSimpleShaderWithColor<poca::core::Vec3mf, poca::core::Color4D>(m_pointBuffer, m_colorBuffer);
	else if (m_normalBuffer.empty())
		_cam->drawSphereRendering<poca::core::Vec3mf, float>(m_textureLutID, m_pointBuffer, m_featureBuffer,
			m_minOriginalFeature, m_maxOriginalFeature, m_currentMinOriginalFeature, m_currentMaxOriginalFeature,
			m_isScaleLUT, pointSize, _ssao, screenCoordinates);
	else
		_cam->drawSphereRendering<poca::core::Vec3mf, float>(m_textureLutID, m_pointBuffer, m_normalBuffer, m_featureBuffer,
			m_minOriginalFeature, m_maxOriginalFeature, m_currentMinOriginalFeature, m_currentMaxOriginalFeature,
			m_isScaleLUT, pointSize, _ssao, screenCoordinates);
}

void DetectionSetMultiObjectDisplayCommand::drawPicking(poca::opengl::Camera* _cam)
{
	if (m_pickFBO == nullptr)
		updatePickingFBO(_cam->getWidth(), _cam->getHeight());
	if (m_pickFBO == nullptr)
		return;

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glDisable(GL_BLEND);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_CULL_FACE);

	GLfloat bkColor[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	const bool success = m_pickFBO->bind();
	if (!success) std::cout << "Problem with binding" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	_cam->drawPickingShader<poca::core::Vec3mf, float>(m_pointBuffer, m_idBuffer, m_featureBuffer, m_minOriginalFeature);
	m_pickFBO->release();
	glBindFramebuffer(GL_FRAMEBUFFER, _cam->defaultFramebufferObject());
	glClearColor(bkColor[0], bkColor[1], bkColor[2], bkColor[3]);
}

QString DetectionSetMultiObjectDisplayCommand::getInfosLocalization(const int _id) const
{
	if (_id < 0 || (size_t)_id >= m_pickMap.size())
		return QString();

	const auto& picked = m_pickMap[_id];
	poca::core::MyObjectInterface* child = m_object->getObject(picked.objectIndex);
	poca::geometry::DetectionSet* dset = dynamic_cast<poca::geometry::DetectionSet*>(child->getBasicComponent("DetectionSet"));
	if (dset == nullptr)
		return QString();

	QString text;
	poca::core::stringList nameData = dset->getNameData();
	float x = dset->getMyData("x")->getData<float>()[picked.localIndex];
	float y = dset->getMyData("y")->getData<float>()[picked.localIndex];
	text.append(QString("Localization id: %1\n").arg(picked.localIndex));
	text.append(QString("Coords: [x=%1,y=%2").arg(x).arg(y));
	if (dset->hasData("z")) {
		float z = dset->getMyData("z")->getData<float>()[picked.localIndex];
		text.append(QString(",z=%1").arg(z));
	}
	text.append("]");
	for (const std::string& type : nameData) {
		if (type == "x" || type == "y" || type == "z") continue;
		float val = dset->getMyData(type)->getData<float>()[picked.localIndex];
		text.append(QString("\n%1: %2").arg(type.c_str()).arg(val));
	}
	return text;
}

void DetectionSetMultiObjectDisplayCommand::freeGPUMemory()
{
	m_pointBuffer.freeGPUMemory();
	m_normalBuffer.freeGPUMemory();
	m_idBuffer.freeGPUMemory();
	m_featureBuffer.freeGPUMemory();
	m_colorBuffer.freeGPUMemory();
	m_pickMap.clear();
	m_textureLutID = 0;
}
