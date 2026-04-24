/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListMultiObjectDisplayCommand.cpp
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
#include <General/Histogram.hpp>
#include <General/Misc.h>
#include <Geometry/ObjectListMesh.hpp>
#include <Geometry/ObjectLists.hpp>
#include <Interfaces/ObjectListInterface.hpp>
#include <OpenGL/Camera.hpp>
#include <OpenGL/Helper.h>
#include <OpenGL/RenderCommandContext.hpp>
#include <OpenGL/Shader.hpp>

#include "ObjectListDisplayCommand.hpp"
#include "ObjectListMultiObjectDisplayCommand.hpp"

namespace {
	const std::string kRenderedComponentName = "ObjectLists";
	const std::string kRenderedSubComponentName = "ObjectList";

	void markComponentFamilyHandled(poca::core::CommandExecutionResult& _result)
	{
		poca::opengl::RenderedComponentFamilies families;
		if (_result.has<poca::opengl::RenderedComponentFamilies>())
			families = _result.get<poca::opengl::RenderedComponentFamilies>();
		if (std::find(families.componentNames.begin(), families.componentNames.end(), kRenderedComponentName) == families.componentNames.end())
			families.componentNames.push_back(kRenderedComponentName);
		if (std::find(families.componentNames.begin(), families.componentNames.end(), kRenderedSubComponentName) == families.componentNames.end())
			families.componentNames.push_back(kRenderedSubComponentName);
		_result.set(families);
		_result.set(poca::opengl::ChildObjectRenderingHandled{ true });
	}

	poca::core::Color4D deterministicColor(const size_t _objectIndex, const size_t _localIndex)
	{
		const uint32_t seed = (uint32_t)((_objectIndex + 1) * 2654435761u + (_localIndex + 1) * 2246822519u);
		return poca::core::Color4D(
			(float)((seed >> 0) & 0xFF) / 255.f,
			(float)((seed >> 8) & 0xFF) / 255.f,
			(float)((seed >> 16) & 0xFF) / 255.f,
			1.f);
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

	poca::core::Vec3mf bboxCenter(const poca::core::BoundingBox& _bbox)
	{
		return poca::core::Vec3mf((_bbox[0] + _bbox[3]) * .5f, (_bbox[1] + _bbox[4]) * .5f, (_bbox[2] + _bbox[5]) * .5f);
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
		_object->notify("UpdateMainTabWidgets");
		_object->notifyAll("updateDisplay");
	}

	template<typename F>
	void forEachObjectList(MyMultipleObject* _object, F&& _callback)
	{
		if (_object == nullptr)
			return;

		for (size_t objectIndex = 0; objectIndex < _object->nbColors(); objectIndex++) {
			poca::core::MyObjectInterface* child = _object->getObject(objectIndex);
			if (child == nullptr)
				continue;

			poca::geometry::ObjectLists* lists = dynamic_cast<poca::geometry::ObjectLists*>(child->getBasicComponent("ObjectLists"));
			if (lists == nullptr)
				continue;

			for (uint32_t listIndex = 0; listIndex < lists->nbComponents(); listIndex++) {
				poca::geometry::ObjectListInterface* objs = dynamic_cast<poca::geometry::ObjectListInterface*>(lists->getObjectList(listIndex));
				if (objs != nullptr)
					_callback(objectIndex, child, objs);
			}
		}
	}
}

ObjectListMultiObjectDisplayCommand::ObjectListMultiObjectDisplayCommand(MyMultipleObject* _object)
	: poca::opengl::BasicDisplayCommand(nullptr, "ObjectListMultiObjectDisplayCommand"), m_object(_object),
	m_hasOutlinePoints(false), m_has2DOutlines(false), m_is3D(false), m_textureLutID(0),
	m_minOriginalFeature(0.f), m_maxOriginalFeature(1.f), m_actualValueFeature(1.f), m_objectModelTextureID(0)
{
	m_hasEllipsoids = false;
}

ObjectListMultiObjectDisplayCommand::ObjectListMultiObjectDisplayCommand(const ObjectListMultiObjectDisplayCommand& _o)
	: poca::opengl::BasicDisplayCommand(_o), m_object(_o.m_object), m_hasOutlinePoints(false), m_has2DOutlines(false),
	m_is3D(false), m_textureLutID(0), m_minOriginalFeature(0.f), m_maxOriginalFeature(1.f), m_actualValueFeature(1.f), m_objectModelTextureID(0)
{
	m_hasEllipsoids = false;
}

ObjectListMultiObjectDisplayCommand::~ObjectListMultiObjectDisplayCommand()
{
	freeGPUMemory();
}

void ObjectListMultiObjectDisplayCommand::execute(poca::core::CommandInfo* _infos)
{
	poca::core::CommandExecutionContext context;
	poca::core::CommandExecutionResult result;
	execute(_infos, context, result);
}

void ObjectListMultiObjectDisplayCommand::execute(poca::core::CommandInfo* _infos, const poca::core::CommandExecutionContext& _context, poca::core::CommandExecutionResult& _result)
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
		if (m_idSelection >= 0 && (size_t)m_idSelection < m_object->nbColors()) {
			generateBoundingBoxSelection(m_idSelection);
			const size_t objectIndex = (size_t)m_idSelection;
			appendGlobalObjectInfo(_result, m_object, objectIndex);
			handlePickedObjectSelection(_infos, m_object, objectIndex);
			poca::core::MyObjectInterface* child = m_object->getObject(objectIndex);
			if (child != nullptr) {
				const glm::mat4 model = glm::inverse(m_object->getModelMatrix()) * child->getModelMatrix();
				std::vector<poca::core::Vec3mf> pickedPoints;
				if (_result.has<poca::opengl::PickedPointsResult>())
					pickedPoints = _result.get<poca::opengl::PickedPointsResult>().points;
				pickedPoints.push_back(transformPosition(model, bboxCenter(child->boundingBox())));
				_result.set(poca::opengl::PickedPointsResult{ pickedPoints });
			}
		}
		markComponentFamilyHandled(_result);
	}
	else if (_infos->nameCommand == "changeLUT") {
		freeGPUMemory();
	}
	else if (_infos->nameCommand == "histogram" || _infos->nameCommand == "updateFeature") {
		if (!updateFeatureBuffers())
			freeGPUMemory();
	}
	else if (_infos->nameCommand == "updateTransform") {
		if (!refreshTransformBuffers())
			freeGPUMemory();
	}
	else if (_infos->nameCommand == "useVertexNormals") {
		freeGPUMemory();
	}
	else if (_infos->nameCommand == "freeGPU") {
		freeGPUMemory();
	}
}

poca::core::Command* ObjectListMultiObjectDisplayCommand::copy()
{
	return new ObjectListMultiObjectDisplayCommand(*this);
}

bool ObjectListMultiObjectDisplayCommand::canBatch() const
{
	if (m_object == nullptr || m_object->nbColors() <= 1)
		return false;

	bool hasObjectListChild = false, first = true;
	uint32_t dimension = 0;
	forEachObjectList(m_object, [&](size_t, poca::core::MyObjectInterface*, poca::geometry::ObjectListInterface* objs) {
		hasObjectListChild = true;
		if (first) {
			dimension = objs->dimension();
			first = false;
		}
		else if (dimension != objs->dimension())
			hasObjectListChild = false;
	});
	if (!hasObjectListChild || first)
		return false;
	bool consistentDimensions = true;
	forEachObjectList(m_object, [&](size_t, poca::core::MyObjectInterface*, poca::geometry::ObjectListInterface* objs) {
		if (objs->dimension() != dimension)
			consistentDimensions = false;
	});
	if (!consistentDimensions)
		return false;
	return hasObjectListChild;
}

ObjectListDisplayCommand* ObjectListMultiObjectDisplayCommand::referenceDisplayCommand() const
{
	if (m_object == nullptr) return nullptr;
	ObjectListDisplayCommand* reference = nullptr;
	forEachObjectList(m_object, [&](size_t, poca::core::MyObjectInterface*, poca::geometry::ObjectListInterface* objs) {
		if (reference == nullptr)
			reference = objs->getCommand<ObjectListDisplayCommand>();
	});
	return reference;
}

bool ObjectListMultiObjectDisplayCommand::rebuild()
{
	freeGPUMemory();
	if (!canBatch())
		return false;

	std::vector<poca::core::Vec3mf> points, outlinePoints, triangles, triangleNormals, lines, skeletons, links;
	std::vector<float> pointIds, pointFeatures, outlinePointFeatures, triangleIds, triangleFeatures, lineFeatures, ellipsoidFeatures;
	std::vector<float> pointObjectIndices, outlinePointObjectIndices, triangleObjectIndices, lineObjectIndices, skeletonObjectIndices, linkObjectIndices, ellipsoidObjectIndices;
	std::vector<poca::core::Color4D> colorSkeletons, colorLinks;
	std::vector<glm::mat4> ellipsoidTransforms;
	m_minOriginalFeature = std::numeric_limits<float>::max();
	m_maxOriginalFeature = std::numeric_limits<float>::lowest();
	m_hasOutlinePoints = false;
	m_has2DOutlines = false;
	m_is3D = false;
	m_hasEllipsoids = false;

	forEachObjectList(m_object, [&](size_t objectIndex, poca::core::MyObjectInterface* child, poca::geometry::ObjectListInterface* objs) {
		poca::core::HistogramInterface* histInterface = objs->getCurrentHistogram();
		poca::core::Histogram<float>* histogram = dynamic_cast<poca::core::Histogram<float>*>(histInterface);
		if (histogram == nullptr)
			return;
		const std::vector<float>& values = histogram->getValues();
		const std::vector<bool>& selection = objs->getSelection();

		m_is3D = objs->dimension() == 3;
		m_minOriginalFeature = std::min(m_minOriginalFeature, histInterface->getMin());
		m_maxOriginalFeature = std::max(m_maxOriginalFeature, histInterface->getMax());

		std::vector<poca::core::Vec3mf> localPoints;
		objs->generateLocs(localPoints);
		points.insert(points.end(), localPoints.begin(), localPoints.end());
		pointObjectIndices.insert(pointObjectIndices.end(), localPoints.size(), (float)objectIndex);
		std::vector<float> localPointIds;
		objs->generateLocsPickingIndices(localPointIds);
		pointIds.insert(pointIds.end(), localPointIds.size(), (float)(objectIndex + 1));
		std::vector<float> localPointFeatures;
		if (objs->isHiLow()) {
			float inter = histInterface->getMax() - histInterface->getMin();
			float selectedValue = histInterface->getMin() + inter / 4.f;
			float notSelectedValue = histInterface->getMin() + inter * (3.f / 4.f);
			objs->getLocsFeatureInSelectionHiLow(localPointFeatures, selection, selectedValue, notSelectedValue);
		}
		else {
			objs->getLocsFeatureInSelection(localPointFeatures, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		}
		if (!objs->isSelected())
			std::fill(localPointFeatures.begin(), localPointFeatures.end(), poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		pointFeatures.insert(pointFeatures.end(), localPointFeatures.begin(), localPointFeatures.end());

		std::vector<poca::core::Vec3mf> localOutlinePoints;
		objs->generateOutlineLocs(localOutlinePoints);
		if (!localOutlinePoints.empty()) {
			m_hasOutlinePoints = true;
			outlinePoints.insert(outlinePoints.end(), localOutlinePoints.begin(), localOutlinePoints.end());
			outlinePointObjectIndices.insert(outlinePointObjectIndices.end(), localOutlinePoints.size(), (float)objectIndex);
			std::vector<float> localOutlinePointFeatures;
			if (objs->isHiLow()) {
				float inter = histInterface->getMax() - histInterface->getMin();
				float selectedValue = histInterface->getMin() + inter / 4.f;
				float notSelectedValue = histInterface->getMin() + inter * (3.f / 4.f);
				objs->getOutlineLocsFeatureInSelectionHiLow(localOutlinePointFeatures, selection, selectedValue, notSelectedValue);
			}
			else {
				objs->getOutlineLocsFeatureInSelection(localOutlinePointFeatures, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
			}
			if (!objs->isSelected())
				std::fill(localOutlinePointFeatures.begin(), localOutlinePointFeatures.end(), poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
			outlinePointFeatures.insert(outlinePointFeatures.end(), localOutlinePointFeatures.begin(), localOutlinePointFeatures.end());
		}

		std::vector<poca::core::Vec3mf> localTriangles;
		objs->generateTriangles(localTriangles);
		triangles.insert(triangles.end(), localTriangles.begin(), localTriangles.end());
		triangleObjectIndices.insert(triangleObjectIndices.end(), localTriangles.size(), (float)objectIndex);
		std::vector<poca::core::Vec3mf> localNormals;
		objs->generateNormals(localNormals);
		triangleNormals.insert(triangleNormals.end(), localNormals.begin(), localNormals.end());
		std::vector<float> localTriangleIds;
		objs->generatePickingIndices(localTriangleIds);
		triangleIds.insert(triangleIds.end(), localTriangleIds.size(), (float)(objectIndex + 1));
		std::vector<float> localTriangleFeatures;
		if (objs->isHiLow()) {
			float inter = histInterface->getMax() - histInterface->getMin();
			float selectedValue = histInterface->getMin() + inter / 4.f;
			float notSelectedValue = histInterface->getMin() + inter * (3.f / 4.f);
			objs->getFeatureInSelectionHiLow(localTriangleFeatures, selection, selectedValue, notSelectedValue);
		}
		else {
			objs->getFeatureInSelection(localTriangleFeatures, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		}
		if (!objs->isSelected())
			std::fill(localTriangleFeatures.begin(), localTriangleFeatures.end(), poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		triangleFeatures.insert(triangleFeatures.end(), localTriangleFeatures.begin(), localTriangleFeatures.end());

		if (objs->dimension() == 2) {
			std::vector<poca::core::Vec3mf> localLines;
			objs->generateOutlines(localLines);
			if (!localLines.empty()) {
				m_has2DOutlines = true;
				lines.insert(lines.end(), localLines.begin(), localLines.end());
				lineObjectIndices.insert(lineObjectIndices.end(), localLines.size(), (float)objectIndex);
				std::vector<float> localLineFeatures;
				if (objs->isHiLow()) {
					float inter = histInterface->getMax() - histInterface->getMin();
					float selectedValue = histInterface->getMin() + inter / 4.f;
					float notSelectedValue = histInterface->getMin() + inter * (3.f / 4.f);
					objs->getOutlinesFeatureInSelectionHiLow(localLineFeatures, selection, selectedValue, notSelectedValue);
				}
				else {
					objs->getOutlinesFeatureInSelection(localLineFeatures, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
				}
				if (!objs->isSelected())
					std::fill(localLineFeatures.begin(), localLineFeatures.end(), poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
				lineFeatures.insert(lineFeatures.end(), localLineFeatures.begin(), localLineFeatures.end());
			}
		}

		if (objs->hasSkeletons()) {
			poca::geometry::ObjectListMesh* omesh = dynamic_cast<poca::geometry::ObjectListMesh*>(objs);
			if (omesh != nullptr) {
				const poca::core::MyArrayVec3mf& localSkeletons = omesh->getSkeletons();
				const poca::core::MyArrayVec3mf& localLinks = omesh->getLinks();
				const std::vector<uint32_t>& firstSkeletons = localSkeletons.getFirstElements();
				const std::vector<uint32_t>& firstLinks = localLinks.getFirstElements();
				const std::vector<poca::core::Vec3mf>& skeletonData = localSkeletons.getData();
				const std::vector<poca::core::Vec3mf>& linkData = localLinks.getData();
				for (size_t idx = 0; idx + 1 < firstSkeletons.size(); idx++) {
					const poca::core::Color4D color = deterministicColor(objectIndex, idx);
					for (size_t cur = firstSkeletons[idx]; cur < firstSkeletons[idx + 1]; cur++) {
						skeletons.push_back(skeletonData[cur]);
						skeletonObjectIndices.push_back((float)objectIndex);
						colorSkeletons.push_back(color);
					}
				}
				for (size_t idx = 0; idx + 1 < firstLinks.size(); idx++) {
					const poca::core::Color4D color = deterministicColor(objectIndex, idx);
					for (size_t cur = firstLinks[idx]; cur < firstLinks[idx + 1]; cur++) {
						links.push_back(linkData[cur]);
						linkObjectIndices.push_back((float)objectIndex);
						colorLinks.push_back(color);
					}
				}
			}
		}

		if (objs->dimension() == 3 && objs->hasData("major") && objs->hasData("minor") && objs->hasData("minor2")) {
			m_hasEllipsoids = true;
			const std::vector<std::array<poca::core::Vec3mf, 3>>& axisPCA = objs->getAxisObjects();
			const std::vector<float>& major = objs->getMyData("major")->getData<float>();
			const std::vector<float>& minor = objs->getMyData("minor")->getData<float>();
			const std::vector<float>& minor2 = objs->getMyData("minor2")->getData<float>();

			std::vector<float> localEllipsoidFeatures(objs->nbElements());
			if (objs->isHiLow()) {
				float inter = histInterface->getMax() - histInterface->getMin();
				float selectedValue = histInterface->getMin() + inter / 4.f;
				float notSelectedValue = histInterface->getMin() + inter * (3.f / 4.f);
				for (size_t idx = 0; idx < objs->nbElements(); idx++)
					localEllipsoidFeatures[idx] = selection[idx] ? selectedValue : notSelectedValue;
			}
			else {
				for (size_t idx = 0; idx < objs->nbElements(); idx++)
					localEllipsoidFeatures[idx] = selection[idx] ? values[idx] : poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER;
			}
			if (!objs->isSelected())
				std::fill(localEllipsoidFeatures.begin(), localEllipsoidFeatures.end(), poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
			ellipsoidFeatures.insert(ellipsoidFeatures.end(), localEllipsoidFeatures.begin(), localEllipsoidFeatures.end());

			for (size_t idx = 0; idx < objs->nbElements(); idx++) {
				glm::mat4 matrix(1.f);
				glm::mat3 rotation = glm::mat3(
					axisPCA[idx][0].x(), axisPCA[idx][0].y(), axisPCA[idx][0].z(),
					axisPCA[idx][1].x(), axisPCA[idx][1].y(), axisPCA[idx][1].z(),
					axisPCA[idx][2].x(), axisPCA[idx][2].y(), axisPCA[idx][2].z());
				poca::core::Vec3mf centroidTmp = objs->computeBarycenterElement((int)idx);
				const glm::vec3 centroid = glm::vec3(centroidTmp[0], centroidTmp[1], centroidTmp[2]);
				const glm::vec3 scales = glm::vec3(major[idx] / 2.f, minor[idx] / 2.f, minor2[idx] / 2.f);
				matrix = glm::translate(matrix, centroid);
				matrix *= glm::mat4(rotation);
				matrix = glm::scale(matrix, scales);
				ellipsoidTransforms.push_back(matrix);
				ellipsoidObjectIndices.push_back((float)objectIndex);
			}
		}
	});

	if (triangles.empty() && points.empty())
		return false;
	if (m_minOriginalFeature == std::numeric_limits<float>::max()) {
		m_minOriginalFeature = 0.f;
		m_maxOriginalFeature = 1.f;
	}
	m_actualValueFeature = m_maxOriginalFeature;

	poca::geometry::ObjectListInterface* reference = nullptr;
	forEachObjectList(m_object, [&](size_t, poca::core::MyObjectInterface*, poca::geometry::ObjectListInterface* objs) {
		if (reference == nullptr)
			reference = objs;
	});
	if (reference == nullptr)
		return false;

	m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(reference->getPalette());
	if (!points.empty()) {
		m_pointBuffer.generateBuffer(points.size(), 3, GL_FLOAT);
		m_idLocsBuffer.generateBuffer(pointIds.size(), 1, GL_FLOAT);
		m_locsFeatureBuffer.generateBuffer(pointFeatures.size(), 1, GL_FLOAT);
		m_pointObjectIndexBuffer.generateBuffer(pointObjectIndices.size(), 1, GL_FLOAT);
		m_pointBuffer.updateBuffer(points.data());
		m_idLocsBuffer.updateBuffer(pointIds.data());
		m_locsFeatureBuffer.updateBuffer(pointFeatures.data());
		m_pointObjectIndexBuffer.updateBuffer(pointObjectIndices.data());
	}
	if (!outlinePoints.empty()) {
		m_outlinePointBuffer.generateBuffer(outlinePoints.size(), 3, GL_FLOAT);
		m_outlineLocsFeatureBuffer.generateBuffer(outlinePointFeatures.size(), 1, GL_FLOAT);
		m_outlinePointObjectIndexBuffer.generateBuffer(outlinePointObjectIndices.size(), 1, GL_FLOAT);
		m_outlinePointBuffer.updateBuffer(outlinePoints.data());
		m_outlineLocsFeatureBuffer.updateBuffer(outlinePointFeatures.data());
		m_outlinePointObjectIndexBuffer.updateBuffer(outlinePointObjectIndices.data());
	}
	if (!triangles.empty()) {
		m_triangleBuffer.generateBuffer(triangles.size(), 3, GL_FLOAT);
		m_triangleNormalBuffer.generateBuffer(triangleNormals.size(), 3, GL_FLOAT);
		m_triangleFeatureBuffer.generateBuffer(triangleFeatures.size(), 1, GL_FLOAT);
		m_idBuffer.generateBuffer(triangleIds.size(), 1, GL_FLOAT);
		m_triangleObjectIndexBuffer.generateBuffer(triangleObjectIndices.size(), 1, GL_FLOAT);
		m_triangleBuffer.updateBuffer(triangles.data());
		m_triangleNormalBuffer.updateBuffer(triangleNormals.data());
		m_triangleFeatureBuffer.updateBuffer(triangleFeatures.data());
		m_idBuffer.updateBuffer(triangleIds.data());
		m_triangleObjectIndexBuffer.updateBuffer(triangleObjectIndices.data());
		m_boundingBoxSelection.generateBuffer(24, 3, GL_FLOAT);
	}
	if (!lines.empty()) {
		m_lineBuffer.generateBuffer(lines.size(), 3, GL_FLOAT);
		m_lineFeatureBuffer.generateBuffer(lineFeatures.size(), 1, GL_FLOAT);
		m_lineObjectIndexBuffer.generateBuffer(lineObjectIndices.size(), 1, GL_FLOAT);
		m_lineBuffer.updateBuffer(lines.data());
		m_lineFeatureBuffer.updateBuffer(lineFeatures.data());
		m_lineObjectIndexBuffer.updateBuffer(lineObjectIndices.data());
	}
	if (!skeletons.empty()) {
		m_skeletonBuffer.generateBuffer(skeletons.size(), 3, GL_FLOAT);
		m_skeletonBuffer.updateBuffer(skeletons.data());
		m_colorSkeletonBuffer.generateBuffer(colorSkeletons.size(), 4, GL_FLOAT);
		m_colorSkeletonBuffer.updateBuffer(colorSkeletons.data());
		m_skeletonObjectIndexBuffer.generateBuffer(skeletonObjectIndices.size(), 1, GL_FLOAT);
		m_skeletonObjectIndexBuffer.updateBuffer(skeletonObjectIndices.data());
	}
	if (!links.empty()) {
		m_linksBuffer.generateBuffer(links.size(), 3, GL_FLOAT);
		m_linksBuffer.updateBuffer(links.data());
		m_colorLinksBuffer.generateBuffer(colorLinks.size(), 4, GL_FLOAT);
		m_colorLinksBuffer.updateBuffer(colorLinks.data());
		m_linkObjectIndexBuffer.generateBuffer(linkObjectIndices.size(), 1, GL_FLOAT);
		m_linkObjectIndexBuffer.updateBuffer(linkObjectIndices.data());
	}
	if (!ellipsoidTransforms.empty()) {
		m_ellipsoidTransformBuffer.generateBuffer(ellipsoidTransforms.size(), 4, GL_FLOAT);
		m_ellipsoidTransformBuffer.updateBuffer(ellipsoidTransforms.data());
		m_ellipsoidFeatureBuffer.generateBuffer(ellipsoidFeatures.size(), 1, GL_FLOAT);
		m_ellipsoidFeatureBuffer.updateBuffer(ellipsoidFeatures.data());
		m_ellipsoidObjectIndexBuffer.generateBuffer(ellipsoidObjectIndices.size(), 1, GL_FLOAT);
		m_ellipsoidObjectIndexBuffer.updateBuffer(ellipsoidObjectIndices.data());
	}
	return updateObjectModelBuffer();
}

void ObjectListMultiObjectDisplayCommand::display(poca::opengl::Camera* _cam, const bool _offscreen, const bool _ssao, poca::core::CommandExecutionResult& _result)
{
	if (!canBatch())
		return;
	if (m_triangleBuffer.empty() && m_pointBuffer.empty() && !rebuild())
		return;
	ObjectListDisplayCommand* referenceCommand = referenceDisplayCommand();
	if (referenceCommand == nullptr)
		return;

	drawElements(_cam, _ssao, referenceCommand);
	if (!_offscreen)
		drawPicking(_cam, referenceCommand);
	markComponentFamilyHandled(_result);
}

void ObjectListMultiObjectDisplayCommand::drawElements(poca::opengl::Camera* _cam, const bool _ssao, ObjectListDisplayCommand* _referenceCommand)
{
	const bool pointRendering = _referenceCommand->getParameter<bool>("pointRendering");
	const bool outlinePointRendering = _referenceCommand->getParameter<bool>("outlinePointRendering");
	const bool shapeRendering = _referenceCommand->getParameter<bool>("shapeRendering");
	const bool fill = _referenceCommand->getParameter<bool>("fill");
	const bool displayBboxSelection = _referenceCommand->getParameter<bool>("bboxSelection");
	const bool skeletonRendering = _referenceCommand->getParameter<bool>("skeletonRendering");
	const bool linkRendering = _referenceCommand->getParameter<bool>("linkRendering");
	const bool cullFaceActivated = _cam->cullFaceActivated();
	const std::string cullFaceType = _referenceCommand->getParameter<std::string>("cullFaceType");
	const float alpha = _referenceCommand->getParameter<float>("alpha");
	const bool translucentRendering = _referenceCommand->hasParameter("translucentRendering") ? _referenceCommand->getParameter<bool>("translucentRendering") : false;
	const bool useTranslucentMeshRendering = translucentRendering && m_is3D;

	if (useTranslucentMeshRendering) {
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);
		glDisable(GL_CULL_FACE);
	}
	else {
		if (alpha < 1.f) glDisable(GL_DEPTH_TEST);
		else glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
		if (cullFaceActivated) glEnable(GL_CULL_FACE);
		else glDisable(GL_CULL_FACE);
	}
	glDisable(GL_BLEND);
	glCullFace(GL_BACK);

	const glm::mat4& proj = _cam->getProjectionMatrix(), & view = _cam->getViewMatrix(), & cameraModel = _cam->getModelMatrix();
	auto bindObjectModels = [this](poca::opengl::Shader* _shader) {
		_shader->setInt("objectModels", 1);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_BUFFER, m_objectModelTextureID);
	};
	auto releaseObjectModels = []() {
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_BUFFER, 0);
		glActiveTexture(GL_TEXTURE0);
	};

	if (pointRendering && !m_pointBuffer.empty()) {
		const uint32_t pointSize = _referenceCommand->getParameter<uint32_t>("pointSizeGL");
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		poca::opengl::Shader* shader = _cam->getShader("multiObjectSphereRenderingShader");
		glEnable(GL_BLEND);
		shader->use();
		shader->setMat4("MVP", proj * view * cameraModel);
		shader->setMat4("view", view);
		shader->setMat4("model", cameraModel);
		shader->setMat4("projection", proj);
		shader->setInt("lutTexture", 0);
		shader->setFloat("minFeatureValue", m_minOriginalFeature);
		shader->setFloat("maxFeatureValue", m_maxOriginalFeature);
		shader->setFloat("currentMinFeatureValue", m_minOriginalFeature);
		shader->setFloat("currentMaxFeatureValue", m_maxOriginalFeature);
		shader->setBool("scaleLUT", false);
		shader->setBool("useSpecialColors", false);
		shader->setBool("screenRadius", false);
		shader->setVec2("pixelSize", 2.f / (float)_cam->width(), 2.f / (float)_cam->height());
		shader->setFloat("radius", pointSize);
		shader->setVec4v("clipPlanes", _cam->getClipPlanes());
		shader->setInt("nbClipPlanes", _cam->nbClippingPlanes());
		shader->setBool("clip", _cam->clip());
		shader->setFloat("nbPoints", m_pointBuffer.getNbElements());
		shader->setVec3("light_position", _cam->getEye());
		shader->setBool("applyUniformColor", false);
		glm::vec3 orientation = glm::normalize(_cam->getRotationSum() * glm::vec3(0.f, 0.f, 1.f));
		shader->setBool("activatedCulling", false);
		shader->setVec3("cameraForward", orientation);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_1D, m_textureLutID);
		bindObjectModels(shader);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(2);
		glEnableVertexAttribArray(4);
		m_pointBuffer.bindBuffer(0);
		m_locsFeatureBuffer.bindBuffer(2);
		m_pointObjectIndexBuffer.bindBuffer(4);
		glDrawArrays(m_pointBuffer.getMode(), 0, m_pointBuffer.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(2);
		glDisableVertexAttribArray(4);
		glBindTexture(GL_TEXTURE_1D, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		releaseObjectModels();
		shader->release();
	}

	if (outlinePointRendering && !m_outlinePointBuffer.empty()) {
		const uint32_t pointSize = _referenceCommand->getParameter<uint32_t>("pointSizeGL");
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		poca::opengl::Shader* shader = _cam->getShader("multiObjectSphereRenderingShader");
		glEnable(GL_BLEND);
		shader->use();
		shader->setMat4("MVP", proj * view * cameraModel);
		shader->setMat4("view", view);
		shader->setMat4("model", cameraModel);
		shader->setMat4("projection", proj);
		shader->setInt("lutTexture", 0);
		shader->setFloat("minFeatureValue", m_minOriginalFeature);
		shader->setFloat("maxFeatureValue", m_maxOriginalFeature);
		shader->setFloat("currentMinFeatureValue", m_minOriginalFeature);
		shader->setFloat("currentMaxFeatureValue", m_maxOriginalFeature);
		shader->setBool("scaleLUT", false);
		shader->setBool("useSpecialColors", false);
		shader->setBool("screenRadius", false);
		shader->setVec2("pixelSize", 2.f / (float)_cam->width(), 2.f / (float)_cam->height());
		shader->setFloat("radius", pointSize);
		shader->setVec4v("clipPlanes", _cam->getClipPlanes());
		shader->setInt("nbClipPlanes", _cam->nbClippingPlanes());
		shader->setBool("clip", _cam->clip());
		shader->setFloat("nbPoints", m_outlinePointBuffer.getNbElements());
		shader->setVec3("light_position", _cam->getEye());
		shader->setBool("applyUniformColor", false);
		glm::vec3 orientation = glm::normalize(_cam->getRotationSum() * glm::vec3(0.f, 0.f, 1.f));
		shader->setBool("activatedCulling", false);
		shader->setVec3("cameraForward", orientation);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_1D, m_textureLutID);
		bindObjectModels(shader);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(2);
		glEnableVertexAttribArray(4);
		m_outlinePointBuffer.bindBuffer(0);
		m_outlineLocsFeatureBuffer.bindBuffer(2);
		m_outlinePointObjectIndexBuffer.bindBuffer(4);
		glDrawArrays(m_outlinePointBuffer.getMode(), 0, m_outlinePointBuffer.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(2);
		glDisableVertexAttribArray(4);
		glBindTexture(GL_TEXTURE_1D, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		releaseObjectModels();
		shader->release();
	}

	if (cullFaceType == "front") glCullFace(GL_FRONT);
	else glCullFace(GL_BACK);

	if (skeletonRendering && !m_skeletonBuffer.empty())
	{
		poca::opengl::Shader* shader = _cam->getShader("multiObjectSimpleShader");
		shader->use();
		shader->setMat4("MVP", proj * view * cameraModel);
		shader->setMat4("model", cameraModel);
		shader->setFloat("alpha", 1.f);
		shader->setBool("useSpecialColors", true);
		shader->setVec4v("clipPlanes", _cam->getClipPlanes());
		shader->setInt("nbClipPlanes", _cam->nbClippingPlanes());
		shader->setBool("clip", _cam->clip());
		shader->setBool("activatedCulling", false);
		shader->setVec3("cameraForward", glm::normalize(_cam->getRotationSum() * glm::vec3(0.f, 0.f, 1.f)));
		bindObjectModels(shader);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(3);
		glEnableVertexAttribArray(5);
		m_skeletonBuffer.bindBuffer(0);
		m_colorSkeletonBuffer.bindBuffer(3);
		m_skeletonObjectIndexBuffer.bindBuffer(5);
		glDrawArrays(m_skeletonBuffer.getMode(), 0, m_skeletonBuffer.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(3);
		glDisableVertexAttribArray(5);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		releaseObjectModels();
		shader->release();
	}

	if (linkRendering && !m_linksBuffer.empty())
	{
		poca::opengl::Shader* shader = _cam->getShader("multiObjectSimpleShader");
		shader->use();
		shader->setMat4("MVP", proj * view * cameraModel);
		shader->setMat4("model", cameraModel);
		shader->setFloat("alpha", 1.f);
		shader->setBool("useSpecialColors", true);
		shader->setVec4v("clipPlanes", _cam->getClipPlanes());
		shader->setInt("nbClipPlanes", _cam->nbClippingPlanes());
		shader->setBool("clip", _cam->clip());
		shader->setBool("activatedCulling", false);
		shader->setVec3("cameraForward", glm::normalize(_cam->getRotationSum() * glm::vec3(0.f, 0.f, 1.f)));
		bindObjectModels(shader);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(3);
		glEnableVertexAttribArray(5);
		m_linksBuffer.bindBuffer(0);
		m_colorLinksBuffer.bindBuffer(3);
		m_linkObjectIndexBuffer.bindBuffer(5);
		glDrawArrays(m_linksBuffer.getMode(), 0, m_linksBuffer.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(3);
		glDisableVertexAttribArray(5);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		releaseObjectModels();
		shader->release();
	}

	glPolygonMode(GL_FRONT_AND_BACK, fill ? GL_FILL : GL_LINE);
	glEnable(GL_BLEND);
	if (shapeRendering && !m_triangleBuffer.empty()) {
		if (m_is3D) {
			if (useTranslucentMeshRendering)
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			poca::opengl::Shader* shader = useTranslucentMeshRendering ? _cam->getShader("multiObjectObjectRenderingTranslucentShader") : (_ssao ? _cam->getShader("multiObjectObjectRenderingSSAOShader") : _cam->getShader("multiObjectObjectRenderingShader"));
			glm::vec3 orientation = _cam->getRotationSum() * glm::vec3(0.f, 0.f, 1.f);
			glm::vec3 pos(orientation + _cam->getCenter());
			pos *= 2 * _cam->getOriginalDistanceOrtho();
			shader->use();
			shader->setMat4("model", cameraModel);
			shader->setMat4("view", view);
			shader->setMat4("projection", proj);
			shader->setInt("lutTexture", 0);
			shader->setFloat("minFeatureValue", m_minOriginalFeature);
			shader->setFloat("maxFeatureValue", m_maxOriginalFeature);
			shader->setVec4v("clipPlanes", _cam->getClipPlanes());
			shader->setInt("nbClipPlanes", _cam->nbClippingPlanes());
			shader->setBool("clip", _cam->clip());
			shader->setVec3("lightColor", 1.0f, 1.0f, 1.0f);
			shader->setVec3("lightPos", pos);
			shader->setVec3("viewPos", pos);
			shader->setBool("applyIllumination", fill ? true : false);
			shader->setVec3("light_position", _cam->getEye());
			shader->setFloat("alpha", alpha);
			shader->setBool("applyUniformColor", false);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_1D, m_textureLutID);
			bindObjectModels(shader);
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);
			glEnableVertexAttribArray(2);
			glEnableVertexAttribArray(3);
			m_triangleBuffer.bindBuffer(0);
			m_triangleNormalBuffer.bindBuffer(1);
			m_triangleFeatureBuffer.bindBuffer(2);
			m_triangleObjectIndexBuffer.bindBuffer(3);
			glDrawArrays(m_triangleBuffer.getMode(), 0, m_triangleBuffer.getNbElements());
			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);
			glDisableVertexAttribArray(2);
			glDisableVertexAttribArray(3);
			glBindTexture(GL_TEXTURE_1D, 0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			releaseObjectModels();
			shader->release();
		}
		else if (fill) {
			poca::opengl::Shader* shader = _cam->getShader("multiObjectSimpleShader");
			shader->use();
			shader->setMat4("MVP", proj * view * cameraModel);
			shader->setMat4("model", cameraModel);
			shader->setFloat("alpha", alpha);
			shader->setBool("useSpecialColors", false);
			shader->setInt("lutTexture", 0);
			shader->setFloat("minFeatureValue", m_minOriginalFeature);
			shader->setFloat("maxFeatureValue", m_maxOriginalFeature);
			shader->setBool("clip", _cam->clip());
			shader->setVec4v("clipPlanes", _cam->getClipPlanes());
			shader->setInt("nbClipPlanes", _cam->nbClippingPlanes());
			shader->setBool("activatedCulling", false);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_1D, m_textureLutID);
			bindObjectModels(shader);
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(2);
			glEnableVertexAttribArray(5);
			m_triangleBuffer.bindBuffer(0);
			m_triangleFeatureBuffer.bindBuffer(2);
			m_triangleObjectIndexBuffer.bindBuffer(5);
			glDrawArrays(m_triangleBuffer.getMode(), 0, m_triangleBuffer.getNbElements());
			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(2);
			glDisableVertexAttribArray(5);
			glBindTexture(GL_TEXTURE_1D, 0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			releaseObjectModels();
			shader->release();
		}
		else if (!m_lineBuffer.empty()) {
			glDisable(GL_DEPTH_TEST);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			poca::opengl::Shader* shader = _cam->getShader("multiObjectLineRenderingShader");
			shader->use();
			shader->setMat4("MVP", proj * view * cameraModel);
			shader->setMat4("model", cameraModel);
			const glm::uvec4 viewport = _cam->getViewport();
			shader->setVec4("viewport", (float)viewport.x, (float)viewport.y, (float)viewport.z, (float)viewport.w);
			shader->setVec2("resolution", (float)_cam->width(), (float)_cam->height());
			shader->setFloat("thickness", 1.f);
			shader->setFloat("antialias", 1.f);
			shader->setBool("useSingleColor", false);
			shader->setInt("lutTexture", 0);
			shader->setFloat("minFeatureValue", m_minOriginalFeature);
			shader->setFloat("maxFeatureValue", m_maxOriginalFeature);
			shader->setBool("clip", _cam->clip());
			shader->setVec4v("clipPlanes", _cam->getClipPlanes());
			shader->setInt("nbClipPlanes", _cam->nbClippingPlanes());
			shader->setBool("activatedCulling", false);
			shader->setVec3("cameraForward", glm::normalize(_cam->getRotationSum() * glm::vec3(0.f, 0.f, 1.f)));
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_1D, m_textureLutID);
			bindObjectModels(shader);
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(2);
			glEnableVertexAttribArray(3);
			m_lineBuffer.bindBuffer(0);
			m_lineFeatureBuffer.bindBuffer(2);
			m_lineObjectIndexBuffer.bindBuffer(3);
			glDrawArrays(m_lineBuffer.getMode(), 0, m_lineBuffer.getNbElements());
			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(2);
			glDisableVertexAttribArray(3);
			glBindTexture(GL_TEXTURE_1D, 0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			releaseObjectModels();
			shader->release();
		}
	}

	glDisable(GL_BLEND);
	glDepthMask(GL_TRUE);
	glDisable(GL_DEPTH_TEST);
	if (m_idSelection >= 0 && displayBboxSelection && !m_boundingBoxSelection.empty()) {
		GLfloat bkColor[4];
		glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
		const poca::core::Color4D color = poca::core::contrastColor(
			poca::core::Color4D(bkColor[0] * 255.f, bkColor[1] * 255.f, bkColor[2] * 255.f, bkColor[3] * 255.f));
		_cam->drawUniformShader<poca::core::Vec3mf>(m_boundingBoxSelection, color);
	}

	const bool ellipsoidRendering = _referenceCommand->getParameter<bool>("ellipsoidRendering");
	if (ellipsoidRendering && !m_ellipsoidTransformBuffer.empty()) {
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);
		glDisable(GL_CULL_FACE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		poca::opengl::Shader* shader = _cam->getShader("multiObject3DInstanceRenderingShader");
		shader->use();
		shader->setMat4("MVP", proj * view * cameraModel);
		shader->setMat4("model", cameraModel);
		shader->setBool("useSingleColor", false);
		shader->setInt("lutTexture", 0);
		shader->setFloat("minFeatureValue", m_minOriginalFeature);
		shader->setFloat("maxFeatureValue", m_maxOriginalFeature);
		shader->setVec4v("clipPlanes", _cam->getClipPlanes());
		shader->setInt("nbClipPlanes", _cam->nbClippingPlanes());
		shader->setBool("clip", _cam->clip());

		poca::opengl::HelperSingleton* helper = poca::opengl::HelperSingleton::instance();
		poca::opengl::QuadSingleGLBuffer<float>& ellipsoidBuffer = helper->getEllipsoidBuffer();
		poca::opengl::QuadSingleGLBuffer<GLushort>& ellipsoidIndicesBuffer = helper->getEllipsoidIndicesBuffer();
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_1D, m_textureLutID);
		bindObjectModels(shader);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(9);
		glEnableVertexAttribArray(10);
		ellipsoidBuffer.bindBuffer(0);
		ellipsoidIndicesBuffer.bindBuffer(1);
		m_ellipsoidFeatureBuffer.bindBuffer(9);
		glVertexAttribDivisor(9, 1);
		m_ellipsoidObjectIndexBuffer.bindBuffer(10);
		glVertexAttribDivisor(10, 1);
		for (int i = 0; i < 4; i++) {
			glEnableVertexAttribArray(5 + i);
			m_ellipsoidTransformBuffer.bindBuffer(5 + i, (void*)(4 * sizeof(float) * i));
		}
		glDrawElementsInstanced(GL_QUADS, helper->getNbIndicesUnitSphere(), GL_UNSIGNED_SHORT, 0, (GLsizei)m_ellipsoidTransformBuffer.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(9);
		glDisableVertexAttribArray(10);
		for (int i = 0; i < 4; i++)
			glDisableVertexAttribArray(5 + i);
		glBindTexture(GL_TEXTURE_1D, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		releaseObjectModels();
		shader->release();
	}
}

void ObjectListMultiObjectDisplayCommand::drawPicking(poca::opengl::Camera* _cam, ObjectListDisplayCommand* _referenceCommand)
{
	if (m_pickFBO == nullptr)
		updatePickingFBO(_cam->getWidth(), _cam->getHeight());
	if (m_pickFBO == nullptr)
		return;

	const bool shapeRendering = _referenceCommand->getParameter<bool>("shapeRendering");
	glEnable(GL_DEPTH_TEST);
	GLfloat bkColor[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	const bool success = m_pickFBO->bind();
	if (!success) std::cout << "Problem with binding" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	poca::opengl::Shader* shader = _cam->getShader("multiObjectPickShader");
	shader->use();
	shader->setMat4("MVP", _cam->getProjectionMatrix() * _cam->getViewMatrix() * _cam->getModelMatrix());
	shader->setMat4("model", _cam->getModelMatrix());
	shader->setBool("hasFeature", true);
	shader->setFloat("minFeatureValue", m_minOriginalFeature);
	shader->setVec4v("clipPlanes", _cam->getClipPlanes());
	shader->setInt("nbClipPlanes", _cam->nbClippingPlanes());
	shader->setBool("clip", _cam->clip());
	shader->setInt("objectModels", 1);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_BUFFER, m_objectModelTextureID);
	if (shapeRendering && !m_triangleBuffer.empty()) {
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glEnableVertexAttribArray(3);
		m_triangleBuffer.bindBuffer(0);
		m_idBuffer.bindBuffer(1);
		m_triangleFeatureBuffer.bindBuffer(2);
		m_triangleObjectIndexBuffer.bindBuffer(3);
		glDrawArrays(m_triangleBuffer.getMode(), 0, m_triangleBuffer.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);
		glDisableVertexAttribArray(3);
	}
	else if (!m_pointBuffer.empty()) {
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glEnableVertexAttribArray(3);
		m_pointBuffer.bindBuffer(0);
		m_idLocsBuffer.bindBuffer(1);
		m_locsFeatureBuffer.bindBuffer(2);
		m_pointObjectIndexBuffer.bindBuffer(3);
		glDrawArrays(m_pointBuffer.getMode(), 0, m_pointBuffer.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);
		glDisableVertexAttribArray(3);
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindTexture(GL_TEXTURE_BUFFER, 0);
	glActiveTexture(GL_TEXTURE0);
	shader->release();
	m_pickFBO->release();
	glBindFramebuffer(GL_FRAMEBUFFER, _cam->defaultFramebufferObject());
	glClearColor(bkColor[0], bkColor[1], bkColor[2], bkColor[3]);
}

QString ObjectListMultiObjectDisplayCommand::getInfosTriangle(const int _id) const
{
	if (_id < 0 || (size_t)_id >= m_object->nbColors())
		return QString();
	return globalObjectInfo(m_object, (size_t)_id);
}

void ObjectListMultiObjectDisplayCommand::generateBoundingBoxSelection(const int _idx)
{
	if (_idx < 0 || (size_t)_idx >= m_object->nbColors())
		return;
	poca::core::MyObjectInterface* child = m_object->getObject((size_t)_idx);
	if (child == nullptr)
		return;
	std::vector<poca::core::Vec3mf> cube(24);
	poca::geometry::createCubeFromVector(cube, child->boundingBox());
	m_boundingBoxSelection.updateBuffer(cube.data());
}

void ObjectListMultiObjectDisplayCommand::freeGPUMemory()
{
	m_pointBuffer.freeGPUMemory();
	m_outlinePointBuffer.freeGPUMemory();
	m_idLocsBuffer.freeGPUMemory();
	m_locsFeatureBuffer.freeGPUMemory();
	m_outlineLocsFeatureBuffer.freeGPUMemory();
	m_pointObjectIndexBuffer.freeGPUMemory();
	m_outlinePointObjectIndexBuffer.freeGPUMemory();
	m_triangleBuffer.freeGPUMemory();
	m_triangleNormalBuffer.freeGPUMemory();
	m_idBuffer.freeGPUMemory();
	m_triangleFeatureBuffer.freeGPUMemory();
	m_triangleObjectIndexBuffer.freeGPUMemory();
	m_lineBuffer.freeGPUMemory();
	m_lineFeatureBuffer.freeGPUMemory();
	m_lineObjectIndexBuffer.freeGPUMemory();
	m_skeletonBuffer.freeGPUMemory();
	m_linksBuffer.freeGPUMemory();
	m_colorSkeletonBuffer.freeGPUMemory();
	m_colorLinksBuffer.freeGPUMemory();
	m_skeletonObjectIndexBuffer.freeGPUMemory();
	m_linkObjectIndexBuffer.freeGPUMemory();
	m_boundingBoxSelection.freeGPUMemory();
	m_ellipsoidTransformBuffer.freeGPUMemory();
	m_ellipsoidFeatureBuffer.freeGPUMemory();
	m_ellipsoidObjectIndexBuffer.freeGPUMemory();
	m_objectModelBuffer.freeGPUMemory();
	if (m_objectModelTextureID != 0) {
		glDeleteTextures(1, &m_objectModelTextureID);
		m_objectModelTextureID = 0;
	}
	m_textureLutID = 0;
}

bool ObjectListMultiObjectDisplayCommand::updateObjectModelBuffer()
{
	if (m_object == nullptr || m_object->nbColors() == 0)
		return false;

	std::vector<glm::vec4> matrices;
	matrices.reserve(m_object->nbColors() * 4);
	const glm::mat4 parentInvModel = glm::inverse(m_object->getModelMatrix());
	for (size_t n = 0; n < m_object->nbColors(); n++) {
		poca::core::MyObjectInterface* child = m_object->getObject(n);
		const glm::mat4 model = child == nullptr ? glm::mat4(1.f) : parentInvModel * child->getModelMatrix();
		matrices.push_back(model[0]);
		matrices.push_back(model[1]);
		matrices.push_back(model[2]);
		matrices.push_back(model[3]);
	}

	if (m_objectModelBuffer.empty())
		m_objectModelBuffer.generateBuffer(matrices.size(), 4, GL_FLOAT, GL_TEXTURE_BUFFER);
	m_objectModelBuffer.updateBuffer(matrices);
	if (m_objectModelTextureID == 0)
		glGenTextures(1, &m_objectModelTextureID);
	glBindTexture(GL_TEXTURE_BUFFER, m_objectModelTextureID);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, m_objectModelBuffer.getBufferVertex());
	glBindTexture(GL_TEXTURE_BUFFER, 0);
	return true;
}

bool ObjectListMultiObjectDisplayCommand::refreshTransformBuffers()
{
	if (!updateObjectModelBuffer())
		return false;
	if (m_idSelection >= 0)
		generateBoundingBoxSelection(m_idSelection);
	return true;
}

bool ObjectListMultiObjectDisplayCommand::updateFeatureBuffers()
{
	if (!canBatch() || (m_locsFeatureBuffer.empty() && m_triangleFeatureBuffer.empty() && m_lineFeatureBuffer.empty() && m_ellipsoidFeatureBuffer.empty()))
		return false;

	std::vector<float> pointFeatures, outlinePointFeatures, triangleFeatures, lineFeatures, ellipsoidFeatures;
	m_minOriginalFeature = std::numeric_limits<float>::max();
	m_maxOriginalFeature = std::numeric_limits<float>::lowest();

	forEachObjectList(m_object, [&](size_t, poca::core::MyObjectInterface*, poca::geometry::ObjectListInterface* objs) {
		poca::core::HistogramInterface* histInterface = objs->getCurrentHistogram();
		poca::core::Histogram<float>* histogram = dynamic_cast<poca::core::Histogram<float>*>(histInterface);
		if (histogram == nullptr)
			return;

		m_minOriginalFeature = std::min(m_minOriginalFeature, histInterface->getMin());
		m_maxOriginalFeature = std::max(m_maxOriginalFeature, histInterface->getMax());
		const std::vector<float>& values = histogram->getValues();
		const std::vector<bool>& selection = objs->getSelection();

		std::vector<float> localPointFeatures;
		if (objs->isHiLow()) {
			float inter = histInterface->getMax() - histInterface->getMin();
			float selectedValue = histInterface->getMin() + inter / 4.f;
			float notSelectedValue = histInterface->getMin() + inter * (3.f / 4.f);
			objs->getLocsFeatureInSelectionHiLow(localPointFeatures, selection, selectedValue, notSelectedValue);
		}
		else {
			objs->getLocsFeatureInSelection(localPointFeatures, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		}
		if (!objs->isSelected())
			std::fill(localPointFeatures.begin(), localPointFeatures.end(), poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		pointFeatures.insert(pointFeatures.end(), localPointFeatures.begin(), localPointFeatures.end());

		std::vector<poca::core::Vec3mf> localOutlinePoints;
		objs->generateOutlineLocs(localOutlinePoints);
		if (!localOutlinePoints.empty()) {
			std::vector<float> localOutlinePointFeatures;
			if (objs->isHiLow()) {
				float inter = histInterface->getMax() - histInterface->getMin();
				float selectedValue = histInterface->getMin() + inter / 4.f;
				float notSelectedValue = histInterface->getMin() + inter * (3.f / 4.f);
				objs->getOutlineLocsFeatureInSelectionHiLow(localOutlinePointFeatures, selection, selectedValue, notSelectedValue);
			}
			else {
				objs->getOutlineLocsFeatureInSelection(localOutlinePointFeatures, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
			}
			if (!objs->isSelected())
				std::fill(localOutlinePointFeatures.begin(), localOutlinePointFeatures.end(), poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
			outlinePointFeatures.insert(outlinePointFeatures.end(), localOutlinePointFeatures.begin(), localOutlinePointFeatures.end());
		}

		std::vector<float> localTriangleFeatures;
		if (objs->isHiLow()) {
			float inter = histInterface->getMax() - histInterface->getMin();
			float selectedValue = histInterface->getMin() + inter / 4.f;
			float notSelectedValue = histInterface->getMin() + inter * (3.f / 4.f);
			objs->getFeatureInSelectionHiLow(localTriangleFeatures, selection, selectedValue, notSelectedValue);
		}
		else {
			objs->getFeatureInSelection(localTriangleFeatures, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		}
		if (!objs->isSelected())
			std::fill(localTriangleFeatures.begin(), localTriangleFeatures.end(), poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		triangleFeatures.insert(triangleFeatures.end(), localTriangleFeatures.begin(), localTriangleFeatures.end());

		if (objs->dimension() == 2) {
			std::vector<float> localLineFeatures;
			if (objs->isHiLow()) {
				float inter = histInterface->getMax() - histInterface->getMin();
				float selectedValue = histInterface->getMin() + inter / 4.f;
				float notSelectedValue = histInterface->getMin() + inter * (3.f / 4.f);
				objs->getOutlinesFeatureInSelectionHiLow(localLineFeatures, selection, selectedValue, notSelectedValue);
			}
			else {
				objs->getOutlinesFeatureInSelection(localLineFeatures, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
			}
			if (!objs->isSelected())
				std::fill(localLineFeatures.begin(), localLineFeatures.end(), poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
			lineFeatures.insert(lineFeatures.end(), localLineFeatures.begin(), localLineFeatures.end());
		}

		if (objs->dimension() == 3 && objs->hasData("major") && objs->hasData("minor") && objs->hasData("minor2")) {
			std::vector<float> localEllipsoidFeatures(objs->nbElements());
			if (objs->isHiLow()) {
				float inter = histInterface->getMax() - histInterface->getMin();
				float selectedValue = histInterface->getMin() + inter / 4.f;
				float notSelectedValue = histInterface->getMin() + inter * (3.f / 4.f);
				for (size_t idx = 0; idx < objs->nbElements(); idx++)
					localEllipsoidFeatures[idx] = selection[idx] ? selectedValue : notSelectedValue;
			}
			else {
				for (size_t idx = 0; idx < objs->nbElements(); idx++)
					localEllipsoidFeatures[idx] = selection[idx] ? values[idx] : poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER;
			}
			if (!objs->isSelected())
				std::fill(localEllipsoidFeatures.begin(), localEllipsoidFeatures.end(), poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
			ellipsoidFeatures.insert(ellipsoidFeatures.end(), localEllipsoidFeatures.begin(), localEllipsoidFeatures.end());
		}
	});

	if (m_minOriginalFeature == std::numeric_limits<float>::max()) {
		m_minOriginalFeature = 0.f;
		m_maxOriginalFeature = 1.f;
	}
	m_actualValueFeature = m_maxOriginalFeature;
	if (!m_locsFeatureBuffer.empty())
		m_locsFeatureBuffer.updateBuffer(pointFeatures);
	if (!m_outlineLocsFeatureBuffer.empty())
		m_outlineLocsFeatureBuffer.updateBuffer(outlinePointFeatures);
	if (!m_triangleFeatureBuffer.empty())
		m_triangleFeatureBuffer.updateBuffer(triangleFeatures);
	if (!m_lineFeatureBuffer.empty())
		m_lineFeatureBuffer.updateBuffer(lineFeatures);
	if (!m_ellipsoidFeatureBuffer.empty())
		m_ellipsoidFeatureBuffer.updateBuffer(ellipsoidFeatures);
	return true;
}
