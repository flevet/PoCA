/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListDisplayCommand.cpp
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
#include <glm/gtx/string_cast.hpp>

#include <General/Engine.hpp>
#include <General/Palette.hpp>
#include <General/Histogram.hpp>
#include <General/MyData.hpp>
#include <General/Misc.h>
#include <Interfaces/HistogramInterface.hpp>
#include <Interfaces/CameraInterface.hpp>
#include <OpenGL/Shader.hpp>
#include <Interfaces/ObjectFeaturesFactoryInterface.hpp>
#include <OpenGL/Helper.h>
#include <General/Engine.hpp>
#include <Geometry/DetectionSet.hpp>
#include <Objects/MyObject.hpp>
#include <OpenGL/Helper.h>
#include <Geometry/ObjectListMesh.hpp>
#include <Cuda/Misc.h>

#include "ObjectListDisplayCommand.hpp"

char* vsLight = "#version 330 core\n"
"layout(location = 0) in vec3 aPos;\n"
"layout(location = 1) in vec3 aNormal;"
"layout(location = 2) in float vertexFeature;\n"
"uniform mat4 model;\n"
"uniform mat4 view;\n"
"uniform mat4 projection;\n"
"const int MAX_CLIPPING_PLANES = 50;\n"
"uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];\n"
"uniform int nbClipPlanes;\n"
"out float feature;\n"
"out vec3 FragPos;\n"
"out vec3 Normal;\n"
"out float vclipDistance;\n"
"void main() {\n"
"	vec4 pos = vec4(aPos, 1.0);\n"
"	FragPos = vec3(model * pos);\n"
"	Normal = mat3(transpose(inverse(model))) * aNormal;\n"
"	gl_Position = projection * view * vec4(FragPos, 1.0);\n"
"	feature = vertexFeature;\n"
"	vclipDistance = 3.402823466e+38;\n"
"	for(int n = 0; n < nbClipPlanes; n++){\n"
"		float d = dot(pos, clipPlanes[n]);\n"
"		vclipDistance = d < vclipDistance ? d : vclipDistance;\n"
"	}\n"
"}";

char* fsLight = "#version 330 core\n"
"in float feature;\n"
"in vec3 Normal;\n"
"in vec3 FragPos;\n"
"in float vclipDistance;\n"
"out vec4 color;\n"
"uniform mat4 view;\n"
"uniform sampler1D lutTexture;\n"
"uniform float minFeatureValue;\n"
"uniform float maxFeatureValue;\n"
"uniform vec3 lightPos;\n"
"uniform vec3 viewPos;\n"
"uniform vec3 lightColor;\n"
"uniform bool applyIllumination;\n"
"uniform bool clip;\n"
"void main() {\n"
"	if (clip && vclipDistance < 0.f)\n"
"		discard;\n"
"	if (feature < minFeatureValue)\n"
"		discard;\n"
"	float inter = maxFeatureValue - minFeatureValue;\n"
"	vec3 objectColor = texture(lutTexture, ((feature - minFeatureValue) / inter)).xyz;\n"
"	// ambient\n"
"	float ambientStrength = 0.1;\n"
"	vec3 ambient = ambientStrength * lightColor;\n"
"	// diffuse \n"
"	vec3 norm = normalize(Normal);\n"
"	vec3 lightDir = normalize(lightPos - FragPos);\n"
"	float diff = max(dot(norm, lightDir), 0.0);\n"
"	vec3 diffuse = diff * lightColor;\n"
"	// specular\n"
"	float specularStrength = 0.5;\n"
"	vec3 viewDir = normalize(viewPos - FragPos);\n"
"	vec3 reflectDir = reflect(-lightDir, norm);\n"
"	float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);\n"
"	vec3 specular = specularStrength * spec * lightColor;\n"
"	vec3 result;\n"
"	if (applyIllumination)\n"
"		result = (ambient + diffuse + specular) * objectColor;\n"
"	else\n"
"		result = objectColor;\n"
"	color = vec4(result, 0.4);\n"
"}";

char* fsLightSSAO = "#version 330 core\n"
"layout (location = 0) out vec3 gPosition;\n"
"layout(location = 1) out vec3 gNormal;\n"
"layout(location = 2) out vec3 gAlbedo;\n"
"in float feature;\n"
"in vec3 Normal;\n"
"in vec3 FragPos;\n"
"in float vclipDistance;\n"
"uniform mat4 view;\n"
"uniform sampler1D lutTexture;\n"
"uniform float minFeatureValue;\n"
"uniform float maxFeatureValue;\n"
"uniform vec3 lightPos;\n"
"uniform vec3 viewPos;\n"
"uniform vec3 lightColor;\n"
"uniform bool applyIllumination;\n"
"uniform bool clip;\n"
"void main() {\n"
"	if (clip && vclipDistance < 0.f)\n"
"		discard;\n"
"	if (feature < minFeatureValue)\n"
"		discard;\n"
"	float inter = maxFeatureValue - minFeatureValue;\n"
"	vec3 objectColor = texture(lutTexture, ((feature - minFeatureValue) / inter)).xyz;\n"
"	// ambient\n"
"	float ambientStrength = 0.1;\n"
"	vec3 ambient = ambientStrength * lightColor;\n"
"	// diffuse \n"
"	vec3 norm = normalize(Normal);\n"
"	vec3 lightDir = normalize(lightPos - FragPos);\n"
"	float diff = max(dot(norm, lightDir), 0.0);\n"
"	vec3 diffuse = diff * lightColor;\n"
"	// specular\n"
"	float specularStrength = 0.5;\n"
"	vec3 viewDir = normalize(viewPos - FragPos);\n"
"	vec3 reflectDir = reflect(-lightDir, norm);\n"
"	float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);\n"
"	vec3 specular = specularStrength * spec * lightColor;\n"
"	vec3 result;\n"
"	if (applyIllumination)\n"
"		result = (ambient + diffuse + specular) * objectColor;\n"
"	else\n"
"		result = objectColor;\n"
"	gAlbedo = result;\n"
"	gPosition = (view * vec4(FragPos, 1)).xyz;\n"
"	gNormal = (view * vec4(Normal, 1)).xyz;\n"
"}";


ObjectListDisplayCommand::ObjectListDisplayCommand(poca::geometry::ObjectListInterface* _objs) : poca::opengl::BasicDisplayCommand(_objs, "ObjectListDisplayCommand"), m_fill(true), m_textureLutID(0)
{
	m_objects = _objs;
	m_pickOneObject = NULL;

	
	const nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
	addCommandInfo(poca::core::CommandInfo(false, "fill", true));
	addCommandInfo(poca::core::CommandInfo(false, "pointRendering", true));
	addCommandInfo(poca::core::CommandInfo(false, "outlinePointRendering", true));
	addCommandInfo(poca::core::CommandInfo(false, "shapeRendering", true));
	addCommandInfo(poca::core::CommandInfo(false, "bboxSelection", true));
	addCommandInfo(poca::core::CommandInfo(false, "pointSizeGL", 6u));
	addCommandInfo(poca::core::CommandInfo(false, "ellipsoidRendering", true));
	addCommandInfo(poca::core::CommandInfo(false, "cullFaceType", std::string("front")));
	addCommandInfo(poca::core::CommandInfo(false, "skeletonRendering", true));
	addCommandInfo(poca::core::CommandInfo(false, "linkRendering", true));
	if (parameters.contains(name())) {
		nlohmann::json param = parameters[name()];
		if(param.contains("fill"))
			loadParameters(poca::core::CommandInfo(false, "fill", param["fill"].get<bool>()));
		if (param.contains("pointRendering"))
			loadParameters(poca::core::CommandInfo(false, "pointRendering", param["pointRendering"].get<bool>()));
		if (param.contains("outlinePointRendering"))
			loadParameters(poca::core::CommandInfo(false, "outlinePointRendering", param["outlinePointRendering"].get<bool>()));
		if (param.contains("shapeRendering"))
			loadParameters(poca::core::CommandInfo(false, "shapeRendering", param["shapeRendering"].get<bool>()));
		if (param.contains("bboxSelection"))
			loadParameters(poca::core::CommandInfo(false, "bboxSelection", param["bboxSelection"].get<bool>()));
		if (param.contains("pointSizeGL"))
			loadParameters(poca::core::CommandInfo(false, "pointSizeGL", param["pointSizeGL"].get<uint32_t>()));
		if (param.contains("ellipsoidRendering"))
			loadParameters(poca::core::CommandInfo(false, "ellipsoidRendering", param["ellipsoidRendering"].get<bool>()));
		if (param.contains("cullFaceType"))
			loadParameters(poca::core::CommandInfo(false, "cullFaceType", param["cullFaceType"].get <std::string>()));
		if (param.contains("skeletonRendering"))
			loadParameters(poca::core::CommandInfo(false, "centroidRendering", param["centroidRendering"].get<bool>()));
		if (param.contains("linkRendering"))
			loadParameters(poca::core::CommandInfo(false, "shapeRendering", param["shapeRendering"].get<bool>()));
	}
}

ObjectListDisplayCommand::ObjectListDisplayCommand(const ObjectListDisplayCommand& _o) : poca::opengl::BasicDisplayCommand(_o), m_fill(_o.m_fill)
{
	m_objects = _o.m_objects;
}

ObjectListDisplayCommand::~ObjectListDisplayCommand()
{
	freeGPUMemory();
}

void ObjectListDisplayCommand::execute(poca::core::CommandInfo* _infos)
{
	poca::opengl::BasicDisplayCommand::execute(_infos);
	if (_infos->nameCommand == "histogram" || _infos->nameCommand == "updateFeature") {
		generateFeatureBuffer();
	}
	else if (_infos->nameCommand == "display") {
		poca::opengl::Camera* cam = _infos->getParameterPtr<poca::opengl::Camera>("camera");
		bool offscrean = false, ssao = false;;
		if (_infos->hasParameter("offscreen"))
			offscrean = _infos->getParameter<bool>("offscreen");
		if (_infos->hasParameter("ssao"))
			ssao = _infos->getParameter<bool>("ssao");
		display(cam, offscrean, ssao);
	}
	else if (_infos->nameCommand == "changeLUT") {
		m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(m_objects->getPalette());
		if(_infos->hasParameter("regenerateFeatureBuffer"))
			generateFeatureBuffer();
	}
	else if (_infos->nameCommand == "pick") {
		if (!m_objects->isSelected()) return;
		QString infos = getInfosTriangle(m_idSelection);
		if (infos.isEmpty()) return;
		poca::core::stringList listInfos = _infos->getParameter<poca::core::stringList>("infos");
		listInfos.push_back(infos.toLatin1().data());
		_infos->addParameter("infos", listInfos);
		if (m_idSelection >= 0) {
			generateBoundingBoxSelection(m_idSelection);
			if (_infos->hasParameter("pickedPoints")) {
				std::vector <poca::core::Vec3mf> pickedPoints = _infos->getParameter< std::vector <poca::core::Vec3mf>>("pickedPoints");
					pickedPoints.push_back(m_objects->computeBarycenterElement(m_idSelection));
					_infos->addParameter("pickedPoints", pickedPoints);
			}
		}
	}
	else if (_infos->nameCommand == "doubleClickCamera") {
		size_t idSelection = m_idSelection;
		if (_infos->hasParameter("objectID"))
			idSelection = _infos->getParameter<size_t>("objectID");
		if (idSelection >= 0 && idSelection < m_objects->nbElements()) {
			poca::core::BoundingBox bbox = m_objects->computeBoundingBoxElement(idSelection);
			poca::opengl::Camera* cam = _infos->getParameterPtr<poca::opengl::Camera>("camera");
			displayZoomToBBox(cam, bbox);
			if (_infos->hasParameter("bbox")) {
				poca::core::BoundingBox bbox2 = _infos->getParameter<poca::core::BoundingBox>("bbox");
				for (size_t n = 0; n < 3; n++)
					bbox[n] = bbox[n] < bbox2[n] ? bbox[n] : bbox2[n];
				for (size_t n = 3; n < 6; n++)
					bbox[n] = bbox[n] > bbox2[n] ? bbox[n] : bbox2[n];
			}
			_infos->addParameter("bbox", bbox);
			_infos->addParameter("fbo", m_pickOneObject);
			_infos->addParameter("id", idSelection);
		}
	}
	else if (_infos->nameCommand == "getObjectPickedID") {
		if(m_idSelection != -1 && m_pickingEnabled)
			_infos->addParameter("id", m_idSelection);
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
}

poca::core::CommandInfo ObjectListDisplayCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "fill" || _nameCommand == "pointRendering" || _nameCommand == "shapeRendering" || _nameCommand == "ellipsoidRendering" || _nameCommand == "bboxSelection" || _nameCommand == "outlinePointRendering") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "pointSizeGL") {
		uint32_t val = _parameters.get<uint32_t>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "sortWRTCameraPosition") {
		glm::vec3 cameraPosition, cameraForward;
		bool complete = _parameters.contains("cameraPosition") && _parameters.contains("cameraForward");
		if (complete) {
			cameraPosition = _parameters["cameraPosition"].get<glm::vec3>();
			cameraForward = _parameters["cameraForward"].get<glm::vec3>();
			return poca::core::CommandInfo(false, _nameCommand, "cameraPosition", cameraPosition, "cameraForward", "cameraForward");
		}
	}
	else if (_nameCommand == "updateFeature") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	return poca::opengl::BasicDisplayCommand::createCommand(_nameCommand, _parameters);
}

poca::core::Command* ObjectListDisplayCommand::copy()
{
	return new ObjectListDisplayCommand(*this);
}

void ObjectListDisplayCommand::display(poca::opengl::Camera* _cam, const bool _offscreen, const bool _ssao)
{
	if (!m_objects->isSelected()) return;

	drawElements(_cam, _ssao);
	if (!_offscreen)
		drawPicking(_cam);
}

void ObjectListDisplayCommand::drawElements(poca::opengl::Camera* _cam, const bool _ssao)
{
	if (m_triangleBuffer.empty())
		createDisplay();
	if (m_pickFBO == NULL)
		updatePickingFBO(_cam->getWidth(), _cam->getHeight());

	poca::opengl::Shader* shader = _ssao ? _cam->getShader("objectRenderingSSAOShader") : _cam->getShader("objectRenderingShader");

	bool pointRendering = getParameter<bool>("pointRendering");
	bool outlinePointRendering = getParameter<bool>("outlinePointRendering");
	bool shapeRendering = getParameter<bool>("shapeRendering");
	bool fill = getParameter<bool>("fill");
	bool displayBboxSelection = getParameter<bool>("bboxSelection");
	bool cullFaceActivated = _cam->cullFaceActivated();
	std::string cullFaceType = getParameter<std::string>("cullFaceType");
	bool skeletonRendering = getParameter<bool>("skeletonRendering");
	bool linkRendering = getParameter<bool>("linkRendering");

	const poca::core::BoundingBox bbox = m_objects->boundingBox();
	glm::vec3 orientation = _cam->getRotationSum() * glm::vec3(0.f, 0.f, 1.f);
	glm::vec3 pos(orientation + _cam->getCenter());
	pos *= 2 * _cam->getOriginalDistanceOrtho();

	glEnable(GL_DEPTH_TEST);
	if (cullFaceActivated)
		glEnable(GL_CULL_FACE);
	else
		glDisable(GL_CULL_FACE);

	glDisable(GL_BLEND);
	glCullFace(GL_BACK);
	if (pointRendering) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		uint32_t pointSize = getParameter<uint32_t>("pointSizeGL");
		_cam->drawSphereRendering<poca::core::Vec3mf, float>(m_textureLutID, m_pointBuffer, m_locsFeatureBuffer, m_minOriginalFeature, m_maxOriginalFeature, pointSize, _ssao);
	}

	if (outlinePointRendering) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		uint32_t pointSize = getParameter<uint32_t>("pointSizeGL");
		_cam->drawSphereRendering<poca::core::Vec3mf, float>(m_textureLutID, m_outlinePointBuffer, m_outlineLocsFeatureBuffer, m_minOriginalFeature, m_maxOriginalFeature, pointSize, _ssao);
	}

	if (cullFaceType == "front")
		glCullFace(GL_FRONT);
	else
		glCullFace(GL_BACK);

	glPolygonMode(GL_FRONT_AND_BACK, fill ? GL_FILL : GL_LINE);
	glDisable(GL_BLEND);
	if (shapeRendering) {
		if (m_objects->dimension() == 3) {
			const glm::mat4& proj = _cam->getProjectionMatrix(), & view = _cam->getViewMatrix(), & model = _cam->getModelMatrix();
			shader->use();
			shader->setMat4("model", model);
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
			glActiveTexture(GL_TEXTURE0); 
			glBindTexture(GL_TEXTURE_1D, m_textureLutID);
			const std::vector <size_t>& sizeStrides = m_triangleBuffer.getSizeBuffers();
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);
			glEnableVertexAttribArray(2);
			for (unsigned int chunk = 0; chunk < m_triangleBuffer.getNbBuffers(); chunk++) {
				m_triangleBuffer.bindBuffer(chunk, 0);
				m_triangleNormalBuffer.bindBuffer(chunk, 1);
				m_triangleFeatureBuffer.bindBuffer(chunk, 2);
				glDrawArrays(m_triangleBuffer.getMode(), 0, (GLsizei)sizeStrides[chunk]); // Starting from vertex 0; 3 vertices total -> 1 triangle
			}
			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);
			glDisableVertexAttribArray(2);

			glBindTexture(GL_TEXTURE_1D, 0); // Unbind any textures
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			shader->release();
			GL_CHECK_ERRORS();
		}
		else {
			glDisable(GL_BLEND);
			if(m_lineBuffer.empty() || fill)
				_cam->drawSimpleShader<poca::core::Vec3mf, float>(m_textureLutID, m_triangleBuffer, m_triangleFeatureBuffer, m_minOriginalFeature, m_maxOriginalFeature);
			else
				_cam->drawSimpleShader<poca::core::Vec3mf, float>(m_textureLutID, m_lineBuffer, m_lineFeatureBuffer, m_minOriginalFeature, m_maxOriginalFeature);
		}
	}
	GL_CHECK_ERRORS();

	glDisable(GL_BLEND);
	if (m_idSelection >= 0 && shapeRendering && displayBboxSelection) {
		GLfloat bkColor[4];
		glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
		poca::core::Color4D color = poca::core::contrastColor(poca::core::Color4D(bkColor[0] * 255.f, bkColor[1] * 255.f, bkColor[2] * 255.f, bkColor[3] * 255.f));
		_cam->drawUniformShader<poca::core::Vec3mf>(m_boundingBoxSelection, color);
	}
	GL_CHECK_ERRORS();

	if (m_objects->hasSkeletons()) {
		if (m_skeletonBuffer.empty())
			createDisplaySkeleton();

		if (skeletonRendering && !m_skeletonBuffer.empty())
			_cam->drawSimpleShader<poca::core::Vec3mf, poca::core::Color4D>(m_skeletonBuffer, m_colorSkeletonBuffer);

		if (linkRendering && !m_linksBuffer.empty())
			_cam->drawSimpleShader<poca::core::Vec3mf, poca::core::Color4D>(m_linksBuffer, m_colorLinksBuffer);
	}
	//m_skeletonBuffer

	drawEllipsoid(_cam);
	GL_CHECK_ERRORS();
}

void ObjectListDisplayCommand::drawEllipsoid(poca::opengl::Camera* _cam)
{
	bool ellipsoidRendering = getParameter<bool>("ellipsoidRendering");
	if (m_ellipsoidTransformBuffer.empty() || !ellipsoidRendering)
		return;

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
	glDisable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	poca::opengl::Shader* shader = _cam->getShader("3DInstanceRenderingShader");
	const glm::mat4& proj = _cam->getProjectionMatrix(), & view = _cam->getViewMatrix(), & model = _cam->getModelMatrix();
	shader->use();
	shader->setMat4("MVP", proj * view * model);
	shader->setBool("useSingleColor", false);

	shader->setInt("lutTexture", 0);
	shader->setFloat("minFeatureValue", m_minOriginalFeature);
	shader->setFloat("maxFeatureValue", m_maxOriginalFeature);

	shader->setVec4v("clipPlanes", _cam->getClipPlanes());
	shader->setInt("nbClipPlanes", _cam->nbClippingPlanes());
	shader->setBool("clip", _cam->clip());

	poca::opengl::HelperSingleton* helper = poca::opengl::HelperSingleton::instance();
	poca::opengl::QuadSingleGLBuffer <float>& ellipsoidBuffer = helper->getEllipsoidBuffer();
	poca::opengl::QuadSingleGLBuffer <GLushort>& ellipsoidIndicesBuffer = helper->getEllipsoidIndicesBuffer();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_1D, m_textureLutID);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(9);
	ellipsoidBuffer.bindBuffer(0);
	ellipsoidIndicesBuffer.bindBuffer(1);
	m_ellipsoidFeatureBuffer.bindBuffer(9);
	// This is the important bit... set the divisor for the color array
	// to 1 to get OpenGL to give us a new value of "feature" per-instance
	// rather than per-vertex.
	glVertexAttribDivisor(9, 1);
	for (int i = 0; i < 4; i++)
	{
		glEnableVertexAttribArray(5 + i);
		m_ellipsoidTransformBuffer.bindBuffer(5 + i, (void*)(4 * sizeof(float) * i));
	}
	glDrawElementsInstanced(GL_QUADS, helper->getNbIndicesUnitSphere(), GL_UNSIGNED_SHORT, 0, m_objects->nbElements());
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(9);
	for (int i = 0; i < 4; i++)
		glDisableVertexAttribArray(5 + i);
	glBindTexture(GL_TEXTURE_1D, 0); // Unbind any textures
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	shader->release();
	GL_CHECK_ERRORS();
}


void ObjectListDisplayCommand::drawPicking(poca::opengl::Camera* _cam)
{
	if (m_pickFBO == NULL)
		updatePickingFBO(_cam->getWidth(), _cam->getHeight());

	if (m_pickFBO == NULL) return;

	bool pointRendering = getParameter<bool>("pointRendering");
	bool shapeRendering = getParameter<bool>("shapeRendering");
	bool fill = getParameter<bool>("fill");

	glEnable(GL_DEPTH_TEST);
	GLfloat bkColor[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	bool success = m_pickFBO->bind();
	if (!success) std::cout << "Problem with binding" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (shapeRendering)
		_cam->drawPickingShader<poca::core::Vec3mf, float>(m_triangleBuffer, m_idBuffer, m_triangleFeatureBuffer, m_minOriginalFeature);
	else
		_cam->drawPickingShader<poca::core::Vec3mf, float>(m_pointBuffer, m_idLocsBuffer, m_locsFeatureBuffer, m_minOriginalFeature);

	success = m_pickFBO->release();
	if (!success) std::cout << "Problem with releasing" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, _cam->defaultFramebufferObject());
	glClearColor(bkColor[0], bkColor[1], bkColor[2], bkColor[3]);
	GL_CHECK_ERRORS();
}

void ObjectListDisplayCommand::displayZoomToBBox(poca::opengl::Camera* _cam, const poca::core::BoundingBox& _bbox)
{
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);

	if (m_pickOneObject == NULL) {
		int sizeImage = 400;
		m_pickOneObject = new QOpenGLFramebufferObject(sizeImage, sizeImage, QOpenGLFramebufferObject::Depth, GL_TEXTURE_2D, GL_RGB);
		glBindTexture(GL_TEXTURE_2D, m_pickOneObject->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, sizeImage, sizeImage, 0, GL_RGB, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);

	}

	bool fill = getParameter<bool>("fill");
	bool cullFaceActivated = _cam->cullFaceActivated();

	glViewport(0, 0, 400, 400);
	//Compute projection and model matrices
	float w = _bbox[3] - _bbox[0], h = _bbox[4] - _bbox[1], t = _bbox[5] - _bbox[2];
	float translationX = _bbox[0] + (_bbox[3] - _bbox[0]) / 2.f;
	float translationY = _bbox[1] + (_bbox[4] - _bbox[1]) / 2.f;
	float translationZ = _bbox[2] + (_bbox[5] - _bbox[2]) / 2.f;
	glm::mat4 model = glm::translate(glm::mat4(1.f), glm::vec3(-translationX, -translationY, -translationZ));

	float distanceOrtho = w > h ? w / 2 : h / 2;
	float projLeft = -distanceOrtho;
	float projRight = distanceOrtho;
	float projBottom = -distanceOrtho;
	float projUp = distanceOrtho;
	float projNear = -distanceOrtho * sqrt(3);
	float projFar = distanceOrtho * sqrt(3);
	glm::mat4 proj = glm::ortho(projLeft, projRight, projUp, projBottom, projNear, projFar);
	const glm::mat4& view = _cam->getViewMatrix();

	bool success = m_pickOneObject->bind();
	if (!success) std::cout << "Problem with binding" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if (cullFaceActivated)
		glEnable(GL_CULL_FACE);
	else
		glDisable(GL_CULL_FACE);
	glCullFace(GL_FRONT);
	glPolygonMode(GL_FRONT_AND_BACK, fill ? GL_FILL : GL_LINE);

	glLineWidth(4.f);

	if (m_objects->dimension() == 3) {
		const poca::core::BoundingBox bbox = m_objects->boundingBox();
		glm::vec3 orientation = _cam->getRotationSum() * glm::vec3(0.f, 0.f, 1.f);
		glm::vec3 pos(orientation + _cam->getCenter());
		pos *= 2 * _cam->getOriginalDistanceOrtho();

		poca::opengl::Shader* shader = _cam->getShader("objectRenderingShader");
		shader->use();
		shader->setMat4("model", model);
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
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_1D, m_textureLutID);
		const std::vector <size_t>& sizeStrides = m_triangleBuffer.getSizeBuffers();
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		for (unsigned int chunk = 0; chunk < m_triangleBuffer.getNbBuffers(); chunk++) {
			m_triangleBuffer.bindBuffer(chunk, 0);
			m_triangleNormalBuffer.bindBuffer(chunk, 1);
			m_triangleFeatureBuffer.bindBuffer(chunk, 2);
			glDrawArrays(m_triangleBuffer.getMode(), 0, (GLsizei)sizeStrides[chunk]); // Starting from vertex 0; 3 vertices total -> 1 triangle
		}
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);

		glBindTexture(GL_TEXTURE_1D, 0); // Unbind any textures
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		shader->release();
	}
	else {
		poca::opengl::Shader* shader = _cam->getShader("simpleShader");
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setInt("lutTexture", 0);
		shader->setFloat("minFeatureValue", m_minOriginalFeature);
		shader->setFloat("maxFeatureValue", m_maxOriginalFeature);
		shader->setFloat("useSpecialColors", 0);
		shader->setVec4v("clipPlanes", _cam->getClipPlanes());
		shader->setInt("nbClipPlanes", _cam->nbClippingPlanes());
		shader->setBool("clip", _cam->clip());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_1D, m_textureLutID);

		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
		const std::vector <size_t>& sizeStrides = m_triangleBuffer.getSizeBuffers();
		for (unsigned int chunk = 0; chunk < m_triangleBuffer.getNbBuffers(); chunk++) {
			glEnableVertexAttribArray(0);
			m_triangleBuffer.bindBuffer(chunk, 0);
			glEnableVertexAttribArray(2);
			m_triangleFeatureBuffer.bindBuffer(chunk, 2);
			glDrawArrays(GL_TRIANGLES, 0, (GLsizei)sizeStrides[chunk]); // Starting from vertex 0; 3 vertices total -> 1 triangle
			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(2);
		}
		glBindTexture(GL_TEXTURE_1D, 0); // Unbind any textures
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		success = m_pickOneObject->release();
		if (!success) std::cout << "Problem with releasing" << std::endl;
		shader->release();
		glDisable(GL_BLEND);
	}

	GL_CHECK_ERRORS();

	glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
}

void ObjectListDisplayCommand::createDisplay()
{
	freeGPUMemory();

	m_textureLutID = poca::opengl::HelperSingleton::instance()->generateLutTexture(m_objects->getPalette());

	//For locs
	std::vector <poca::core::Vec3mf> locsB;
	m_objects->generateLocs(locsB);
	std::vector <poca::core::Vec3mf> normsB;
	m_objects->generateNormalLocs(normsB);
	std::vector <float> idsLocs;
	m_objects->generateLocsPickingIndices(idsLocs);
	m_pointBuffer.generateBuffer(locsB.size(), 3, GL_FLOAT);
	m_pointNormalBuffer.generateBuffer(normsB.size(), 3, GL_FLOAT);
	m_locsFeatureBuffer.generateBuffer(locsB.size(), 1, GL_FLOAT);
	m_idLocsBuffer.generateBuffer(locsB.size(), 1, GL_FLOAT);
	m_pointBuffer.updateBuffer(locsB.data());
	m_pointNormalBuffer.updateBuffer(normsB.data());
	m_idLocsBuffer.updateBuffer(idsLocs.data());
	
	std::vector <poca::core::Vec3mf> outlineLocs;
	m_objects->generateOutlineLocs(outlineLocs);
	m_outlinePointBuffer.generateBuffer(outlineLocs.size(), 3, GL_FLOAT);
	m_outlineLocsFeatureBuffer.generateBuffer(outlineLocs.size(), 1, GL_FLOAT);
	m_outlinePointBuffer.updateBuffer(outlineLocs.data());

	const std::vector <poca::core::Vec3mf>& normalOutline = m_objects->getNormalOutlineLocs();
	if (!normalOutline.empty()) {
		std::vector <poca::core::Vec3mf> normalsOutLoc(outlineLocs.size() * 2);
		for (size_t n = 0; n < outlineLocs.size(); n++) {
			normalsOutLoc[2 * n] = outlineLocs[n];
			normalsOutLoc[2 * n + 1] = outlineLocs[n] + 10.f * normalOutline[n];
		}
		m_normalsBuffer.generateBuffer(normalsOutLoc.size(), 3, GL_FLOAT);
		m_normalsBuffer.updateBuffer(normalsOutLoc.data());
	}

	//For objects
	std::vector <poca::core::Vec3mf> triangles;
	m_objects->generateTriangles(triangles);
	std::vector <poca::core::Vec3mf> normals;
	m_objects->generateNormals(normals);
	std::vector <float> ids;
	m_objects->generatePickingIndices(ids);
	m_triangleBuffer.generateBuffer(triangles.size(), 512 * 512, 3, GL_FLOAT);
	m_triangleNormalBuffer.generateBuffer(normals.size(), 512 * 512, 3, GL_FLOAT);
	m_triangleFeatureBuffer.generateBuffer(triangles.size(), 512 * 512, 1, GL_FLOAT);
	m_idBuffer.generateBuffer(triangles.size(), 512 * 512, 1, GL_FLOAT);
	m_triangleBuffer.updateBuffer(triangles.data());
	m_triangleNormalBuffer.updateBuffer(normals.data());
	m_idBuffer.updateBuffer(ids.data());

	m_boundingBoxSelection.generateBuffer(24, 3, GL_FLOAT);

	if (m_objects->dimension() == 2) {
		std::vector <poca::core::Vec3mf> outlines;
		m_objects->generateOutlines(outlines);
		m_lineBuffer.generateBuffer(outlines.size(), 3, GL_FLOAT);
		m_lineBuffer.updateBuffer(outlines.data());

		m_lineFeatureBuffer.generateBuffer(outlines.size(), 1, GL_FLOAT);
	}

	if (m_objects->dimension() == 3) {
		if (m_objects->hasData("major")) {
			const std::vector < std::array<poca::core::Vec3mf, 3>>& axisPCA = m_objects->getAxisObjects();
			const std::vector<float>& major = m_objects->getMyData("major")->getData<float>();
			const std::vector<float>& minor = m_objects->getMyData("minor")->getData<float>();
			const std::vector<float>& minor2 = m_objects->getMyData("minor2")->getData<float>();

			std::vector <glm::mat4> matrices(m_objects->nbElements());
			for (unsigned int n = 0; n < m_objects->nbElements(); n++) {
				matrices[n] = glm::mat4(1);

				glm::mat3 rotation = glm::mat3(axisPCA[n][0].x(), axisPCA[n][0].y(), axisPCA[n][0].z(), axisPCA[n][1].x(), axisPCA[n][1].y(), axisPCA[n][1].z(), axisPCA[n][2].x(), axisPCA[n][2].y(), axisPCA[n][2].z());

				poca::core::Vec3mf centroidTmp = m_objects->computeBarycenterElement(n);
				const glm::vec3& centroid = glm::vec3(centroidTmp[0], centroidTmp[1], centroidTmp[2]);// m_centroidEllipsoid[n];

				glm::vec3 values = glm::vec3(major[n] / 2.f, minor[n] / 2.f, minor2[n] / 2.f);
				matrices[n] = glm::translate(matrices[n], centroid);
				matrices[n] *= glm::mat4(rotation);
				matrices[n] = glm::scale(matrices[n], values);
			}

			m_ellipsoidTransformBuffer.generateBuffer(matrices.size(), 4, GL_FLOAT);
			m_ellipsoidTransformBuffer.updateBuffer(matrices.data());

			m_ellipsoidFeatureBuffer.generateBuffer(major.size(), 1, GL_FLOAT);
		}
	}

	poca::core::HistogramInterface* hist = m_objects->getCurrentHistogram();
	generateFeatureBuffer(hist);
}

void ObjectListDisplayCommand::createDisplaySkeleton()
{
	if (!m_objects->hasSkeletons()) return;
	poca::geometry::ObjectListMesh* omesh = dynamic_cast <poca::geometry::ObjectListMesh*>(m_objects);
	const poca::core::MyArrayVec3mf& skeletons = omesh->getSkeletons();
	const poca::core::MyArrayVec3mf& links = omesh->getLinks();

	const auto& skeletonsId = skeletons.getFirstElements();
	const auto& linksId = links.getFirstElements();

	std::vector<poca::core::Color4D> colorSkeletons(skeletons.nbData()), colorLinks(links.nbData());
	for (size_t n = 0; n < skeletonsId.size() - 1; n++) {
		poca::core::Color4D color(((float)rand()) / ((float)RAND_MAX), ((float)rand()) / ((float)RAND_MAX), ((float)rand()) / ((float)RAND_MAX), 1.f);

		for (size_t idx = skeletonsId[n]; idx < skeletonsId[n + 1]; idx++)
			colorSkeletons[idx] = color;

		for (size_t idx = linksId[n]; idx < linksId[n + 1]; idx++)
			colorLinks[idx] = color;
	}

	m_skeletonBuffer.generateBuffer(skeletons.nbData(), 3, GL_FLOAT);
	m_skeletonBuffer.updateBuffer(skeletons.allElements());
	m_colorSkeletonBuffer.generateBuffer(colorSkeletons.size(), 4, GL_FLOAT);
	m_colorSkeletonBuffer.updateBuffer(colorSkeletons.data());

	m_linksBuffer.generateBuffer(links.nbData(), 3, GL_FLOAT);
	m_linksBuffer.updateBuffer(links.allElements());
	m_colorLinksBuffer.generateBuffer(colorLinks.size(), 4, GL_FLOAT);
	m_colorLinksBuffer.updateBuffer(colorLinks.data());
}

void ObjectListDisplayCommand::freeGPUMemory()
{
	m_pointBuffer.freeGPUMemory();
	m_idLocsBuffer.freeGPUMemory();
	m_locsFeatureBuffer.freeGPUMemory();

	m_triangleBuffer.freeGPUMemory();
	m_lineBuffer.freeGPUMemory();
	m_idBuffer.freeGPUMemory();
	m_triangleFeatureBuffer.freeGPUMemory();
	m_lineFeatureBuffer.freeGPUMemory();

	m_boundingBoxSelection.freeGPUMemory();

	m_triangleBuffer.freeGPUMemory();

	m_ellipsoidTransformBuffer.freeGPUMemory();

	if (m_pickOneObject != NULL)
		delete m_pickOneObject;
	m_pickOneObject = NULL;

	m_textureLutID = 0;
}

void ObjectListDisplayCommand::generateFeatureBuffer(poca::core::HistogramInterface* _histogram)
{
	if (_histogram == NULL)
		_histogram = m_objects->getCurrentHistogram();
	poca::core::Histogram<float>* histogram = dynamic_cast<poca::core::Histogram<float>*>(_histogram);
	const std::vector<float>& values = histogram->getValues();
	const std::vector<bool>& selection = m_objects->getSelection();
	m_minOriginalFeature = _histogram->getMin();
	m_maxOriginalFeature = _histogram->getMax();
	m_actualValueFeature = m_maxOriginalFeature;

	std::vector <float> featureLocs, featureValues, featureOutlines, featureOutlineLocs, featureEllipsoid(values.size());
	if (m_objects->isHiLow()) {
		float inter = m_maxOriginalFeature - m_minOriginalFeature;
		float selectedValue = m_minOriginalFeature + inter / 4.f;
		float notSelectedValue = m_minOriginalFeature + inter * (3.f / 4.f);
		m_objects->getLocsFeatureInSelectionHiLow(featureLocs, selection, selectedValue, notSelectedValue);
		m_objects->getFeatureInSelectionHiLow(featureValues, selection, selectedValue, notSelectedValue);
		m_objects->getOutlineLocsFeatureInSelectionHiLow(featureOutlineLocs, selection, selectedValue, notSelectedValue);
		if (!m_lineBuffer.empty())
			m_objects->getOutlinesFeatureInSelectionHiLow(featureOutlines, selection, selectedValue, notSelectedValue);

		for (auto n = 0; n < values.size(); n++)
			featureEllipsoid[n] = selection[n] ? selectedValue : notSelectedValue;
	}
	else {
		m_objects->getLocsFeatureInSelection(featureLocs, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		m_objects->getFeatureInSelection(featureValues, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		m_objects->getOutlineLocsFeatureInSelection(featureOutlineLocs, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);
		if (!m_lineBuffer.empty())
			m_objects->getOutlinesFeatureInSelection(featureOutlines, values, selection, poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER);

		for (auto n = 0; n < values.size(); n++)
			featureEllipsoid[n] = selection[n] ? values[n] : poca::opengl::Shader::MIN_VALUE_FEATURE_SHADER;
	}
	m_locsFeatureBuffer.updateBuffer(featureLocs.data());
	m_triangleFeatureBuffer.updateBuffer(featureValues.data());
	m_outlineLocsFeatureBuffer.updateBuffer(featureOutlineLocs.data());
	if (!m_lineBuffer.empty())
		m_lineFeatureBuffer.updateBuffer(featureOutlines.data());

	if (!m_ellipsoidFeatureBuffer.empty())
		m_ellipsoidFeatureBuffer.updateBuffer(featureEllipsoid.data());
}

QString ObjectListDisplayCommand::getInfosTriangle(const int _id) const
{
	QString text;
	if (_id >= 0) {
		text.append(QString("Object id: %1").arg(_id + 1));
		poca::core::stringList nameData = m_objects->getNameData();
		for (std::string type : nameData) {
			float val = m_objects->getMyData(type)->getData<float>()[_id];
			text.append(QString("\n%1: %2").arg(type.c_str()).arg(val));
		}
	}
	return text;
}

void ObjectListDisplayCommand::generateBoundingBoxSelection(const int _idx)
{
	if (_idx < 0) return;
	bool shapeRendering = getParameter<bool>("shapeRendering");
	if (!shapeRendering) return;
	
	poca::core::BoundingBox bbox = m_objects->computeBoundingBoxElement(_idx);

	std::vector <poca::core::Vec3mf> cube(24);
	poca::geometry::createCubeFromVector(cube, bbox);
	m_boundingBoxSelection.updateBuffer(cube.data());
}

void ObjectListDisplayCommand::sortWrtCameraPosition(const glm::vec3& _cameraPosition, const glm::vec3& _cameraForwardVec)
{
	if (m_objects->dimension() == 2) return;
#ifdef NO_CUDA
	const std::vector <uint32_t>& locs = m_objects->getLocsObjects().getData();
	const float* xs = m_objects->getXs(), * ys = m_objects->getXs(), * zs = m_objects->getXs();
	std::vector <uint32_t> indices(locs.size());
	std::iota(std::begin(indices), std::end(indices), 0);

	//Compute a vector of distances of the points to the camera position
	std::vector <float> distances(locs.size());
	for (size_t n = 0; n < locs.size(); n++) {
		uint32_t index = locs[n];
		distances[n] = glm::dot(glm::vec3(xs[index], ys[index], zs[index]) - _cameraPosition, _cameraForwardVec);
	}
	//Sort wrt the distance to the camera position
	std::sort(indices.begin(), indices.end(),
		[&](int A, int B) -> bool {
			return distances[A] < distances[B];
		});
	m_pointBuffer.updateIndices(indices);
#else
	size_t numData = m_objects->getLocsObjects().nbData();
	std::vector <uint32_t> indices(numData);
	std::iota(std::begin(indices), std::end(indices), 0);
	const float* xs = m_objects->getXs(), * ys = m_objects->getXs(), * zs = m_objects->getXs();
	if (static_cast <poca::geometry::ObjectListMesh*>(m_objects) != NULL) {
		sortArrayWRTPoint_GPU(xs, ys, zs, numData, poca::core::Vec3mf(_cameraPosition.x, _cameraPosition.y, _cameraPosition.z), indices);
	}
	else {
		const std::vector <uint32_t>& locs = m_objects->getLocsObjects().getData();
		//Compute a vector of distances of the points to the camera position
		std::vector <float> distances(locs.size());
#pragma omp parallel for
		for (size_t n = 0; n < locs.size(); n++) {
			uint32_t index = locs[n];
			distances[n] = glm::dot(glm::vec3(xs[index], ys[index], zs[index]) - _cameraPosition, _cameraForwardVec);
		}
		sortArrayWRTKeys(distances, indices);
	}
	m_pointBuffer.updateIndices(indices);
#endif
}

