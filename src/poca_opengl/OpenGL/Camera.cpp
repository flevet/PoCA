/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Camera.cpp
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
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/projection.hpp>
#include <iostream>
#include <QtGui/QMouseEvent>
#include <QtGui/QPainter>
#include <QtGui/QPainterPath>
#include <QtGui/QOpenGLFramebufferObject>
#include <QtCore/qmath.h>
#include <QtWidgets/QFileDialog>
#include <CGAL/intersections.h>

#include <General/Command.hpp>
#include <General/CommandableObject.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <DesignPatterns/MediatorWObjectFWidget.hpp>
#include <Interfaces/ROIInterface.hpp>
#include <General/Vec6.hpp>
#include <General/Vec2.hpp>
#include <General/Misc.h>
#include <OpenGL/Helper.h>
#include <Geometry/CGAL_includes.hpp>

#include "Camera.hpp"
#include "Shader.hpp"
#include "TextDisplayer.hpp"
#include "../General/Roi.hpp"

char* vs = "#version 330 core\n"
"layout(location = 0) in vec3 vertexPosition_modelspace;\n"
"layout(location = 2) in float vertexFeature;\n"
"layout(location = 3) in vec4 vertexColor;\n"
"layout(location = 4) in vec3 vertexNormal;\n"
"uniform mat4 MVP;\n"
"const int MAX_CLIPPING_PLANES = 50;\n"
"uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];\n"
"uniform int nbClipPlanes;\n"
"out float feature;\n"
"out vec4 colorIn;\n"
"out vec3 normal;\n"
"out float vclipDistance;\n"
"void main() {\n"
"	vec4 pos = vec4(vertexPosition_modelspace, 1);\n"
"	gl_Position = MVP * pos;\n"
"	feature = vertexFeature;\n"
"	colorIn = vertexColor;\n"
"	normal = vertexNormal;\n"
"	vclipDistance = 3.402823466e+38;\n"
"	for(int n = 0; n < nbClipPlanes; n++){\n"
"		float d = dot(pos, clipPlanes[n]);\n"
"		vclipDistance = d < vclipDistance ? d : vclipDistance;\n"
"	}\n"
"}";

char* fs = "#version 330 core\n"
"in float feature;\n"
"in vec4 colorIn;\n"
"in vec3 normal;\n"
"in float vclipDistance;\n"
"out vec4 color;\n"
"uniform sampler1D lutTexture;\n"
"uniform float minFeatureValue;\n"
"uniform float maxFeatureValue;\n"
"uniform float alpha;\n"
"uniform bool useSpecialColors;\n"
"uniform bool activatedCulling; \n"
"uniform vec3 cameraForward; \n"
"uniform bool clip;\n"
"void main() {\n"
"	if (clip && vclipDistance < 0.f)\n"
"		discard;\n"
"	if (activatedCulling) {\n"
"		float res = dot(cameraForward, normal); \n"
"		if (res > 0.f)\n"
"			discard; \n"
"	}\n"
"	if (useSpecialColors) {\n"
"		if (colorIn.a == 0.)\n"
"			discard;\n"
"		color = colorIn;\n"
"	}\n"
"	else {\n"
"		if (feature < minFeatureValue)\n"
"			discard;\n"
"		float inter = maxFeatureValue - minFeatureValue;\n"
"		color = vec4(texture(lutTexture, ((feature - minFeatureValue) / inter)).xyz, alpha);\n"//0.01);\n"
"	}\n"
"}";

char* vsPick = "#version 330 core\n"
"layout(location = 0) in vec3 vertexPosition_modelspace;\n"
"layout(location = 1) in float vertexIndex;\n"
"layout(location = 2) in float vertexFeature;\n"
"#define FLT_MAX 3.402823466e+38;\n"
"uniform mat4 MVP;\n"
"uniform bool hasFeature;\n"
"const int MAX_CLIPPING_PLANES = 50;\n"
"uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];\n"
"uniform int nbClipPlanes;\n"
"out float id;\n"
"out float feature;\n"
"out float vclipDistance;\n"
"void main() {\n"
"	vec4 pos = vec4(vertexPosition_modelspace, 1);\n"
"	gl_Position = MVP * pos;\n"
"	id = vertexIndex;\n"
"	feature = hasFeature ? vertexFeature : FLT_MAX;\n"
"	vclipDistance = 3.402823466e+38;\n"
"	for(int n = 0; n < nbClipPlanes; n++){\n"
"		float d = dot(pos, clipPlanes[n]);\n"
"		vclipDistance = d < vclipDistance ? d : vclipDistance;\n"
"	}\n"
"}";

char* fsPick = "#version 330 core\n"
"layout (location = 0) out float gIndex;\n"
"uniform float minFeatureValue;\n"
"uniform bool clip;\n"
"in float id;\n"
"in float feature;\n"
"in float vclipDistance;\n"
"void main() {\n"
"	if (clip && vclipDistance < 0.f)\n"
"		discard;\n"
"	if(feature < minFeatureValue)\n"
"		discard;\n"
"	gIndex = id;\n"
"}";

char* fsUniformColor = "#version 330 core\n"
"out vec4 color;\n"
"in vec3 normal;\n"
"in float vclipDistance;\n"
"uniform vec4 singleColor;\n"
"uniform bool activatedCulling; \n"
"uniform vec3 cameraForward; \n"
"uniform bool clip;\n"
"void main() {\n"
"	if (clip && vclipDistance < 0.f)\n"
"		discard;\n"
"	if (activatedCulling) {\n"
"		float res = dot(cameraForward, normal); \n"
"		if (res > 0.f)\n"
"			discard; \n"
"	}\n"
"	color = singleColor;\n"
"}";

char* vsStipple = "#version 330\n"
"layout(location = 0) in vec3 inPos;\n"
"flat out vec3 startPos;\n"
"out vec3 vertPos;\n"
"out float vclipDistance;\n"
"uniform mat4 MVP;\n"
"const int MAX_CLIPPING_PLANES = 50;\n"
"uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];\n"
"uniform int nbClipPlanes;\n"
"void main()\n"
"{\n"
"	vec4 pos = MVP * vec4(inPos, 1.0);\n"
"	gl_Position = pos;\n"
"	vertPos = pos.xyz / pos.w;\n"
"	startPos = vertPos;\n"
"	vclipDistance = 3.402823466e+38;\n"
"	for(int n = 0; n < nbClipPlanes; n++){\n"
"		float d = dot(pos, clipPlanes[n]);\n"
"		vclipDistance = d < vclipDistance ? d : vclipDistance;\n"
"	}\n""}";

char* fsStipple = "#version 330\n"
"flat in vec3 startPos;\n"
"in vec3 vertPos;\n"
"in float vclipDistance;\n"
"out vec4 fragColor;\n"
"uniform vec4 singleColor;\n"
"uniform vec2  u_resolution;\n"
"uniform float u_dashSize;\n"
"uniform float u_gapSize;\n"
"uniform bool clip;\n"
"void main()\n"
"{"
"	if (clip && vclipDistance < 0.f)\n"
"		discard;\n"
"	vec2  dir = (vertPos.xy - startPos.xy) * u_resolution / 2.0;\n"
"	float dist = length(dir);\n"
"	if (fract(dist / (u_dashSize + u_gapSize)) > u_dashSize / (u_dashSize + u_gapSize))\n"
"		discard;\n"
"	fragColor = singleColor;\n"
"}";

char* vsTexture = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec2 aTexCoords;\n"
"out vec2 TexCoords;\n"
"void main() {\n"
"	TexCoords = aTexCoords;\n"
"	vec4 pos = vec4(aPos, 1.0);"
"	gl_Position = pos;\n"
"}";

char* fsTexture = "#version 330 core\n"
"in vec2 TexCoords;\n"
"out vec4 color;\n"
"uniform sampler2D image;\n"
"void main() {\n"
"	color = texture( image, TexCoords );\n"
"}";

char* fsTextureFBO = "#version 330 core\n"
"in vec2 TexCoords;\n"
"out vec4 color;\n"
"uniform sampler2D offscreen;\n"
"uniform sampler1D lut;\n"
"void main() {\n"
"	float coord = texture( offscreen, TexCoords ).x;\n"
"	if(coord <= 0.f)\n"
"		discard;\n"
"	color = texture( lut, coord );\n"
"}";

char* vsPointRenderingRotation = "#version 330 core\n"
"layout(location = 0) in vec3 vertexPosition_modelspace;\n"
"layout(location = 2) in float vertexFeature;\n"
"layout(location = 3) in vec4 vertexColor;\n"
"uniform mat4 MVP;\n"
"const int MAX_CLIPPING_PLANES = 50;\n"
"uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];\n"
"uniform int nbClipPlanes;\n"
"uniform float sizePoints;\n"
"uniform float antialias;\n"
"out float feature;\n"
"out vec4 colorIn;\n"
"out float vclipDistance;\n"
"void main() {\n"
"	vec4 pos = vec4(vertexPosition_modelspace, 1);\n"
"	gl_Position = MVP * pos;\n"
"	gl_PointSize = sizePoints;\n"
"	feature = vertexFeature;\n"
"	colorIn = vertexColor;\n"
"	vclipDistance = 3.402823466e+38;\n"
"	for(int n = 0; n < nbClipPlanes; n++){\n"
"		float d = dot(pos, clipPlanes[n]);\n"
"		vclipDistance = d < vclipDistance ? d : vclipDistance;\n"
"	}\n"
"}";

char* fsPointRenderingRotation = "#version 330 core\n"
"in float feature;\n"
"in vec4 colorIn;\n"
"in float vclipDistance;\n"
"out vec4 color;\n"
"uniform sampler1D lutTexture;\n"
"uniform float minFeatureValue;\n"
"uniform float maxFeatureValue;\n"
"uniform bool useSpecialColors;\n"
"uniform bool clip;\n"
"void main() {\n"
"	if (clip && vclipDistance < 0.f)\n"
"		discard;\n"
"	if (useSpecialColors) {\n"
"		color = vec4(colorIn.rgb, 1.0);\n"
"	}\n"
"	else {\n"
"		if (feature < minFeatureValue)\n"
"			discard;\n"
"		float inter = maxFeatureValue - minFeatureValue;\n"
"		color = vec4(texture(lutTexture, ((feature - minFeatureValue) / inter)).xyz, 1.0);\n"
"	}\n"
"}";

namespace glm {
	void to_json(nlohmann::json& j, const glm::mat4& m) {
		j = { { "m00", m[0][0] }, { "m01", m[0][1] },  { "m02", m[0][2] }, { "m03", m[0][3] },
				{ "m10", m[1][0] }, { "m11", m[1][1] },  { "m12", m[1][2] }, { "m13", m[1][3] },
				{ "m20", m[2][0] }, { "m21", m[2][1] },  { "m22", m[2][2] }, { "m23", m[2][3] },
				{ "m30", m[3][0] }, { "m31", m[3][1] },  { "m32", m[3][2] }, { "m33", m[3][3] } };
	};

	void from_json(const nlohmann::json& j, glm::mat4& m) {
		m[0][0] = j.at("m00").get<float>();
		m[0][1] = j.at("m01").get<float>();
		m[0][2] = j.at("m02").get<float>();
		m[0][3] = j.at("m03").get<float>();
		m[1][0] = j.at("m10").get<float>();
		m[1][1] = j.at("m11").get<float>();
		m[1][2] = j.at("m12").get<float>();
		m[1][3] = j.at("m13").get<float>();
		m[2][0] = j.at("m20").get<float>();
		m[2][1] = j.at("m21").get<float>();
		m[2][2] = j.at("m22").get<float>();
		m[2][3] = j.at("m23").get<float>();
		m[3][0] = j.at("m30").get<float>();
		m[3][1] = j.at("m31").get<float>();
		m[3][2] = j.at("m32").get<float>();
		m[3][3] = j.at("m33").get<float>();
	}

	void to_json(nlohmann::json& j, const glm::vec3& P) {
		j = { { "x", P.x }, { "y", P.y }, { "z", P.z } };
	};

	void from_json(const nlohmann::json& j, glm::vec3& P) {
		P.x = j.at("x").get<float>();
		P.y = j.at("y").get<float>();
		P.z = j.at("z").get<float>();
	}

	void to_json(nlohmann::json& j, const glm::quat& q) {
		j = { { "x", q.x }, { "y", q.y }, { "z", q.z }, { "w", q.w } };
	};

	void from_json(const nlohmann::json& j, glm::quat& q) {
		q.x = j.at("x").get<float>();
		q.y = j.at("y").get<float>();
		q.z = j.at("z").get<float>();
		q.w = j.at("w").get<float>();
	}
}

namespace poca::opengl {

	double clockToMilliseconds(clock_t ticks) {
		// units/(units/time) => time (seconds) * 1000 = milliseconds
		return (ticks / (double)CLOCKS_PER_SEC) * 1000.0;
	}

	glm::vec3 cubeNormals[6] = { glm::vec3(0., 0., 1.), glm::vec3(0., 1., 0.), glm::vec3(1., 0., 0.),
		glm::vec3(0., 0., -1.), glm::vec3(0., -1., 0.), glm::vec3(-1., 0., 0.) };

	std::array <size_t, 4> cubeFaceIndexPoints[6] = { std::array <size_t, 4>{4, 5, 6, 7}, std::array <size_t, 4>{1, 5, 6, 2} , std::array <size_t, 4>{3, 2, 6, 7},
		std::array <size_t, 4>{0, 1, 2, 3} , std::array <size_t, 4>{0, 4, 7, 3} , std::array <size_t, 4>{0, 1, 5, 4} };

	std::array <size_t, 4> cubeFaceIndexEdges[6] = { std::array <size_t, 4>{4, 5, 6, 7}, std::array <size_t, 4>{1, 8, 5, 10} , std::array <size_t, 4>{2, 10, 6, 11},
		std::array <size_t, 4>{0, 1, 2, 3} , std::array <size_t, 4>{3, 9, 7, 11} , std::array <size_t, 4>{0, 8, 4, 9} };

	std::array <size_t, 2> cubeEdgeIndexPoints[12] = { std::array <size_t, 2>{0, 1}, std::array <size_t, 2>{1, 2}, std::array <size_t, 2>{2, 3}, std::array <size_t, 2>{3, 0},
		std::array <size_t, 2>{4, 5}, std::array <size_t, 2>{5, 6}, std::array <size_t, 2>{6, 7}, std::array <size_t, 2>{7, 4},
		std::array <size_t, 2>{1, 5}, std::array <size_t, 2>{4, 0}, std::array <size_t, 2>{2, 6}, std::array <size_t, 2>{7, 3} };

	std::array <size_t, 2> cubeEdgeIndexFaces[12] = { std::array <size_t, 2>{3, 5}, std::array <size_t, 2>{3, 1}, std::array <size_t, 2>{3, 2}, std::array <size_t, 2>{3, 4},
		std::array <size_t, 2>{0, 5}, std::array <size_t, 2>{0, 1}, std::array <size_t, 2>{0, 2}, std::array <size_t, 2>{0, 4},
		std::array <size_t, 2>{5, 1}, std::array <size_t, 2>{4, 5}, std::array <size_t, 2>{2, 1}, std::array <size_t, 2>{2, 4} };

	float Shader::MIN_VALUE_FEATURE_SHADER = -10000.f;

	size_t cptAnimation = 0, currentcommandAnimation = 0;
	poca::core::CommandInfos commandInfos;

	InfosObjectImages::InfosObjectImages() :m_fbo(NULL)
	{

	}

	InfosObjectImages::InfosObjectImages(QOpenGLFramebufferObject* _fbo, const poca::core::Vec4mf& _rect, const poca::core::BoundingBox& _bbox, const StateCamera& _state, const int _id) : m_rect(_rect), m_bbox(_bbox), m_state(_state), m_id(_id)
	{
		int w = _fbo->width(), h = _fbo->height();
		m_fbo = new QOpenGLFramebufferObject(w, h, QOpenGLFramebufferObject::Depth, GL_TEXTURE_2D, GL_RGB);
		glBindTexture(GL_TEXTURE_2D, m_fbo->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);

		QOpenGLFramebufferObject::blitFramebuffer(m_fbo, _fbo);
		GL_CHECK_ERRORS();
	}

	InfosObjectImages::~InfosObjectImages()
	{
		//glDeleteTextures(1, &m_textureId);
		if(m_fbo != NULL)
			delete m_fbo;
	}

	void InfosObjectImages::set(QOpenGLFramebufferObject* _fbo, const poca::core::Vec4mf& _rect, const poca::core::BoundingBox& _bbox, const StateCamera& _state, const int _id)
	{
		int w = _fbo->width(), h = _fbo->height();
		if (m_fbo != NULL)
			delete m_fbo;
		m_fbo = new QOpenGLFramebufferObject(w, h, QOpenGLFramebufferObject::Depth, GL_TEXTURE_2D, GL_RGB);
		glBindTexture(GL_TEXTURE_2D, m_fbo->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);

		QOpenGLFramebufferObject::blitFramebuffer(m_fbo, _fbo);
		GL_CHECK_ERRORS();
		m_rect = _rect;
		m_bbox = _bbox;
		m_state = _state;
		m_id = _id;
	}

	const bool InfosObjectImages::inside(const int _x, const int _y) const
	{
		float x2 = m_rect[0] + m_rect[2], y2 = m_rect[1] + m_rect[3];
		return m_rect[0] <= _x && _x <= x2 && m_rect[1] <= _y && _y <= y2;
	}

	Camera::Camera(poca::core::MyObjectInterface* _obj, const size_t _dim, QWidget* _parent, Qt::WindowFlags _f) :QOpenGLWidget(_parent, _f), m_dimension(_dim), m_object(_obj), m_buttonOn(false), m_sizePatch(100), m_undoPossible(false), m_leftButtonOn(false), m_middleButtonOn(false), m_rightButtonOn(false), m_displayBoundingBox(true), m_nbMainGrid(4.f), m_nbIntermediateGrid(2.f), m_displayGrid(true), m_timer(NULL), m_timerCameraPath(NULL), m_alreadyInitialized(false), m_multAnimation(1.f), m_scaling(false), m_insidePatchId(-1), m_currentInteractionMode(-1), m_ROI(NULL), m_sourceFactorBlending(GL_SRC_ALPHA), m_destFactorBlending(GL_ONE_MINUS_SRC_ALPHA), m_curIndexSource(6), m_curIndexDest(7), m_activateAntialias(true), m_preventRotation(false), m_fillPolygon(true), m_resetedProj(true)
	{
		this->setObjectName("Camera");
		this->setMouseTracking(true);
		this->addActionToObserve("updateDisplay");
		this->setWindowTitle(_obj->getName().c_str());

		m_clip.resize(6);

		m_offscreenFBO = NULL;
		m_texDisplayer = NULL;
		m_stateCamera.m_rotation = glm::quat(1.f, 0, 0, 0);
		m_stateCamera.m_rotationSum = glm::quat(1.f, 0, 0, 0);

		poca::core::BoundingBox bbox = _obj->boundingBox();
		float w = bbox[3] - bbox[0], h = bbox[4] - bbox[1], t = bbox[5] - bbox[2];
		m_originalDistanceOrtho = w > h ? w / 2 : h / 2;
		m_originalDistanceOrtho = m_originalDistanceOrtho > t ? m_originalDistanceOrtho : t;

		this->setFocusPolicy(Qt::StrongFocus);
	}

	Camera::~Camera()
	{
		for (std::map <std::string, Shader*>::iterator it = m_shaders.begin(); it != m_shaders.end(); it++)
			delete it->second;
		m_shaders.clear();

		if (m_offscreenFBO != NULL)
			delete m_offscreenFBO;
		m_offscreenFBO = NULL;
	}

	void Camera::initializeGL()
	{
		if (m_alreadyInitialized) return;

		//glShadeModel(GL_SMOOTH);
		glClearColor(0.0f, 0.0f, 0.0f, 0.5f);
		glDisable(GL_COLOR_MATERIAL);

		glDepthRange(0., 1.);

		m_applyClippingPlanes = true;

		GLenum estado = glewInit();
		if (estado != GLEW_OK)
		{
			std::cerr << "Failed to initialize Glew." << std::endl;
			exit(EXIT_FAILURE);
		}

//#ifdef DEBUG
		const unsigned char* glvendor = glGetString(GL_VENDOR);
		const unsigned char* glrend = glGetString(GL_RENDERER);
		const unsigned char* glver = glGetString(GL_VERSION);
		const unsigned char* glshad = glGetString(GL_SHADING_LANGUAGE_VERSION);
		std::cout << "Vendor: " << glvendor << std::endl;
		std::cout << "Renderer: " << glrend << std::endl;
		std::cout << "OpenGL Version: " << glver << std::endl;
		std::cout << "Shader Version: " << glshad << std::endl;
		//#endif

		GLint maxx;
		glGetIntegerv(GL_POINT_SIZE_MAX, &maxx);
		std::cout << "Max GL Point Size = " << maxx << std::endl;

		m_faceGridBuffer.freeGPUMemory();
		m_faceGridBuffer.generateBuffer(18, 3, GL_FLOAT);
		m_lineGridBuffer.freeGPUMemory();
		m_lineGridBuffer.generateBuffer(6, 3, GL_FLOAT);
		m_lineBboxBuffer.freeGPUMemory();
		m_lineBboxBuffer.generateBuffer(6, 3, GL_FLOAT);

		m_quadVertBuffer.freeGPUMemory();
		std::vector <poca::core::Vec3mf> quadVertices = {
			// positions        // texture Coords
			poca::core::Vec3mf(-1.f,  1.f, 0.0f),
			poca::core::Vec3mf(-1.f, -1.f, 0.0f),
			poca::core::Vec3mf(1.f,  1.f, 0.0f),
			poca::core::Vec3mf(1.f, -1.f, 0.0f)
		};
		m_quadVertBuffer.generateBuffer(4, 512 * 512, 3, GL_FLOAT);
		m_quadVertBuffer.updateBuffer(quadVertices.data());

		m_quadUvBuffer.freeGPUMemory();
		std::vector <poca::core::Vec2mf> quadUVs = {
			// positions        // texture Coords
			poca::core::Vec2mf(0.0f, 1.0f),
			poca::core::Vec2mf(0.0f, 0.0f),
			poca::core::Vec2mf(1.0f, 1.0f),
			poca::core::Vec2mf(1.0f, 0.0f),
		};
		m_quadUvBuffer.generateBuffer(4, 512 * 512, 2, GL_FLOAT);
		m_quadUvBuffer.updateBuffer(quadUVs.data());

		m_quadVertBufferFlippedH.freeGPUMemory();
		std::vector <poca::core::Vec3mf> quadVerticesFlippedH = {
			// positions        // texture Coords
			poca::core::Vec3mf(-1.f,  -1.f, 0.0f),
			poca::core::Vec3mf(-1.f, 1.f, 0.0f),
			poca::core::Vec3mf(1.f,  -1.f, 0.0f),
			poca::core::Vec3mf(1.f, 1.f, 0.0f)
		};
		m_quadVertBufferFlippedH.generateBuffer(4, 512 * 512, 3, GL_FLOAT);
		m_quadVertBufferFlippedH.updateBuffer(quadVerticesFlippedH.data());

		m_quadUvBufferFlippedH.freeGPUMemory();
		std::vector <poca::core::Vec2mf> quadUVsFlippedH = {
			// positions        // texture Coords
			poca::core::Vec2mf(0.0f, 0.0f),
			poca::core::Vec2mf(0.0f, 1.0f),
			poca::core::Vec2mf(1.0f, 0.0f),
			poca::core::Vec2mf(1.0f, 1.0f),
		};
		m_quadUvBufferFlippedH.generateBuffer(4, 512 * 512, 2, GL_FLOAT);
		m_quadUvBufferFlippedH.updateBuffer(quadUVs.data());

		m_simplePointBuffer.freeGPUMemory();
		m_simplePointBuffer.generateBuffer(1, 512 * 512, 3, GL_FLOAT);
		std::vector <poca::core::Vec3mf> p = { poca::core::Vec3mf(0.f, 0.f, 0.f) };
		m_simplePointBuffer.updateBuffer(p);

		m_debugPointBuffer.freeGPUMemory();
		m_debugPointBuffer.generateBuffer(1, 512 * 512, 3, GL_FLOAT);
		m_debugPointBuffer.updateBuffer(p);

		m_roundedRectBuffer.freeGPUMemory();
		std::vector <poca::core::Vec2mf> roundedRectVerts = {
			// positions        // texture Coords
			poca::core::Vec2mf(0.0f, 0.0f),
			poca::core::Vec2mf(1.0f, 0.0f),
			poca::core::Vec2mf(0.0f, 1.0f),
			poca::core::Vec2mf(1.0f, 1.0f),
		};
		m_roundedRectBuffer.generateBuffer(4, 512 * 512, 2, GL_FLOAT);
		m_roundedRectBuffer.updateBuffer(roundedRectVerts.data());

		std::vector <poca::core::Vec3mf> vertsTmp = {
			poca::core::Vec3mf(0.0f,  0.0f, 0.0f),
			poca::core::Vec3mf(0.0f,  0.0f, 0.0f),
			poca::core::Vec3mf(0.0f,  0.0f, 0.0f)
		};
		m_centerSquaresBuffer.generateBuffer(3, 3, GL_FLOAT);
		m_centerSquaresBuffer.updateBuffer(vertsTmp);
		m_axisSquaresBuffer.generateBuffer(3, 3, GL_FLOAT);
		m_axisSquaresBuffer.updateBuffer(vertsTmp);

		m_cropBuffer.generateBuffer(24, 512 * 512, 3, GL_FLOAT);

		createArrowsFrame();

		resetProjection();
		reset();
		m_ssaoShader.init(this->width(), this->height());

		m_alreadyInitialized = true;
	}

	void Camera::renderRoundedBoxShadow(const float _xmin, const float _ymin, const float _xmax, const float _ymax, const float _r, const float _g, const float _b, const float _a, const float _sigma, const float _corner)
	{
		Shader* shader = getShader("roundedRectShader");
		shader->use();
		shader->setVec4("box", _xmin, _ymin, _xmax, _ymax)	;
		shader->setVec4("color", _r, _g, _b, _a);
		shader->setFloat("sigma", _sigma);
		shader->setFloat("corner", _corner);
		shader->setVec2("window", (float)width(), (float)height());
		glEnableVertexAttribArray(0);
		m_roundedRectBuffer.bindBuffer(0, 0);
		glDrawArrays(m_roundedRectBuffer.getMode(), 0, m_roundedRectBuffer.getSizeBuffers()[0]);
		glDisableVertexAttribArray(0);
		shader->release();
	}

	void Camera::drawOffscreen()
	{
		bool success = m_offscreenFBO->bind();
		if (!success) std::cout << "Problem with binding" << std::endl;
		glViewport(m_viewport[0], m_viewport[1], m_viewport[2], m_viewport[3]);
		bool ok;
		uint32_t thickness = 5;
		float antialias = 1.f;

		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(m_object);
		if (!comObj) return;

		if (comObj->hasParameter("colorBakground")) {
			std::array <unsigned char, 4> rgba = comObj->getParameter< std::array <unsigned char, 4>>("colorBakground");
			glClearColor(rgba[0], rgba[1], rgba[2], rgba[3]);
		}
		else
			glClearColor(0.f, 0.f, 0.f, 1.f);

		if (comObj->hasParameter("lineWidthGL"))
			thickness = comObj->getParameter<uint32_t>("lineWidthGL");

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (m_resetedProj)
			m_applyClippingPlanes = false;
		else
			m_applyClippingPlanes = true;
		GLfloat bkColor[4];
		glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
		poca::core::Color4D color = poca::core::contrastColor(poca::core::Color4D(bkColor[0] * 255.f, bkColor[1] * 255.f, bkColor[2] * 255.f, bkColor[3] * 255.f));
		color *= 255.f;
		m_matrixModel = glm::translate(glm::mat4(1.f), m_stateCamera.m_translationModel);
		m_stateCamera.m_matrixView = m_stateCamera.m_matrix;
		glDisable(GL_DEPTH_TEST);
		if (m_displayGrid)
			displayGrid();
		if (m_displayBoundingBox)
			displayBoundingBox(thickness, antialias);
		glEnable(GL_DEPTH_TEST);
		m_object->executeGlobalCommand(&poca::core::CommandInfo(false, "display", "camera", this, "offscreen", true));

		success = m_offscreenFBO->release();
		if (!success) std::cout << "Problem with releasing" << std::endl;

		glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebufferObject());

		/*QImage fboImage(m_offscreenFBO->toImage(false));
		QImage image(fboImage.constBits(), fboImage.width(), fboImage.height(), QImage::Format_RGB32);
		bool res = image.save(QString("e:/poca_%1.png").arg(cptAnimation));
		if (!res)
			std::cout << "Problem with saving" << std::endl;*/
	}

	void Camera::paintGL()
	{
		drawElements();
	}

	void Camera::drawElements(QOpenGLFramebufferObject * _buffOffscreen)
	{
		GL_CHECK_ERRORS();
		recomputeFrame(m_currentCrop);
		clock_t beginFrame = clock();

		GL_CHECK_ERRORS();
		if (_buffOffscreen != NULL) {
			bool success = _buffOffscreen->bind();
			if (!success) std::cout << "Problem with binding" << std::endl;
		}

		GL_CHECK_ERRORS();
		glViewport(m_viewport[0], m_viewport[1], m_viewport[2], m_viewport[3]);
		glDepthRange(0., 1.0);

		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(m_object);
		if (!comObj) return;

		//Get all parameters saved by object commands
		std::array <uint8_t, 4> colorBack{ 0, 0, 0, 255 }, colorFont{ 255, 255, 255, 255 };
		std::array <float, 4> colorSelectedROIs{ 1.f, 0.f, 1.f, 1.f }, colorUnselectedROIs{ 1.f, 0.f, 0.f, 1.f };
		uint32_t thickness = 5, pointSizeGL = 1;
		bool displayFont = true;
		float fontSize = 20.f, antialias = 1.f;
		GL_CHECK_ERRORS();

		if (comObj->hasParameter("colorBakground")) {
			std::array <unsigned char, 4> rgba = comObj->getParameter< std::array <unsigned char, 4>>("colorBakground");
			colorBack = { rgba[0], rgba[1], rgba[2], rgba[3] };
		}

		if (comObj->hasParameter("lineWidthGL"))
			thickness = comObj->getParameter<uint32_t>("lineWidthGL");
		if (comObj->hasParameter("antialias"))
			m_activateAntialias = comObj->getParameter<bool>("antialias");
		if (comObj->hasParameter("fontDisplay"))
			displayFont = comObj->getParameter<bool>("fontDisplay");
		if (comObj->hasParameter("fontSize"))
			fontSize = comObj->getParameter<float>("fontSize");
		if (comObj->hasParameter("cullFace"))
			m_cullFace = comObj->getParameter<bool>("cullFace");
		if (comObj->hasParameter("fillPolygon"))
			m_fillPolygon = comObj->getParameter<bool>("fillPolygon");
		if (comObj->hasParameter("clip"))
			m_applyClippingPlanes = comObj->getParameter<bool>("clip");
		if (comObj->hasParameter("pointSizeGL"))
			pointSizeGL = comObj->getParameter<uint32_t>("pointSizeGL");

		GL_CHECK_ERRORS();

		glClearColor((float)colorBack[0] / 255.f, (float)colorBack[1] / 255.f, (float)colorBack[2] / 255.f, (float)colorBack[3] / 255.f);
		double a = 1 - (0.299 * colorBack[0] + 0.587 * colorBack[1] + 0.114 * colorBack[2]) / 255;
		if (a < 0.5)
			colorFont = { 0, 0, 0, 255 }; // bright colors - black font
		else
			colorFont = { 255, 255, 255, 255 }; // dark colors - white font

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		GL_CHECK_ERRORS();

		GLfloat bkColor[4];
		glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
		poca::core::Color4D color = poca::core::contrastColor(poca::core::Color4D(bkColor[0] * 255.f, bkColor[1] * 255.f, bkColor[2] * 255.f, bkColor[3] * 255.f));
		color *= 255.f;
		GL_CHECK_ERRORS();

		m_matrixModel = glm::translate(glm::mat4(1.f), m_stateCamera.m_translationModel);
		m_stateCamera.m_matrixView = m_stateCamera.m_matrix;
		GL_CHECK_ERRORS();

		glPointSize(pointSizeGL);
		GL_CHECK_ERRORS();
		//glLineWidth(thickness);
		GL_CHECK_ERRORS();
		glDisable(GL_DEPTH_TEST);
		GL_CHECK_ERRORS();
		displayGrid();
		GL_CHECK_ERRORS();
		displayBoundingBox(thickness, antialias);
		GL_CHECK_ERRORS();

		glEnable(GL_DEPTH_TEST);
		glLineWidth(thickness);

		bool ssao = false;
		if (comObj->hasParameter("useSSAO"))
			ssao = comObj->getParameter<bool>("useSSAO");
		GL_CHECK_ERRORS();

		if (!ssao) {
			if(_buffOffscreen == NULL)
				m_object->executeGlobalCommand(&poca::core::CommandInfo(false, "display", "camera", this));
			else
				m_object->executeGlobalCommand(&poca::core::CommandInfo(false, "display", "camera", this, "offscreen", true));
		}
		else {
			drawSSAO(_buffOffscreen);
		}

		GL_CHECK_ERRORS();

		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CLIP_DISTANCE0);
		glDisable(GL_CLIP_DISTANCE1);
		//glDisable(GL_CLIP_DISTANCE2);
		glDisable(GL_CLIP_DISTANCE3);
		glDisable(GL_CLIP_DISTANCE4);
		//glDisable(GL_CLIP_DISTANCE5);

		bool displayROIs = true;

		if (comObj->hasParameter("displayROIs"))
			displayROIs = comObj->getParameter<bool>("displayROIs");
		if (comObj->hasParameter("colorSelectedROIs")) {
			std::array <unsigned char, 4> rgba = comObj->getParameter< std::array <unsigned char, 4>>("colorSelectedROIs");
			colorSelectedROIs = { (float)rgba[0] / 255.f, (float)rgba[1] / 255.f, (float)rgba[2] / 255.f, (float)rgba[3] / 255.f };
		}
		if (comObj->hasParameter("colorUnselectedROIs")) {
			std::array <unsigned char, 4> rgba = comObj->getParameter< std::array <unsigned char, 4>>("colorUnselectedROIs");
			colorUnselectedROIs = { (float)rgba[0] / 255.f, (float)rgba[1] / 255.f, (float)rgba[2] / 255.f, (float)rgba[3] / 255.f };
		}

		if (displayROIs) {
			for (poca::core::ROIInterface* roi : m_object->getROIs())
				roi->draw(this, roi->selected() ? colorSelectedROIs : colorUnselectedROIs, thickness);
		}
		if (m_ROI != NULL)
			m_ROI->draw(this, colorSelectedROIs, thickness);

		GL_CHECK_ERRORS();

		glDisable(GL_DEPTH_TEST);

		poca::opengl::Shader* shader = this->getShader("uniformColorShader");
		float w = width(), h = height();
		glm::mat4 projText = glm::ortho(0.f, w, 0.f, h);
		shader->use();
		shader->setMat4("MVP", projText);
		glEnableVertexAttribArray(0);
		if (m_debugPointBuffer.getNbElements() != 0) {
			shader->setVec4("singleColor", 1.f, 0.f, 1.f, 1.f);
			m_debugPointBuffer.bindBuffer(0, 0);
			glDrawArrays(m_debugPointBuffer.getMode(), 0, m_debugPointBuffer.getSizeBuffers()[0]);
		}
		glDisable(GL_POINT_SPRITE);
		glDisable(GL_PROGRAM_POINT_SIZE);
		glPointSize(pointSizeGL);
		glEnable(GL_POINT_SMOOTH);
		shader->setVec4("singleColor", 1.f, 0.f, 0.f, 1.f);
		glDisable(GL_POINT_SPRITE);
		glDisable(GL_PROGRAM_POINT_SIZE);
		if (m_currentInteractionMode == Crop && m_buttonOn) {
			m_cropBuffer.bindBuffer(0, 0);
			glDrawArrays(m_cropBuffer.getMode(), 0, m_cropBuffer.getSizeBuffers()[0]);
		}
		glDisableVertexAttribArray(0);
		shader->release();

		GL_CHECK_ERRORS();

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDisable(GL_CULL_FACE);

		if (!m_infoObjects.empty()) {
			glClearColor((float)colorFont[0] / 255.f, (float)colorFont[1] / 255.f, (float)colorFont[2] / 255.f, (float)colorFont[3] / 255.f);
			glEnable(GL_SCISSOR_TEST);
			int w = this->width(), h = this->getHeight();
			if (w > 2 * m_sizePatch && h > m_sizePatch) {
				int x = w - 120, y = 0;
				bool inCamera = true;
				for (size_t n = 0; n < m_infoObjects.size() && inCamera; n++) {
					y = m_infoObjects[n]->m_rect.y();
					m_infoObjects[n]->m_rect.setX(x);
					glScissor(x - 2, h - y - m_sizePatch - 2, m_sizePatch + 4, m_sizePatch + 4);
					glClear(GL_COLOR_BUFFER_BIT);
					glViewport(x, h - y - m_sizePatch, m_sizePatch, m_sizePatch);
					drawTexture(m_infoObjects[n]->m_fbo->texture(), true);
					y += m_sizePatch + 20;
					inCamera = (y + m_sizePatch) < h;
				}
			}
			glDisable(GL_SCISSOR_TEST);
		}

		glClearColor((float)colorBack[0] / 255.f, (float)colorBack[1] / 255.f, (float)colorBack[2] / 255.f, (float)colorBack[3] / 255.f);

		glViewport(m_viewportThumbnailFrame[0], m_viewportThumbnailFrame[1], m_viewportThumbnailFrame[2], m_viewportThumbnailFrame[3]);
		glClear(GL_DEPTH_BUFFER_BIT);
		auto clippingSave = m_applyClippingPlanes;
		m_applyClippingPlanes = false;
		displayArrowsFrame();
		m_applyClippingPlanes = clippingSave;

		glViewport(m_viewport[0], m_viewport[1], m_viewport[2], m_viewport[3]);
		projText = glm::ortho(0.f, w, h, 0.f);
		glEnable(GL_BLEND);

		float lineHeight = 0.0f;
		float x = 15.f;
		float y = 10.f;
		GL_CHECK_ERRORS();

		if (m_texDisplayer == NULL)
			m_texDisplayer = new TextDisplayer();

		m_texDisplayer->setFontSize(fontSize);

		float boxSigma = 0.25f, shadowSigma = 2.f, corner = 25.f;
		for (QString info : m_infoPicking) {
			QStringList list = info.split("\n");
			float yRectOrig = y - 5.f, xRect = 0, yRect = yRectOrig;
			//Determine bounding rectangles for texts
			for (QString text : list) {
				float xTmp = m_texDisplayer->widthOfStr(text.toLatin1().data(), x, y);
				float yTmp = m_texDisplayer->lineHeight();
				xRect = xTmp > xRect ? xTmp : xRect;
				yRect += yTmp;
			}
			//Draw bounding rectangle
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			renderRoundedBoxShadow(0.f, this->height() - (yRect + 10.f), xRect + 35.f, this->height() - yRectOrig, 1, 1, 0, 0.75, boxSigma, corner);

			glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
			//Draw text
			for (QString text : list) {
				float xTmp = m_texDisplayer->widthOfStr(text.toLatin1().data(), x, y);
				float yTmp = m_texDisplayer->lineHeight();
				poca::core::Vec2mf dxy = m_texDisplayer->renderText(projText, text.toLatin1().data(), 0, 0, 0, 255, x, y, 0);
				y += dxy[1];
			}
			y += 15.f;
		}
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		if (displayFont)
			for (const std::pair<poca::core::Vec3mf, std::string>& frameTxt : m_frameTexts) {
				m_texDisplayer->renderText(projText, frameTxt.second.c_str(), colorFont[0], colorFont[1], colorFont[2], colorFont[3], frameTxt.first[0], frameTxt.first[1], 0, FONS_ALIGN_CENTER | FONS_ALIGN_MIDDLE);
			}
		for (const std::pair<poca::core::Vec3mf, std::string>& frameTxt : m_frameTextsThumbnail) {
			m_texDisplayer->renderText(projText, frameTxt.second.c_str(), colorFont[0], colorFont[1], colorFont[2], colorFont[3], frameTxt.first[0], frameTxt.first[1], 0, FONS_ALIGN_CENTER | FONS_ALIGN_MIDDLE);
		}
		GL_CHECK_ERRORS();
		glDisable(GL_BLEND);

		if (_buffOffscreen != NULL) {
			bool success = _buffOffscreen->release();
			if (!success) std::cout << "Problem with releasing" << std::endl;

			QImage fboImage(m_offscreenFBO->toImage(true));
			QImage image(fboImage.constBits(), fboImage.width(), fboImage.height(), QImage::Format_RGB32);
			m_movieFrames.push_back(image.copy(0, 0, image.width(), image.height()));
		}

		glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebufferObject());
		GL_CHECK_ERRORS();
}

	void Camera::drawSSAO(QOpenGLFramebufferObject* _buffOffscreen) {
		const SsaoShader& classShaderSSAO = m_ssaoShader;
		const Shader& shaderSSAO = classShaderSSAO.getShaderSSAO();
		const Shader& shaderSSAOBlur = classShaderSSAO.getShaderSSAOBlur();
		const Shader& shaderLightingPass = classShaderSSAO.getShaderSSAOLighting();

		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(m_object);
		if (!comObj) return;

		float w = size().width(), h =size().height();
		//Get all parameters saved by object commands
		std::array <uint8_t, 4> colorBack{ 0, 0, 0, 255 };
		bool useSilhouette = false, useDebug = false;
		float radiusSSAO = 5.f, strengthSSAO = 1.f;
		SsaoShader::SSAODebugDisplay currentDebug = SsaoShader::SSAODebugDisplay::SSAO_NO_DEBUG;
		if (comObj->hasParameter("colorBakground")) {
			std::array <unsigned char, 4> rgba = comObj->getParameter< std::array <unsigned char, 4>>("colorBakground");
			colorBack = { rgba[0], rgba[1], rgba[2], rgba[3] };
		}
		if (comObj->hasParameter("radiusSSAO"))
			radiusSSAO = comObj->getParameter<float>("radiusSSAO");
		if (comObj->hasParameter("strengthSSAO"))
			strengthSSAO = comObj->getParameter<float>("strengthSSAO");
		if (comObj->hasParameter("useSilhouetteSSAO"))
			useSilhouette = comObj->getParameter<bool>("useSilhouetteSSAO");
		if (comObj->hasParameter("useDebugSSAO"))
			useDebug = comObj->getParameter<bool>("useDebugSSAO");
		if (comObj->hasParameter("currentDebugSSAO"))
			currentDebug = static_cast<SsaoShader::SSAODebugDisplay>(comObj->getParameter<int>("currentDebugSSAO"));

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glDisable(GL_BLEND);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDisable(GL_CULL_FACE);

		glClearColor((float)colorBack[0] / 255.f, (float)colorBack[1] / 255.f, (float)colorBack[2] / 255.f, (float)colorBack[3] / 255.f);

		// First of all, save the already bound framebuffer
		GLint qt_buffer;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &qt_buffer);

		bool success = classShaderSSAO.m_fboGeometry->bind();
		if (!success) std::cout << "Problem with binding" << std::endl;
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		uint32_t thickness = 5;
		if (comObj->hasParameter("lineWidthGL"))
			thickness = comObj->getParameter<uint32_t>("lineWidthGL");
		m_applyClippingPlanes = true;
		glDisable(GL_DEPTH_TEST);
		displayGrid();
		displayBoundingBox(thickness, 1);

		static const float transparent[] = { 0, 0, 0, 0 };
		for(uint32_t i = 1; i < 5; i++)
			glClearBufferfv(GL_COLOR, i, transparent);

		m_object->executeCommandOnSpecificComponent("ObjectList", &poca::core::CommandInfo(false, "display", "camera", this, "offscreen", true, "ssao", true));

		m_object->executeCommandOnSpecificComponent("DetectionSet", &poca::core::CommandInfo(false, "display", "camera", this, "ssao", true));
		success = classShaderSSAO.m_fboGeometry->release();
		if (!success) std::cout << "Problem with releasing" << std::endl;
		QVector<GLuint> texIds = classShaderSSAO.m_fboGeometry->textures();

		glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

		success = classShaderSSAO.m_fboSilhouette->bind();
		if (!success) std::cout << "Problem with binding" << std::endl;
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		classShaderSSAO.m_shaderSilhouette->use();
		classShaderSSAO.m_shaderSilhouette->setVec4("backColor", (float)colorBack[0] / 255.f, (float)colorBack[1] / 255.f, (float)colorBack[2] / 255.f, (float)colorBack[3] / 255.f);
		for (unsigned int i = 0; i < 64; ++i)
			classShaderSSAO.m_shaderSilhouette->setVec2("directions[" + std::to_string(i) + "]", classShaderSSAO.m_ssaoCircle[i]);
		classShaderSSAO.m_shaderSilhouette->setVec2("screenDimension", w, h);
		classShaderSSAO.m_shaderSilhouette->setFloat("radius", 10.f);
		classShaderSSAO.m_shaderSilhouette->setInt("ssaoInput", 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texIds[3]);
		renderQuad();
		classShaderSSAO.m_shaderSilhouette->release();

		success = classShaderSSAO.m_fboNoise->bind();
		glClear(GL_COLOR_BUFFER_BIT);
		classShaderSSAO.m_shaderSSAO->use();
		// Send kernel + rotation 
		for (unsigned int i = 0; i < 64; ++i)
			classShaderSSAO.m_shaderSSAO->setVec3("samples[" + std::to_string(i) + "]", classShaderSSAO.m_ssaoKernel[i]);
		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		classShaderSSAO.m_shaderSSAO->setMat4("projection", getProjectionMatrix());
		classShaderSSAO.m_shaderSSAO->setMat4("MVP", proj * view * model);
		classShaderSSAO.m_shaderSSAO->setVec2("screenDimension", w, h);
		classShaderSSAO.m_shaderSSAO->setFloat("radius", radiusSSAO);
		classShaderSSAO.m_shaderSSAO->setFloat("strength", strengthSSAO);
		classShaderSSAO.m_shaderSSAO->setInt("gPosition", 0);
		classShaderSSAO.m_shaderSSAO->setInt("gNormal", 1);
		classShaderSSAO.m_shaderSSAO->setInt("texNoise", 2);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texIds[0]);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, texIds[1]);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, classShaderSSAO.m_noiseTexture);
		renderQuad();
		success = classShaderSSAO.m_fboNoise->release();
		if (!success) std::cout << "Problem with releasing" << std::endl;

		success = classShaderSSAO.m_fboBlur->bind();
		if (!success) std::cout << "Problem with binding" << std::endl;
		glClear(GL_COLOR_BUFFER_BIT);
		classShaderSSAO.m_shaderSSAOBlur->use();
		classShaderSSAO.m_shaderSSAOBlur->setInt("ssaoInput", 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, classShaderSSAO.m_fboNoise->texture());
		renderQuad();
		success = classShaderSSAO.m_fboBlur->release();
		if (!success) std::cout << "Problem with releasing" << std::endl;

		success = classShaderSSAO.m_fboSilhouetteBlurred->bind();
		if (!success) std::cout << "Problem with binding" << std::endl;
		glClear(GL_COLOR_BUFFER_BIT);
		classShaderSSAO.m_shaderSSAOBlur->use();
		classShaderSSAO.m_shaderSSAOBlur->setInt("ssaoInput", 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, classShaderSSAO.m_fboSilhouette->texture());
		renderQuad();
		success = classShaderSSAO.m_fboSilhouetteBlurred->release();
		if (!success) std::cout << "Problem with releasing" << std::endl;

		// 4. lighting pass: traditional deferred Blinn-Phong lighting with added screen-space ambient occlusion
		// -----------------------------------------------------------------------------------------------------
		glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
		if (_buffOffscreen != NULL) {
			success = _buffOffscreen->bind();//classShaderSSAO.m_fboLighting->bind();
			if (!success) std::cout << "Problem with binding" << std::endl;
		}
		classShaderSSAO.m_shaderLightingPass->use();
		// send light relevant uniforms
		classShaderSSAO.m_shaderLightingPass->setVec3("light.Position", classShaderSSAO.m_lightPos);
		classShaderSSAO.m_shaderLightingPass->setVec3("light.Color", classShaderSSAO.m_lightColor);
		// Update attenuation parameters
		const float linear = 0.09;
		const float quadratic = 0.032;
		classShaderSSAO.m_shaderLightingPass->setMat4("projection", getProjectionMatrix());
		classShaderSSAO.m_shaderLightingPass->setFloat("light.Linear", linear);
		classShaderSSAO.m_shaderLightingPass->setFloat("light.Quadratic", quadratic);
		classShaderSSAO.m_shaderLightingPass->setBool("useSilhouette", useSilhouette);
		classShaderSSAO.m_shaderLightingPass->setBool("debug", useDebug);
		classShaderSSAO.m_shaderLightingPass->setVec4("backColor", (float)colorBack[0] / 255.f, (float)colorBack[1] / 255.f, (float)colorBack[2] / 255.f, (float)colorBack[3] / 255.f);
		classShaderSSAO.m_shaderLightingPass->setFloat("radius", radiusSSAO);
		classShaderSSAO.m_shaderLightingPass->setInt("gPosition", 0);
		classShaderSSAO.m_shaderLightingPass->setInt("gNormal", 1);
		classShaderSSAO.m_shaderLightingPass->setInt("gAlbedo", 2);
		classShaderSSAO.m_shaderLightingPass->setInt("ssao", 3);
		classShaderSSAO.m_shaderLightingPass->setInt("silhouette", 4);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texIds[0]);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, texIds[1]);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, texIds[2]);
		glActiveTexture(GL_TEXTURE3);
		switch (currentDebug) {
		case SsaoShader::SSAODebugDisplay::SSAO_NO_DEBUG:
		{
			glBindTexture(GL_TEXTURE_2D, classShaderSSAO.m_fboBlur->texture());
			break;
		}
		case SsaoShader::SSAODebugDisplay::SSAO_POS:
		{
			glBindTexture(GL_TEXTURE_2D, texIds[0]);
			break;
		}
		case SsaoShader::SSAODebugDisplay::SSAO_NORMAL:
		{
			glBindTexture(GL_TEXTURE_2D, texIds[1]);
			break;
		}
		case SsaoShader::SSAODebugDisplay::SSAO_COLOR:
		{
			glBindTexture(GL_TEXTURE_2D, texIds[2]);
			break;
		}
		case SsaoShader::SSAODebugDisplay::SSAO_SSAO_MAP:
		{
			glBindTexture(GL_TEXTURE_2D, classShaderSSAO.m_fboBlur->texture());
			break;
		}
		}
		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, classShaderSSAO.m_fboSilhouetteBlurred->texture());
		renderQuad();
		classShaderSSAO.m_shaderLightingPass->release();
	}

	// renderQuad() renders a 1x1 XY quad in NDC
// -----------------------------------------
	void Camera::renderQuad(const bool _flipped) const
	{
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		if (!_flipped){
			m_quadVertBuffer.bindBuffer(0, 0);
			m_quadUvBuffer.bindBuffer(0, 1);
		}
		else {
			m_quadVertBufferFlippedH.bindBuffer(0, 0);
			m_quadUvBufferFlippedH.bindBuffer(0, 1);
		}
		glDrawArrays(GL_TRIANGLE_STRIP, 0, m_quadVertBuffer.getSizeBuffers()[0]); // Starting from vertex 0; 3 vertices total -> 1 triangle
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
	}

	void Camera::drawTexture(const GLuint _textureIDImage, const bool _flipped)
	{
		poca::opengl::Shader* shader = getShader("textureShader");
		shader->use();
		shader->setInt("image", 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, _textureIDImage);
		renderQuad(_flipped);
		shader->release();
	}

	void Camera::drawTextureFBO(const GLuint _textureIDOffscreen, const GLuint _textureIDLUT)
	{
		poca::opengl::Shader* shader = getShader("textureShaderFBO");
		shader->use();
		shader->setInt("offscreen", 0);
		shader->setInt("lut", 1);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, _textureIDOffscreen);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_1D, _textureIDLUT);
		renderQuad();
		shader->release(); 
	}

	void Camera::displayBoundingBox(const float _thickness, const float _antialias)
	{
		if (!m_displayBoundingBox) return;
		GLfloat bkColor[4];
		glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
		for (size_t n = 0; n < 4; n++)
			bkColor[n] *= 255.f;
		poca::core::Color4D colorFront = poca::core::contrastColor(poca::core::Color4D(bkColor[0], bkColor[1], bkColor[2], bkColor[3]));
		
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		poca::opengl::Shader* shader = getShader("line2DShader");
		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setUVec4("viewport", getViewport());
		shader->setVec2("resolution", width(), height());
		shader->setFloat("thickness", (_thickness + 1) * 2.f);
		shader->setFloat("antialias", _antialias);
		shader->setVec4("singleColor", colorFront[0], colorFront[1], colorFront[2], colorFront[3]);
		shader->setBool("useSingleColor", true);
		shader->setBool("activatedCulling", false);
		glEnableVertexAttribArray(0);
		m_lineBboxBuffer.bindBuffer(0);
		glDrawArrays(m_lineBboxBuffer.getMode(), 0, m_lineBboxBuffer.getNbElements());
		glDisableVertexAttribArray(0);
		shader->release();
	}

	void Camera::displayGrid()
	{
		if (m_faceGridBuffer.empty() || !m_displayGrid || m_lineGridBuffer.getNbElements() == 0) return;

		GL_CHECK_ERRORS();
		GLfloat bkColor[4];
		glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
		for (size_t n = 0; n < 4; n++)
			bkColor[n] *= 255.f;
		double a = 1 - (0.299 * bkColor[0] + 0.587 * bkColor[1] + 0.114 * bkColor[2]) / 255;
		poca::core::Color4D	colorGrid;
		if (a < 0.5)
			colorGrid.set(0.5, 0.5, 0.5, 1); // bright colors - black font
		else
			colorGrid.set(0.2, 0.2, 0.2, 1); // dark colors - white font
		GL_CHECK_ERRORS();
		poca::opengl::Shader* shader = getShader("uniformColorShader");
		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setVec4("singleColor", colorGrid[0], colorGrid[1], colorGrid[2], colorGrid[3]);
		shader->setVec4v("clipPlanes", m_clip);
		shader->setInt("nbClipPlanes", nbClippingPlanes());
		shader->setBool("clip", m_applyClippingPlanes);
		GL_CHECK_ERRORS();

		float valueColor = a < 0.5 ? 242.f : 45.f;
		shader->setVec4("singleColor", valueColor / 255.f, valueColor / 255.f, valueColor / 255.f, 1.f);
		glEnableVertexAttribArray(0);
		m_faceGridBuffer.bindBuffer(0);
		glDrawArrays(m_faceGridBuffer.getMode(), 0, m_faceGridBuffer.getNbElements()); // Starting from vertex 0; 3 vertices total -> 1 triangle
		glDisableVertexAttribArray(0);
		GL_CHECK_ERRORS();

		valueColor = a < 0.5 ? 210.f : 75.f;
		shader->setVec4("singleColor", valueColor / 255.f, valueColor / 255.f, valueColor / 255.f, 1.f);
		glEnableVertexAttribArray(0);
		m_lineGridBuffer.bindBuffer(0);
		glDrawArrays(m_lineGridBuffer.getMode(), 0, m_lineGridBuffer.getNbElements()); // Starting from vertex 0; 3 vertices total -> 1 triangle
		glDisableVertexAttribArray(0);
		GL_CHECK_ERRORS();

		shader->release();
	}

	void Camera::recalcModelView(void)
	{
		m_viewport = glm::uvec4(0, 0, this->width(), this->height());
		unsigned int smallestDim = this->width() < this->height() ? this->width() : this->height();

		float factorW = 1.f, factorH = 1.f;
		float diffX = this->width(), diffY = this->height();
		if (diffX > diffY)
			factorW = diffX / diffY;
		else
			factorH = diffY / diffX;

		float projLeft = -m_distanceOrtho * factorW;
		float projRight = m_distanceOrtho * factorW;
		float projBottom = -m_distanceOrtho * factorH;
		float projUp = m_distanceOrtho * factorH;
		float projNear = -m_originalDistanceOrtho * sqrt(3);
		float projFar = m_originalDistanceOrtho * sqrt(3);

		m_matrixProjection = glm::ortho(projLeft, projRight, projBottom, projUp, projNear, projFar);

		m_viewportThumbnailFrame = glm::uvec4(0, 0, smallestDim / 10, smallestDim / 10);
		m_matrixProjectionThumbnailFrame = glm::ortho(-0.5f, 0.5f, -0.5f, 0.5f, -5.f, 5.f);
	}

	void Camera::resizeWindow(const int _x, const int _y, const int _w, const int _h)
	{
	}

	std::array<int, 2> Camera::sizeHintInterface() const
	{
		std::array<int, 2> size = { (int)m_object->getWidth() + 8, (int)m_object->getHeight() + 34 };
		return size;
	}

	QSize Camera::sizeHint() const
	{
		return QSize((int)m_object->getWidth() + 8, (int)m_object->getHeight() + 34);
	}

	void Camera::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo& _aspect)
	{
		poca::core::SubjectInterface* si = dynamic_cast <poca::core::SubjectInterface*>(m_object);
		if (_subject == si && _aspect == "updateInfosObject") {
			poca::core::CommandInfo ci(false, "getInfoObjectCurrentlyPicked");
			m_object->executeCommandOnSpecificComponent("ObjectList", &ci);
			std::string infos;
			if (ci.hasParameter("infos"))
				infos = ci.getParameter<std::string>("infos");
			if (m_infoPicking.empty())
				m_infoPicking.push_back(QString::fromStdString(infos));
			else
				for (QString& text : m_infoPicking) {
					if (text.startsWith("Object id"))
						text = QString::fromStdString(infos);
				}
			update();
		}
		else if (_subject == si && _aspect == "updateInfosObjectOverlap") {
			poca::core::CommandInfo ci(false, "getInfoObjectCurrentlyPicked");
			m_object->executeCommandOnSpecificComponent("ObjectColocalization", &ci);
			std::string infos;
			if (ci.hasParameter("infos"))
				infos = ci.getParameter<std::string>("infos");
			if (m_infoPicking.empty())
				m_infoPicking.push_back(QString::fromStdString(infos));
			else
				for (QString& text : m_infoPicking) {
					if (text.startsWith("Object id"))
						text = QString::fromStdString(infos);
				}
			update();
		}
		if (_subject == si && this->hasActionToObserve(_aspect))
			update();
	}

	void Camera::keyPressEvent(QKeyEvent* _event)
	{
		if (_event->key() == Qt::Key_Z && _event->modifiers() & Qt::ControlModifier && m_undoPossible) {
			m_matrixModel = m_matrixModelSaved;
			m_stateCamera.m_matrix = m_matrixViewSaved;
			m_distanceOrtho = m_distanceOrthoSaved;
			m_stateCamera.m_rotationSum = glm::quat(1.f, 0, 0, 0);
			recalcModelView();
			m_undoPossible = false;
			update();
		}
		else if(_event->key() == Qt::Key_A){
			if (m_timer == NULL) {
				m_timer = new QTimer(this);
				connect(m_timer, SIGNAL(timeout()), this, SLOT(animateRotation()));
			}
			if (m_timer->isActive())
				m_timer->stop();
			else {
				cptAnimation = 0;
				m_timer->start(10);
			}
		}
		else if (_event->key() == Qt::Key_S) {
			QString filename(m_object->getDir().c_str() + QString("/tmp.svg"));
			filename = QFileDialog::getSaveFileName(NULL, QObject::tr("Save svg..."), filename, QObject::tr("SVG files (*.svg)"), 0, QFileDialog::DontUseNativeDialog);
			m_object->executeGlobalCommand(&poca::core::CommandInfo(true, "saveAsSVG", "filename", filename.toStdString()));

		}
		else if (_event->key() == Qt::Key_W) {
			toggleSrcBlendingFactor();
		}
		else if (_event->key() == Qt::Key_X) {
			toggleDstBlendingFactor();
		}
		else if (_event->key() == Qt::Key_P) {
			m_activateAntialias = !m_activateAntialias;
			repaint();
		}
		else if (_event->key() == Qt::Key_C) {
			for (std::map <std::string, Shader*>::iterator it = m_shaders.begin(); it != m_shaders.end(); it++) {
				it->second->destroy();
				delete it->second;
			}
			m_shaders.clear();
			m_ssaoShader.loadShaders();
			repaint();
		}
		else if (_event->key() == Qt::Key_M) {
			m_object->clearROIs();
			repaint();
		}
		else if (_event->key() == Qt::Key_L) {
			m_object->executeCommandOnSpecificComponent("DetectionSet", &poca::core::CommandInfo(true, "selectLocsInROIs"));
			repaint();
		}
		else if (_event->key() == Qt::Key_N) {
			bool ssao = false;
			poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(m_object);
			if (!comObj) return; 
			if (comObj->hasParameter("useSSAO"))
				ssao = comObj->getParameter<bool>("useSSAO");
			m_object->executeCommand(&poca::core::CommandInfo(true, "useSSAO", !ssao));
			repaint();
		}
		else
			QOpenGLWidget::keyPressEvent(_event);
	}

	void Camera::mousePressEvent(QMouseEvent* _event)
	{
		emit(clickInsideWindow());
		makeCurrent();
		setClickPoint(_event->pos().x(), _event->pos().y());
		m_cropPointBegin.set(_event->pos().x(), this->height() - _event->pos().y(), 0.f);
		m_cropPointEnd = m_cropPointBegin;
		poca::core::BoundingBox cropBox(FLT_MAX, FLT_MAX, 0.f, -FLT_MAX, -FLT_MAX, 0.f);
		cropBox.addPointBBox(m_cropPointBegin.x(), m_cropPointBegin.y(), 0.f);
		cropBox.addPointBBox(m_cropPointEnd.x(), m_cropPointEnd.y(), 0.f);
		std::vector <poca::core::Vec3mf> cube(24);
		poca::geometry::createCubeFromVector(cube, cropBox);
		m_cropBuffer.updateBuffer(cube.data());
		float x, y, z = 0.f;
		if (m_currentInteractionMode != poca::opengl::Camera::None) {
			if (m_currentInteractionMode == poca::opengl::Camera::Sphere3DRoiDefinition) {
				if (m_pickedPoints.empty()) return;
				std::vector <uint32_t> indices(m_pickedPoints.size());
				std::iota(std::begin(indices), std::end(indices), 0);
				//Compute a vector of distances of the points to the camera position
				std::vector <float> distances(m_pickedPoints.size());
				for (size_t n = 0; n < m_pickedPoints.size(); n++)
					distances[n] = glm::dot(glm::vec3(m_pickedPoints[n].x(), m_pickedPoints[n].y(), m_pickedPoints[n].z()) - m_stateCamera.m_center, m_stateCamera.m_eye);
				//Sort wrt the distance to the camera position
				std::sort(indices.begin(), indices.end(),
					[&](int A, int B) -> bool {
						return distances[A] < distances[B];
					});
				uint32_t id = indices.front();
				x = m_pickedPoints[id].x();
				y = m_pickedPoints[id].y();
				z = m_pickedPoints[id].z();
				if (m_ROI == NULL)
					m_ROI = poca::core::getROIFromType(m_currentInteractionMode);
				std::cout << "[" << x << ", " << y << ", " << z << "]" << std::endl;
			}
			else if (m_currentInteractionMode == poca::opengl::Camera::PlaneRoiDefinition || m_currentInteractionMode == poca::opengl::Camera::PolyPlaneRoiDefinition) {
				glm::vec3 coords = getWorldCoordinates(glm::vec2(_event->pos().x(), this->height() - _event->pos().y()));
				if (m_ROI == NULL)
					m_ROI = poca::core::getROIFromType(m_currentInteractionMode);
				if (m_ROI != NULL) {
					x = coords[0];
					y = coords[1];
					z = coords[2];
				}
			}
			else if(poca::opengl::Camera::Line2DRoiDefinition <= m_currentInteractionMode && m_currentInteractionMode <= poca::opengl::Camera::Ellipse2DRoiDefinition){
				if (m_ROI == NULL)
					m_ROI = poca::core::getROIFromType(m_currentInteractionMode);
				if (m_ROI != NULL) {
					glm::vec3 coords = getWorldCoordinates(glm::vec2(_event->pos().x(), this->height() - _event->pos().y()));
					x = coords[0];
					y = coords[1];
				}
			}
			if(m_ROI != NULL)
				m_ROI->onClick(x, y, z);
			m_buttonOn = true;
			m_tmp = m_clickPoint;
		}
		else {
			if (m_insidePatchId != -1) {
				if (_event->button() == Qt::LeftButton) {
					m_undoPossible = true;
					m_distanceOrthoSaved = m_distanceOrtho;
					m_matrixModelSaved = m_matrixModel;
					m_matrixViewSaved = m_stateCamera.m_matrix;
					m_stateCamera = m_infoObjects[m_insidePatchId]->m_state;
					zoomToBoundingBox(m_infoObjects[m_insidePatchId]->m_bbox);
					recomputeFrame(m_currentCrop);

					poca::core::MediatorWObjectFWidget* mediator = poca::core::MediatorWObjectFWidget::instance();
					poca::core::CommandInfo ci(false, "selectObject", "id", m_infoObjects[m_insidePatchId]->m_id);
					mediator->actionAskedAllObservers(&ci);
				}
			}
			else {
				switch (_event->button())
				{
				case Qt::LeftButton:
					m_leftButtonOn = true;
					m_scaling = _event->modifiers() == Qt::ShiftModifier ? 1 : 0;
					break;
				case Qt::MiddleButton:
					m_middleButtonOn = true;
					break;
				case Qt::RightButton:
					m_rightButtonOn = true;
					break;
				}
				computePointOnSphere(m_clickPoint, m_startVector);

				m_buttonOn = true;
			}
		}
		doneCurrent();
		update();
	}

	void Camera::mouseMoveEvent(QMouseEvent* _event)
	{
		int x = _event->pos().x(), y = _event->pos().y();
		makeCurrent();
		setClickPoint(_event->pos().x(), _event->pos().y());
		m_cropPointEnd.set(_event->pos().x(), this->height() - _event->pos().y(), 0.f);

		glm::vec3 wrldCoords = getWorldCoordinates(glm::vec2(x, this->height() - y));
		glm::vec2 pixCoorlds = worldToScreenCoordinates(wrldCoords);
		std::vector <poca::core::Vec3mf> p = { poca::core::Vec3mf(m_cropPointEnd.x(), m_cropPointEnd.y()/*pixCoorlds.x, pixCoorlds.y*/, 0.f) };
		m_simplePointBuffer.updateBuffer(p);

		poca::core::BoundingBox cropBox(FLT_MAX, FLT_MAX, 0.f, -FLT_MAX, -FLT_MAX, 0.f);
		cropBox.addPointBBox(m_cropPointBegin.x(), m_cropPointBegin.y(), 0.f);
		cropBox.addPointBBox(m_cropPointEnd.x(), m_cropPointEnd.y(), 0.f);
		std::vector <poca::core::Vec3mf> cube(24);
		poca::geometry::createCubeFromVector(cube, cropBox);
		m_cropBuffer.updateBuffer(cube.data());

		if (!m_buttonOn) {
			m_insidePatchId = -1;
			for(size_t n = 0; n < m_infoObjects.size() && m_insidePatchId == -1; n++){
				if (m_infoObjects[n]->inside(x, y))
					m_insidePatchId = n;
			}
			poca::core::MediatorWObjectFWidget* mediator = poca::core::MediatorWObjectFWidget::instance();
			if (m_insidePatchId == -1) {
				poca::core::CommandInfo ci(false, "pick", 
					"x", _event->pos().x(), 
					"y", _event->pos().y(), 
					"saveImage", false,
					"infos", poca::core::stringList(),
					"pickedPoints", std::vector <poca::core::Vec3mf>());
				m_object->executeGlobalCommand(&ci);
				m_infoPicking.clear();
				poca::core::stringList listInfos = ci.getParameter<poca::core::stringList>("infos");
				for (std::string info : listInfos)
					m_infoPicking.push_back(QString::fromStdString(info));

				m_pickedPoints = ci.getParameter<std::vector <poca::core::Vec3mf>>("pickedPoints");

				mediator->actionAskedAllObservers(&poca::core::CommandInfo(false, "updatePickedObject"));

				glm::vec3 worldCoord = getWorldCoordinates(glm::vec2(x, this->height() - y));
				m_object->executeCommandOnSpecificComponent("VoronoiDiagram" , &poca::core::CommandInfo(false, "determineTrianglesLinkedToPoint", "x", worldCoord[0], "y", worldCoord[1], "z", 0.f));
			}
			else {
				poca::core::CommandInfo ci2("selectObject", "id" , m_infoObjects[m_insidePatchId]->m_id);
				mediator->actionAskedAllObservers(&ci2);
			}
			if (m_currentInteractionMode == poca::opengl::Camera::Polyline2DRoiDefinition) {
				if (m_ROI != NULL) {
					glm::vec3 coords = getWorldCoordinates(glm::vec2(_event->pos().x(), this->height() - _event->pos().y()));
					m_ROI->onMove(coords[0], coords[1]);
				}
			}
			if (m_currentInteractionMode == poca::opengl::Camera::PolyPlaneRoiDefinition) {
				if (m_ROI != NULL) {
					glm::vec3 coords = getWorldCoordinates(glm::vec2(_event->pos().x(), this->height() - _event->pos().y()));
					m_ROI->onMove(coords[0], coords[1], coords[2]);
				}
			}
		}
		else {
			if (m_currentInteractionMode != poca::opengl::Camera::None) {
				if (m_ROI != NULL) {
					if (m_currentInteractionMode == poca::opengl::Camera::Sphere3DRoiDefinition) {
						poca::core::SphereROI* roi = (poca::core::SphereROI*)m_ROI;
						glm::vec3 center(roi->getCenter()[0], roi->getCenter()[1], roi->getCenter()[2]);

						glm::vec3 right = glm::cross(m_stateCamera.m_up, m_stateCamera.m_eye);
						glm::vec3 p1 = getWorldCoordinates(glm::vec2(m_clickPoint[0], this->height() - m_clickPoint[1]));
						glm::vec3 diff = p1 - center;
						glm::vec3 rightVector = -glm::proj(diff, right);
						glm::vec3 upVector = glm::proj(diff, m_stateCamera.m_up);

						glm::vec3 point(center + rightVector + upVector);
						m_ROI->onMove(point[0], point[1], point[2]);
					}
					else {
						glm::vec3 coords = getWorldCoordinates(glm::vec2(_event->pos().x(), this->height() - _event->pos().y()));
						m_ROI->onMove(coords[0], coords[1], coords[2]);
					}
				}
			}
			else {
				if (m_scaling) {
					float w = m_object == NULL ? 100 : m_object->getWidth();
					m_distanceOrtho += ((float)(_event->pos().y() - m_prevClickPoint[1])) * (w / 1000.);
					if (m_distanceOrtho < 0.01f)
						m_distanceOrtho = 0.01f;
					recalcModelView();

					update();
				}
				else if (m_leftButtonOn) {
					computeRotation();
				}
				else if (m_middleButtonOn) {
					float w = m_currentCrop.realWidth(), h = m_currentCrop.realHeight();
					glm::vec3 right = glm::cross(m_stateCamera.m_up, m_stateCamera.m_eye);
					glm::vec3 p1 = getWorldCoordinates(m_clickPoint), p2 = getWorldCoordinates(m_prevClickPoint);
					glm::vec3 diff = p1 - p2;
					glm::vec3 rightVector = -glm::proj(diff, right);
					glm::vec3 upVector = glm::proj(diff, m_stateCamera.m_up);
					m_translation = m_translation + rightVector + upVector;
					m_stateCamera.m_translationModel = glm::vec3(-m_translation.x, -m_translation.y, -m_translation.z);

				}
			}
		}
		doneCurrent();
		update();
	}

	void Camera::mouseReleaseEvent(QMouseEvent* _event)
	{
		makeCurrent();
		switch (_event->button())
		{
		case Qt::LeftButton:
		{
			if (poca::opengl::Camera::Line2DRoiDefinition <= m_currentInteractionMode && m_currentInteractionMode <= poca::opengl::Camera::PlaneRoiDefinition) {
				if (m_currentInteractionMode != poca::opengl::Camera::Polyline2DRoiDefinition && m_currentInteractionMode != poca::opengl::Camera::PolyPlaneRoiDefinition) {
					if (m_ROI != NULL) {
						if (m_currentInteractionMode == poca::opengl::Camera::Sphere3DRoiDefinition) {
							poca::core::SphereROI* roi = (poca::core::SphereROI*)m_ROI;
							glm::vec3 center(roi->getCenter()[0], roi->getCenter()[1], roi->getCenter()[2]);

							glm::vec3 right = glm::cross(m_stateCamera.m_up, m_stateCamera.m_eye);
							glm::vec3 p1 = getWorldCoordinates(glm::vec2(m_clickPoint[0], this->height() - m_clickPoint[1]));
							glm::vec3 diff = p1 - center;
							glm::vec3 rightVector = -glm::proj(diff, right);
							glm::vec3 upVector = glm::proj(diff, m_stateCamera.m_up);

							glm::vec3 point(center + rightVector + upVector);
							m_ROI->finalize(point[0], point[1], point[2]);
							std::cout << m_ROI->toStdString() << std::endl;
						}
						else if (m_currentInteractionMode == poca::opengl::Camera::PlaneRoiDefinition) {
							poca::core::PlaneROI* proi = dynamic_cast<poca::core::PlaneROI*>(m_ROI);
							glm::vec3 coords = getWorldCoordinates(glm::vec2(_event->pos().x(), this->height() - _event->pos().y()));
							m_ROI->finalize(coords[0], coords[1], coords[2]);
							poca::core::Vec3mf p1 = proi->getPoint(0);

							Kernel::Plane_3 plane(Point_3_double(p1[0], p1[1], p1[2]), Point_3_double(coords[0], coords[1], coords[2]), Point_3_double(p1[0] + m_stateCamera.m_eye[0] * 10.f, p1[1] + m_stateCamera.m_eye[1] * 10.f, p1[2] + m_stateCamera.m_eye[2] * 10.f));
							m_clip.push_back(glm::vec4(plane.a(), plane.b(), plane.c(), plane.d()));
							Kernel::Iso_cuboid_3 cube(Point_3_double(m_currentCrop[0], m_currentCrop[1], m_currentCrop[2]), Point_3_double(m_currentCrop[3], m_currentCrop[4], m_currentCrop[5]));
							const auto result = CGAL::intersection(cube, plane);
							if (result) {
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(6, 0, 0)
								if (const Kernel::Triangle_3* t = std::get_if<Kernel::Triangle_3>(&*result)) {
									proi->addFinalPoint(poca::core::Vec3mf(t->vertex(0).x(), t->vertex(0).y(), t->vertex(0).z()));
									proi->addFinalPoint(poca::core::Vec3mf(t->vertex(1).x(), t->vertex(1).y(), t->vertex(1).z()));
									proi->addFinalPoint(poca::core::Vec3mf(t->vertex(2).x(), t->vertex(2).y(), t->vertex(2).z()));
								}
								if (const std::vector < Point_3_double > *s = std::get_if<std::vector < Point_3_double >>(&*result)) {
									for (const auto& p : *s) {
										proi->addFinalPoint(poca::core::Vec3mf(p.x(), p.y(), p.z()));
									}
								}
#else
								if (const Kernel::Triangle_3* t = boost::get<Kernel::Triangle_3>(&*result)) {
									proi->addFinalPoint(poca::core::Vec3mf(t->vertex(0).x(), t->vertex(0).y(), t->vertex(0).z()));
									proi->addFinalPoint(poca::core::Vec3mf(t->vertex(1).x(), t->vertex(1).y(), t->vertex(1).z()));
									proi->addFinalPoint(poca::core::Vec3mf(t->vertex(2).x(), t->vertex(2).y(), t->vertex(2).z()));
								}
								if (const std::vector < Point_3_double >* s = boost::get<std::vector < Point_3_double >>(&*result)) {
									for (const auto& p : *s) {
										proi->addFinalPoint(poca::core::Vec3mf(p.x(), p.y(), p.z()));
									}
								}
#endif
							}
						}
						else {
							glm::vec3 coords = getWorldCoordinates(glm::vec2(_event->pos().x(), this->height() - _event->pos().y()));
							m_ROI->finalize(coords[0], coords[1], coords[2]);
						}
						m_object->addROI(m_ROI);
						m_ROI = NULL;
					}
				}
			}
			else if (m_currentInteractionMode == poca::opengl::Camera::Crop) {
				poca::core::BoundingBox cropBox;
				switch (m_cropPlane) {
				case Plane_XY:
					cropBox.set(FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, FLT_MAX);
					break;
				case Plane_XZ:
					cropBox.set(FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX);
					break;
				case Plane_YZ:
					cropBox.set(-FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX);
					break;
				default:
					break;
				}
				glm::vec3 wrldCoordsBegin = getWorldCoordinates(glm::vec2(m_cropPointBegin.x(), m_cropPointBegin.y()));
				glm::vec3 wrldCoordsEnd = getWorldCoordinates(glm::vec2(m_cropPointEnd.x(), m_cropPointEnd.y()));
				cropBox.addPointBBox(wrldCoordsBegin.x, wrldCoordsBegin.y, wrldCoordsBegin.z);
				cropBox.addPointBBox(wrldCoordsEnd.x, wrldCoordsEnd.y, wrldCoordsEnd.z);
				std::cout << "CropBox = " << cropBox << std::endl;
				m_currentCrop = m_currentCrop.intersect(cropBox);
				std::cout << "m_currentCrop = " << cropBox << std::endl;
				zoomToBoundingBox(m_currentCrop);
			}
			if (m_leftButtonOn && !m_scaling)
				m_object->executeGlobalCommand(&poca::core::CommandInfo(false, "sortWRTCameraPosition", "cameraPosition", m_stateCamera.m_center, "cameraForward", m_stateCamera.m_eye));
			break;
		}
		case Qt::MiddleButton:
			break;
		case Qt::RightButton:
			poca::core::CommandInfo ci(false, "pick",
				"x", _event->pos().x(),
				"y", _event->pos().y(),
				"saveImage", false,
				"infos", poca::core::stringList());
			m_object->executeGlobalCommand(&ci);
			m_infoPicking.clear();
			poca::core::stringList listInfos = ci.getParameter<poca::core::stringList>("infos");
			for (std::string info : listInfos)
				m_infoPicking.push_back(QString::fromStdString(info));

			ci = poca::core::CommandInfo(true, "doubleClickCamera", "camera", this);
			m_object->executeCommandOnSpecificComponent("ObjectList", &ci);
			if (ci.hasParameter("bbox")) {
				poca::core::BoundingBox bbox = ci.getParameter<poca::core::BoundingBox>("bbox");
				QOpenGLFramebufferObject* fbo = ci.getParameterPtr<QOpenGLFramebufferObject>("fbo");
				size_t id = ci.getParameter<size_t>("id");
				int y = 20;
				InfosObjectImages* infos = new InfosObjectImages(fbo, poca::core::Vec4mf(0, y, m_sizePatch, m_sizePatch), bbox, m_stateCamera, id);
				m_infoObjects.insert(m_infoObjects.begin(), infos);
				if (m_infoObjects.size() > 8) {
					InfosObjectImages* tmp = m_infoObjects.back();
					delete tmp;
					m_infoObjects.pop_back();
				}
				for (std::vector <InfosObjectImages*>::iterator it = m_infoObjects.begin(); it != m_infoObjects.end(); it++) {
					(*it)->m_rect.setY(y);
					y += m_sizePatch + 20;

				}
			}
			break;
		}
		m_scaling = m_buttonOn = m_leftButtonOn = m_middleButtonOn = m_rightButtonOn = false;
		doneCurrent();
		update();
	}

	void Camera::mouseDoubleClickEvent(QMouseEvent* _event)
	{
		int x = _event->pos().x(), y = _event->pos().y();
		makeCurrent();
		if (_event->button() == Qt::LeftButton) {
			if (m_currentInteractionMode != poca::opengl::Camera::None) {
				if (m_currentInteractionMode == poca::opengl::Camera::Polyline2DRoiDefinition) {
					if (m_ROI != NULL) {
						glm::vec3 coords = getWorldCoordinates(glm::vec2(_event->pos().x(), this->height() - _event->pos().y()));
						m_ROI->finalize(coords[0], coords[1]);
						m_object->addROI(m_ROI);
						m_ROI = NULL;
					}
				}
				else if (m_currentInteractionMode == poca::opengl::Camera::PolyPlaneRoiDefinition) {
					poca::core::PolyplaneROI* proi = dynamic_cast<poca::core::PolyplaneROI*>(m_ROI);
					if (proi != NULL) {
						glm::vec3 coords = getWorldCoordinates(glm::vec2(_event->pos().x(), this->height() - _event->pos().y()));
						m_ROI->finalize(coords[0], coords[1], coords[2]);
						const std::vector<poca::core::Vec3mf>& points = proi->getPoints();

						for (const auto& pt : points) {
							Kernel::Line_3 line(Point_3_double(pt[0], pt[1], pt[2]), Kernel::Direction_3(m_stateCamera.m_eye[0], m_stateCamera.m_eye[1], m_stateCamera.m_eye[2]));
							Kernel::Iso_cuboid_3 cube(Point_3_double(m_currentCrop[0], m_currentCrop[1], m_currentCrop[2]), Point_3_double(m_currentCrop[3], m_currentCrop[4], m_currentCrop[5]));
							const auto result = CGAL::intersection(line, cube);
							if (result) {
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(6, 0, 0)
								if (const Kernel::Segment_3* s = std::get_if<Kernel::Segment_3>(&*result)) {
									proi->addFinalPoint(poca::core::Vec3mf(s->vertex(0).x(), s->vertex(0).y(), s->vertex(0).z()));
									proi->addFinalPoint(poca::core::Vec3mf(s->vertex(1).x(), s->vertex(1).y(), s->vertex(1).z()));
								}
								if (const Kernel::Point_3* p = std::get_if<Kernel::Point_3>(&*result)) {
									std::cout << "Point " << p->x() << ", " << p->y() << ", " << p->z() << std::endl;
								}
#else
								if (const Kernel::Segment_3* s = boost::get<Kernel::Segment_3>(&*result)) {
									proi->addFinalPoint(poca::core::Vec3mf(s->vertex(0).x(), s->vertex(0).y(), s->vertex(0).z()));
									proi->addFinalPoint(poca::core::Vec3mf(s->vertex(1).x(), s->vertex(1).y(), s->vertex(1).z()));
								}
								if (const Kernel::Point_3* p = boost::get<Kernel::Point_3>(&*result)) {
									std::cout << "Point " << p->x() << ", " << p->y() << ", " << p->z() << std::endl;
								}
#endif
							}
						}
						m_object->addROI(m_ROI);
						m_ROI = NULL;
					}
				}
				return;
			}
			m_insidePatchId = -1;
			for (size_t n = 0; n < m_infoObjects.size() && m_insidePatchId == -1; n++) {
				if (m_infoObjects[n]->inside(x, y))
					m_insidePatchId = n;
			}
			poca::core::MediatorWObjectFWidget* mediator = poca::core::MediatorWObjectFWidget::instance();
			if (m_insidePatchId == -1) {
				poca::core::CommandInfo ci = poca::core::CommandInfo(false, "doubleClickCamera", "camera", this);
				m_object->executeGlobalCommand(&ci);
				if (ci.hasParameter("bbox")) {
					poca::core::BoundingBox bbox = ci.getParameter<poca::core::BoundingBox>("bbox");
					zoomToBoundingBox(bbox);
					update();
				}
			}
		}
	}

	void Camera::wheelEvent(QWheelEvent* _event)
	{
		float mult = _event->angleDelta().y() < 0 ? 1.f : -1.f;

		float w = m_object == NULL ? 100 : m_object->getWidth();
		m_distanceOrtho += mult * 10.f * (w / 1000.);
		if (m_distanceOrtho < 0.01f)
			m_distanceOrtho = 0.01f;
		recalcModelView();

		update();
	}

	void Camera::resizeEvent(QResizeEvent* _event)
	{
		makeCurrent();
		QOpenGLWidget::resizeEvent(_event);
		recalcModelView();
	
		if (m_offscreenFBO != NULL)
			delete m_offscreenFBO;
		int w = this->width(), h = this->height();
		m_offscreenFBO = new QOpenGLFramebufferObject(this->width(), this->height(), QOpenGLFramebufferObject::Depth, GL_TEXTURE_2D, GL_RGBA);
		glBindTexture(GL_TEXTURE_2D, m_offscreenFBO->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, this->width(), this->height(), 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
		
		m_object->executeGlobalCommand(&poca::core::CommandInfo(false, "updatePickingBuffer", "width", this->width(), "height", this->height()));
		m_ssaoShader.update(w, h);

		update();
	}

	Shader* Camera::getShader(const std::string& _nameShader) 
	{
		if (m_shaders.find(_nameShader) != m_shaders.end()) return m_shaders[_nameShader];
		if (_nameShader == "geometrySSAO") {
			return m_ssaoShader.m_shaderGeometryPass;
		}
		if (_nameShader == "simpleShader") {
			Shader* shader = new Shader();
			shader->createAndLinkProgramFromStr(vs, fs);
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "pickShader") {
			Shader* shader = new Shader();
			shader->createAndLinkProgramFromStr(vsPick, fsPick);
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "uniformColorShader") {
			Shader* shader = new Shader();
			shader->createAndLinkProgramFromStr(vs, fsUniformColor);
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "stippleShader") {
			Shader* shader = new Shader();
			shader->createAndLinkProgramFromStr(vsStipple, fsStipple);
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "heatmapShader") {
			Shader* shader = new Shader("./shaders/heatmap.vs", "./shaders/heatmap.fs", "./shaders/heatmap.gs");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "textureShader") {
			Shader* shader = new Shader();
			shader->createAndLinkProgramFromStr(vsTexture, fsTexture);
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "textureShaderFBO") {
			Shader* shader = new Shader();
			shader->createAndLinkProgramFromStr(vsTexture, fsTextureFBO);
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "roundedRectShader") {
			//Shader* shader = new Shader();
			//shader->createAndLinkProgramFromStr(vsRoundedRect, fsRoundedRect);
			Shader* shader = new Shader("./shaders/roundedRect.vs", "./shaders/roundedRect.fs");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "circle2DShader") {
			//Shader* shader = new Shader();
			//shader->createAndLinkProgramFromStr(vsCircle2D, fsCircle2D);
			Shader* shader = new Shader("./shaders/circle2D.vs", "./shaders/circle2D.fs", "./shaders/circle2D.gs");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "line2DShader") {
			Shader* shader = new Shader("./shaders/lineRendering.vs", "./shaders/lineRendering.fs", "./shaders/lineRendering.gs");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "polyline2DShader") {
			//Shader* shader = new Shader();
			//shader->createAndLinkProgramFromStr(vsLines2D_3, fsLines2D_3, gsLines2D_3);
			Shader* shader = new Shader("./shaders/polyline2D.vs", "./shaders/polyline2D.fs", "./shaders/polyline2D.gs");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "pointRenderingRotationShader") {
			Shader* shader = new Shader();
			shader->createAndLinkProgramFromStr(vsPointRenderingRotation, fsPointRenderingRotation);
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "sphere3DShader") {
			Shader* shader = new Shader("./shaders/sphere3DImpostor.vs", "./shaders/sphere3DImpostor.fs", "./shaders/sphere3DImpostor.gs");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "sphereRenderingShader") {
			Shader* shader = new Shader("./shaders/sphereRendering2.vs", "./shaders/sphereRendering2.fs", "./shaders/sphereRendering2.gs");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "objectRenderingShader") {
			Shader* shader = new Shader("./shaders/objectRendering.vs", "./shaders/objectRendering.fs");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "objectRenderingSSAOShader") {
			Shader* shader = new Shader("./shaders/objectRendering.vs", "./shaders/objectRenderingSSAO.fs");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "normalRenderingShader") {
			Shader* shader = new Shader("./shaders/normalRendering.vs", "./shaders/normalRendering.fs", "./shaders/normalRendering.gs");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "2DGaussianRenderingShader") {
			Shader* shader = new Shader("./shaders/2DGaussianShader.vs", "./shaders/2DGaussianShader.fs", "./shaders/2DGaussianShader.gs");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "3DInstanceRenderingShader") {
			Shader* shader = new Shader("./shaders/3DvertexShaderUniformColor_instancedRendering.vs", "./shaders/fragmentShaderUniformOrNotColor.fs");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "alphaBlending") {
			Shader* shader = new Shader("./shaders/alpha_blending.vert", "./shaders/alpha_blending.frag");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "maximumProjection") {
			Shader* shader = new Shader("./shaders/maximum_intensity_projection.vert", "./shaders/maximum_intensity_projection.frag");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "isosurface") {
			Shader* shader = new Shader("./shaders/isosurface.vert", "./shaders/isosurface.frag");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		if (_nameShader == "labelRendering") {
			Shader* shader = new Shader("./shaders/label_rendering.vert", "./shaders/label_rendering.frag");
			m_shaders[_nameShader] = shader;
			return shader;
		}
		return nullptr;
	}

	void Camera::zoomToBoundingBox(const poca::core::BoundingBox& _bbox, const bool _recomputeOrthoD)
	{
		m_resetedProj = false;

		float w = _bbox[3] - _bbox[0], h = _bbox[4] - _bbox[1], t = _bbox[5] - _bbox[2];
		if (_recomputeOrthoD) {
			m_distanceOrtho = w > h ? w / 2 : h / 2;
			//m_distanceOrtho = m_distanceOrtho > t ? m_distanceOrtho : t;
			
			m_translation.x = _bbox[0] + (_bbox[3] - _bbox[0]) / 2.f;
			m_translation.y = _bbox[1] + (_bbox[4] - _bbox[1]) / 2.f;
			m_translation.z = _bbox[2] + (_bbox[5] - _bbox[2]) / 2.f;

			m_stateCamera.m_translationModel = glm::vec3(-m_translation.x, -m_translation.y, -m_translation.z);
		}

		recalcModelView();

		m_clip[0] = glm::vec4(1, 0, 0, -_bbox[0]);
		m_clip[1] = glm::vec4(0, 1, 0, -_bbox[1]);
		m_clip[2] = glm::vec4(0, 0, 1, -_bbox[2]);
		m_clip[3] = glm::vec4(-1, 0, 0, _bbox[3]);
		m_clip[4] = glm::vec4(0, -1, 0, _bbox[4]);
		m_clip[5] = glm::vec4(0, 0, -1, _bbox[5]);
		m_currentCrop = _bbox;

		//paintGL();
		repaint();
	}

	void Camera::resetProjection()
	{
		m_currentCrop = m_object->boundingBox();
		zoomToBoundingBox(m_currentCrop);
		
		//if the crop is reset we expand a little more the clip plane to be sure to not cut part of the models
		float smallest = m_currentCrop.smallestSide();
		float expansion = smallest;
		m_clip.resize(6);
		m_clip[0] = glm::vec4(1, 0, 0, -(m_currentCrop[0] - expansion));
		m_clip[1] = glm::vec4(0, 1, 0, -(m_currentCrop[1] - expansion));
		m_clip[2] = glm::vec4(0, 0, 1, -(m_currentCrop[2] - expansion));
		m_clip[3] = glm::vec4(-1, 0, 0, m_currentCrop[3] + expansion);
		m_clip[4] = glm::vec4(0, -1, 0, m_currentCrop[4] + expansion);
		m_clip[5] = glm::vec4(0, 0, -1, m_currentCrop[5] + expansion);

		m_resetedProj = true;
	}

	const bool Camera::testWhichFaceFrontCamera()
	{
		bool faceFrontCameraHaveChanged = false;

		glm::vec3 camDir = m_stateCamera.m_eye;
		camDir = glm::normalize(camDir);


		for (unsigned int n = 0; n < 6; n++) {
			double dot = glm::dot(cubeNormals[n], camDir);
			m_facingDirections[n] = dot > 0.f;
			faceFrontCameraHaveChanged = faceFrontCameraHaveChanged || (m_facingDirections[n] != m_facingDirections[n]);
		}
		return faceFrontCameraHaveChanged;
	}

	void Camera::determineLeftRightUpFaces(size_t _index)
	{
		glm::vec3 left(-1.f, 0.f, 0.f), up(0.f, 1.f, 0.f), right(1.f, 0.f, 0.f);
		std::array <float, 3> dots[3];
		std::array <size_t, 3> indexFaces;
		size_t cpt = 0;
		for (size_t n = 0; n < 6; n++) {
			if (!m_facingDirections[n]) continue;
			glm::vec4 vec = m_stateCamera.m_matrixView * glm::vec4(cubeNormals[n], 1.f);
			glm::vec3 vec2 = glm::vec3(vec[0], vec[1], vec[2]);
			indexFaces[cpt] = n;
			dots[cpt] = { glm::dot(vec2, left), fabs(glm::dot(vec2, up)), glm::dot(vec2, right) };
			cpt++;
		}
		size_t upIndex = 0;
		for (size_t n = 1; n < 3; n++)
			if (dots[n][1] > dots[upIndex][1]) upIndex = n;
		size_t leftIndex = 0, rightIndex = 0;
		for (size_t n = 1; n < 3; n++) {
			if (n == upIndex) continue;
			if (dots[n][0] > dots[leftIndex][0]) leftIndex = n;
			if (dots[n][2] > dots[rightIndex][2]) rightIndex = n;
		}
	}

	void Camera::recomputeFrame(const poca::core::BoundingBox& _bbox)
	{
		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(m_object);
		if (!comObj) return;

		const poca::core::BoundingBox& bboxObject = m_object->boundingBox();

		testWhichFaceFrontCamera();

		float w = _bbox[3] - _bbox[0], h = _bbox[4] - _bbox[1], t = _bbox[5] - _bbox[2];
		float dMin = w < h ? w : h;

		std::array <poca::core::Vec3mf, 8> pointsCube = { poca::core::Vec3mf(_bbox[0], _bbox[1], _bbox[2]),
			poca::core::Vec3mf(_bbox[0], _bbox[4], _bbox[2]),
			poca::core::Vec3mf(_bbox[3], _bbox[4], _bbox[2]),
			poca::core::Vec3mf(_bbox[3], _bbox[1], _bbox[2]),
			poca::core::Vec3mf(_bbox[0], _bbox[1], _bbox[5]),
			poca::core::Vec3mf(_bbox[0], _bbox[4], _bbox[5]),
			poca::core::Vec3mf(_bbox[3], _bbox[4], _bbox[5]),
			poca::core::Vec3mf(_bbox[3], _bbox[1], _bbox[5]) };

		std::vector <poca::core::Vec3mf> grid;

		float stepGridX, stepGridY, stepGridZ;
		//In case there's a problem with the command
		//We compute the step for the length display on the longest side of the dataset
		float side = w > h ? w : h, nbIntermediate = m_nbMainGrid, cur = 0.f;
		if (m_dimension == 3)
			side = side > t ? side : t;
		m_step = side / nbIntermediate;
		stepGridX = stepGridY = stepGridZ = m_step / m_nbIntermediateGrid;

		if (comObj->hasParameter("useNbForGrid")) {
			bool useNbForGrid = comObj->getParameter<bool>("useNbForGrid");
			if (useNbForGrid) {
				if (comObj->hasParameter("nbGrid")) {
					std::array <uint8_t, 3> nbs = comObj->getParameter<std::array <uint8_t, 3>>("nbGrid");
					stepGridX = w / nbs[0];
					stepGridY = w / nbs[1];
					stepGridZ = w / nbs[2];
				}
			}
			else {
				if (comObj->hasParameter("stepGrid")) {
					std::array <float, 3> nbs = comObj->getParameter<std::array <float, 3>>("stepGrid");
					stepGridX = nbs[0];
					stepGridY = nbs[1];
					stepGridZ = nbs[2];
				}
			}
		}

		glm::vec3 left(-1.f, 0.f, 0.f), up(0.f, 1.f, 0.f), right(1.f, 0.f, 0.f);
		std::array <float, 3> dots[3];
		std::array <size_t, 3> indexFaces;
		size_t cpt = 0;
		for (size_t n = 0; n < 6; n++) {
			if (!m_facingDirections[n]) continue;
			glm::vec4 vec = m_stateCamera.m_matrixView * glm::vec4(cubeNormals[n], 1.f);
			glm::vec3 vec2 = glm::vec3(vec[0], vec[1], vec[2]);
			indexFaces[cpt] = n;
			dots[cpt] = { glm::dot(vec2, left), fabs(glm::dot(vec2, up)), glm::dot(vec2, right) };
			cpt++;
		}
		std::vector <bool> selectedFaces(3, false);
		size_t upIndex = 0, leftIndex = 0, rightIndex = 0;
		for (size_t n = 1; n < 3; n++)
			if (dots[n][1] > dots[upIndex][1]) upIndex = n;
		selectedFaces[upIndex] = true;
		std::pair maxHorizontal = std::make_pair(0, 0);
		for (size_t n = 0; n < 3; n++) {
			if (n == upIndex) continue;
			if (dots[n][0] > dots[maxHorizontal.first][maxHorizontal.second]) maxHorizontal = std::make_pair(n, 0);
			if (dots[n][2] > dots[maxHorizontal.first][maxHorizontal.second]) maxHorizontal = std::make_pair(n, 2);
		}
		if (maxHorizontal.second == 0) {
			leftIndex = maxHorizontal.first;
			selectedFaces[leftIndex] = true;
			for (size_t n = 0; n < selectedFaces.size(); n++) {
				if (selectedFaces[n]) continue;
				rightIndex = n;
			}
		}
		else {
			rightIndex = maxHorizontal.first;
			selectedFaces[rightIndex] = true;
			for (size_t n = 0; n < selectedFaces.size(); n++) {
				if (selectedFaces[n]) continue;
				leftIndex = n;
			}
		}
		size_t leftFace = indexFaces[leftIndex], rightFace = indexFaces[rightIndex], oppUpFace = (indexFaces[upIndex] + 3) % 6, oppLeftFace = (indexFaces[leftIndex] + 3) % 6;
		std::vector <poca::core::Vec3mf> vecsFaces, vecsLines;
		std::vector <poca::core::Vec3mf> vPoints;
		//Lines
		cpt = 0;
		uint32_t nbFacingDir = 0;
		for (auto n = 0; n < 6; n++)
			nbFacingDir = nbFacingDir + (m_facingDirections[n] ? 1 : 0);
		std::set <size_t> displayedEdges;
		auto indexFace = 0;
		if (nbFacingDir < 3) {
			vecsLines.resize(8);
			indexFace = nbFacingDir == 1 ? 0 : 1;
			for (auto i = indexFace; i < 6; i++) {
				if (!m_facingDirections[i]) continue;
				for (auto n = 0; n < 4; n++) {
					size_t e = cubeFaceIndexEdges[i][n];
					vecsLines[cpt++] = pointsCube[cubeEdgeIndexPoints[e][0]];
					vecsLines[cpt++] = pointsCube[cubeEdgeIndexPoints[e][1]];
					displayedEdges.insert(e);
				}
			}
		}
		else {
			vecsLines.resize(6);
			for (size_t n = 0; n < 12; n++) {
				if ((cubeEdgeIndexFaces[n][0] == leftFace && cubeEdgeIndexFaces[n][1] == oppUpFace) || (cubeEdgeIndexFaces[n][0] == oppUpFace && cubeEdgeIndexFaces[n][1] == leftFace)) {
					vecsLines[cpt++] = pointsCube[cubeEdgeIndexPoints[n][0]];
					vecsLines[cpt++] = pointsCube[cubeEdgeIndexPoints[n][1]];
					displayedEdges.insert(n);
				}
				if ((cubeEdgeIndexFaces[n][0] == rightFace && cubeEdgeIndexFaces[n][1] == oppUpFace) || (cubeEdgeIndexFaces[n][0] == oppUpFace && cubeEdgeIndexFaces[n][1] == rightFace)) {
					vecsLines[cpt++] = pointsCube[cubeEdgeIndexPoints[n][0]];
					vecsLines[cpt++] = pointsCube[cubeEdgeIndexPoints[n][1]];
					displayedEdges.insert(n);
				}
				if ((cubeEdgeIndexFaces[n][0] == rightFace && cubeEdgeIndexFaces[n][1] == oppLeftFace) || (cubeEdgeIndexFaces[n][0] == oppLeftFace && cubeEdgeIndexFaces[n][1] == rightFace)) {
					vecsLines[cpt++] = pointsCube[cubeEdgeIndexPoints[n][0]];
					vecsLines[cpt++] = pointsCube[cubeEdgeIndexPoints[n][1]];
					displayedEdges.insert(n);
				}
			}
		}
		float stepGrid = stepGridX;
		//Faces
		cpt = 0;
		m_frameTexts.clear();
		for (size_t n = indexFace; n < 6; n++) {
			if (nbFacingDir < 3 && !m_facingDirections[n]) continue;
			if (nbFacingDir == 3 && m_facingDirections[n]) continue;
			//Creation of the face
			size_t indexes[3] = { 0, 1, 2 };
			for (size_t i = 0; i < 3; i++)
				vecsFaces.push_back(pointsCube[cubeFaceIndexPoints[n][indexes[i]]]);
			size_t indexes2[3] = { 2, 3, 0 };
			for (size_t i = 0; i < 3; i++)
				vecsFaces.push_back(pointsCube[cubeFaceIndexPoints[n][indexes2[i]]]);

			//Creation of the grid
			poca::core::Vec3mf vects[2] = { pointsCube[cubeFaceIndexPoints[n][1]] - pointsCube[cubeFaceIndexPoints[n][0]],
			pointsCube[cubeFaceIndexPoints[n][2]] - pointsCube[cubeFaceIndexPoints[n][1]] };
			for (size_t i = 0; i < 2; i++) {
				const poca::core::Vec3mf& otherVect = vects[(i + 1) % 2];
				poca::core::Vec3mf nv = vects[i];
				nv.normalize();
	
				float dotX = nv.dot(poca::core::Vec3mf(1.f, 0.f, 0.f));
				float dotY = nv.dot(poca::core::Vec3mf(0.f, 1.f, 0.f));
				float dotZ = nv.dot(poca::core::Vec3mf(0.f, 0.f, 1.f));
				if (dotX == 1.f || dotX == -1.f)
					stepGrid = stepGridX;
				if (dotY == 1.f || dotY == -1.f)
					stepGrid = stepGridY;
				if (dotZ == 1.f || dotZ == -1.f)
					stepGrid = stepGridZ;

				cur = stepGrid;
				float lengthVec = vects[i].length(), curLength = cur;
				while ((curLength + stepGrid) < lengthVec) {
					poca::core::Vec3mf tmp = pointsCube[cubeFaceIndexPoints[n][0]] + nv * cur;
					grid.push_back(tmp);
					grid.push_back(tmp + otherVect);
					curLength = (tmp - pointsCube[cubeFaceIndexPoints[n][0]]).length();
					cur += stepGrid;
				}
			}

			//First, determine if one or several edges of the face are part of the displayed bbox
			for (size_t i = 0; i < 4; i++) {
				size_t indexEdge = cubeFaceIndexEdges[n][i];
				if (displayedEdges.find(indexEdge) != displayedEdges.end()) {
					size_t closeEdge = cubeFaceIndexEdges[n][(i + 1) % 4], oppEdge = cubeFaceIndexEdges[n][(i + 2) % 4];
					poca::core::Vec3mf vectDir(pointsCube[cubeEdgeIndexPoints[indexEdge][1]] - pointsCube[cubeEdgeIndexPoints[indexEdge][0]]);
					poca::core::Vec3mf orthDir;
					if(cubeEdgeIndexPoints[closeEdge][0] == cubeEdgeIndexPoints[indexEdge][0] || cubeEdgeIndexPoints[closeEdge][0] == cubeEdgeIndexPoints[indexEdge][1])
						orthDir = pointsCube[cubeEdgeIndexPoints[closeEdge][0]] - pointsCube[cubeEdgeIndexPoints[closeEdge][1]];
					else
						orthDir = pointsCube[cubeEdgeIndexPoints[closeEdge][1]] - pointsCube[cubeEdgeIndexPoints[closeEdge][0]];
					for (size_t k = 0; k < 2; k++) {
						float dot = vectDir.dot(vects[k]);
						if (dot == 0.f) continue;
						poca::core::Vec3mf start;
						if (dot > 0.f)
							start = pointsCube[cubeEdgeIndexPoints[indexEdge][0]];
						else
							start = pointsCube[cubeEdgeIndexPoints[indexEdge][1]];

						const poca::core::Vec3mf& otherVect = vects[(k + 1) % 2];
						poca::core::Vec3mf nv = vects[k];
						nv.normalize();

						float dotX = nv.dot(poca::core::Vec3mf(1.f, 0.f, 0.f));
						float dotY = nv.dot(poca::core::Vec3mf(0.f, 1.f, 0.f));
						float dotZ = nv.dot(poca::core::Vec3mf(0.f, 0.f, 1.f));
						float startingPoint = 0.f;
						if (dotX == 1.f || dotX == -1.f) {
							stepGrid = stepGridX;
							startingPoint = bboxObject.x();
						}
						if (dotY == 1.f || dotY == -1.f) {
							stepGrid = stepGridY;
							startingPoint = bboxObject.y();
						}
						if (dotZ == 1.f || dotZ == -1.f) {
							stepGrid = stepGridZ;
							startingPoint = bboxObject.z();
						}

						cur = stepGrid;
						float lengthVec = vects[k].length(), curLength = cur;
						while ((curLength + stepGrid) < lengthVec) {
							poca::core::Vec3mf tmp = start + nv * cur, tmp2 = tmp + (orthDir * 0.1f);
							curLength = (tmp - start).length();
							cur += stepGrid;

							const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
							glm::vec2 p0(worldToScreenCoordinates(proj, view, model, m_viewport, glm::vec3(tmp[0], tmp[1], tmp[2])));
							glm::vec2 p1(worldToScreenCoordinates(proj, view, model, m_viewport, glm::vec3(tmp2[0], tmp2[1], tmp2[2])));
							glm::vec2 vecPix = glm::normalize(p1 - p0);
							glm::vec2 posTxt = p0 + vecPix * 25.f;//25 pixels
							m_frameTexts.push_back(std::make_pair(poca::core::Vec3mf(posTxt[0], this->height() - posTxt[1], 0.f), std::to_string((uint32_t)(startingPoint + curLength))));
						}
					}
				}
			}
		}
		m_lineBboxBuffer.updateBuffer(vecsLines);
		m_faceGridBuffer.updateBuffer(vecsFaces);
		m_lineGridBuffer.updateBuffer(grid);

		m_debugPointBuffer.updateBuffer(vPoints);
	
		recomputeLegendThumbnailFrame();
	}

	glm::vec3 Camera::getWorldCoordinates(const glm::vec2& _winCoords)
	{
		glm::mat4 projection = m_matrixProjection * m_stateCamera.m_matrixView;
		return glm::unProject(glm::vec3(_winCoords, 0), m_matrixModel, projection, m_viewport);
	}

	glm::vec2 Camera::worldToScreenCoordinates(const glm::vec3& _wpos) const
	{
		glm::vec4 clipSpacePos = (m_matrixProjection * m_stateCamera.m_matrixView * m_matrixModel) * glm::vec4(_wpos, 1.f);
		glm::vec3 ndcSpacePos = glm::vec3(clipSpacePos.x, clipSpacePos.y, clipSpacePos.z) / clipSpacePos.w;
		glm::vec2 viewOffset(m_viewport.x, m_viewport.y), viewSize(m_viewport[2], m_viewport[3]);
		glm::vec2 windowSpacePos = ((glm::vec2(ndcSpacePos.x, ndcSpacePos.y) + 1.f) / 2.f) * viewSize + viewOffset;
		return windowSpacePos;
	}

	glm::vec2 Camera::worldToScreenCoordinates(const glm::mat4& _proj, const glm::mat4& _view, const glm::mat4& _model, const glm::uvec4& _viewport, const glm::vec3& _wpos) const
	{
		glm::vec4 clipSpacePos = (_proj * _view * _model) * glm::vec4(_wpos, 1.f);
		glm::vec3 ndcSpacePos = glm::vec3(clipSpacePos.x, clipSpacePos.y, clipSpacePos.z) / clipSpacePos.w;
		glm::vec2 viewOffset(_viewport.x, _viewport.y), viewSize(_viewport[2], _viewport[3]);
		glm::vec2 windowSpacePos = ((glm::vec2(ndcSpacePos.x, ndcSpacePos.y) + 1.f) / 2.f) * viewSize + viewOffset;
		return windowSpacePos;
	}

	const glm::vec3& Camera::getCenter()
	{
		return m_stateCamera.m_center;
	}

	const glm::vec3& Camera::getEye()
	{
		return m_stateCamera.m_eye;
	}

	const glm::mat4& Camera::getMatrix()
	{
		return m_stateCamera.m_matrix;
	}

	const float* Camera::getMatrixFlat()
	{
		return glm::value_ptr(m_stateCamera.m_matrix);
	}

	const glm::vec3& Camera::getUp()
	{
		return m_stateCamera.m_up;
	}

	void Camera::reset()
	{
		setEye(0.f, 0.f, 1.f);
		setCenter(0.f, 0.f, 0.f);
		setUp(0.f, 1.f, 0.f);

		updateCamera();
	}

	void Camera::setEye(float x, float y, float z)
	{
		m_stateCamera.m_eye.x = x;
		m_stateCamera.m_eye.y = y;
		m_stateCamera.m_eye.z = z;
	}

	void Camera::setEye(const glm::vec3& e)
	{
		m_stateCamera.m_eye = e;
	}

	void Camera::setCenter(float x, float y, float z)
	{
		m_stateCamera.m_center.x = x;
		m_stateCamera.m_center.y = y;
		m_stateCamera.m_center.z = z;
	}

	void Camera::setCenter(const glm::vec3& c)
	{
		m_stateCamera.m_center = c;
	}

	void Camera::setUp(float x, float y, float z)
	{
		m_stateCamera.m_up.x = x;
		m_stateCamera.m_up.y = y;
		m_stateCamera.m_up.z = z;
	}

	void Camera::setUp(const glm::vec3& u)
	{
		m_stateCamera.m_up = u;
	}

	void Camera::updateCamera()
	{
		m_stateCamera.m_matrix = glm::lookAt(m_stateCamera.m_eye, m_stateCamera.m_center, m_stateCamera.m_up);
	}

	void Camera::computeRotation()
	{
		if (m_prevClickPoint == m_clickPoint || m_preventRotation) {
			// Not moving during drag state, so skip unnecessary processing.
			return;
		}


		computePointOnSphere(m_clickPoint, m_stopVector);
		computeRotationBetweenVectors(m_startVector,
			m_stopVector,
			m_stateCamera.m_rotation);
		// Reverse so scene moves with cursor and not away due to camera model.
		m_stateCamera.m_rotation = glm::inverse(m_stateCamera.m_rotation);

		m_stateCamera.m_rotationSum *= m_stateCamera.m_rotation; // Accumulate quaternions.

		updateCameraEyeUp(true, true);

		// After applying drag, reset relative start state.
		m_prevClickPoint = m_clickPoint;
		m_startVector = m_stopVector;
	}

	float factor = 1.f, multiplier = 1.f;

	void Camera::animateRotation()
	{
		if (cptAnimation >= 360) return;
		cptAnimation++;

		float angle = m_multAnimation * 0.0174533f;
		glm::vec3 axis = glm::vec3(0,1,0);
		m_stateCamera.m_rotation = glm::angleAxis(angle, axis);
		// Reverse so scene moves with cursor and not away due to camera model.
		m_stateCamera.m_rotation = glm::inverse(m_stateCamera.m_rotation);
		m_stateCamera.m_rotationSum *= m_stateCamera.m_rotation; // Accumulate quaternions.
		updateCameraEyeUp(true, true);

		makeCurrent();
		if (cptAnimation % 1 == 0) {
			drawElements(m_offscreenFBO);
		}
		repaint();
	}

	void Camera::updateCameraEyeUp(bool eye, bool up)
	{
		if (eye) {
			glm::vec3 eye;
			computeCameraEye(eye);
			setEye(eye);
		}
		if (up) {
			glm::vec3 up;
			computeCameraUp(up);
			setUp(up);
		}
		updateCamera();
	}

	void Camera::computeCameraEye(glm::vec3& eye)
	{
		glm::vec3 orientation = m_stateCamera.m_rotationSum * glm::vec3(0.f, 0.f, 1.f);

		eye = orientation + m_stateCamera.m_center;
	}

	void Camera::computeCameraUp(glm::vec3& up)
	{
		up = glm::normalize(m_stateCamera.m_rotationSum * glm::vec3(0.f, 1.f, 0.f));
	}

	void Camera::computePointOnSphere(
		const glm::vec2& point, glm::vec3& result)
	{
		// https://www.opengl.org/wiki/Object_Mouse_Trackball
		float x = (2.f * point.x - this->width()) / this->width();
		float y = (this->height() - 2.f * point.y) / this->height();

		float length2 = x * x + y * y;

		if (length2 <= .5) {
			result.z = sqrt(1.0 - length2);
		}
		else {
			result.z = 0.5 / sqrt(length2);
		}

		float norm = 1.0 / sqrt(length2 + result.z * result.z);

		result.x = x * norm;
		result.y = y * norm;
		result.z *= norm;
	}

	void Camera::computeRotationBetweenVectors(const glm::vec3& u, const glm::vec3& v, glm::quat& result)
	{
		float cosTheta = glm::dot(u, v);

		if (m_dimension == 3) {
			glm::vec3 rotationAxis;
			static const float EPSILON = 1.0e-5f;

			if (cosTheta < -1.0f + EPSILON) {
				// Parallel and opposite directions.
				rotationAxis = glm::cross(glm::vec3(0.f, 0.f, 1.f), u);

				if (glm::length2(rotationAxis) < 0.01) {
					// Still parallel, retry.
					rotationAxis = glm::cross(glm::vec3(1.f, 0.f, 0.f), u);
				}

				rotationAxis = glm::normalize(rotationAxis);
				result = glm::angleAxis(180.0f, rotationAxis);
			}
			else if (cosTheta > 1.0f - EPSILON) {
				// Parallel and same direction.
				result = glm::quat(1, 0, 0, 0);
				return;
			}
			else {
				float theta = acos(cosTheta);
				rotationAxis = glm::cross(u, v);

				rotationAxis = glm::normalize(rotationAxis);
				result = glm::angleAxis(theta, rotationAxis);
			}
		}
		else {
			float theta = acos(cosTheta);
			glm::vec3 axis = glm::vec3(0, 0, 1);
			m_stateCamera.m_rotation = glm::angleAxis(theta, axis);
		}
	}

	void Camera::setClickPoint(double x, double y)
	{
		m_prevClickPoint = m_clickPoint;
		m_clickPoint.x = x;
		m_clickPoint.y = y;
	}

	void Camera::createArrowsFrame()
	{
		std::vector <poca::core::Vec3mf> arrowVertices;
		float radius = .02f, height = 0.35f;
		if (m_dimension == 3)
			generateDirectionalArrow(poca::core::Vec3mf(0, 0, 1), poca::core::Vec3mf(1, 0, 0), poca::core::Vec3mf(0, 1, 0), radius, height, arrowVertices);
		generateDirectionalArrow(poca::core::Vec3mf(0, 1, 0), poca::core::Vec3mf(1, 0, 0), poca::core::Vec3mf(0, 0, 1), radius, height, arrowVertices);
		generateDirectionalArrow(poca::core::Vec3mf(1, 0, 0), poca::core::Vec3mf(0, 1, 0), poca::core::Vec3mf(0, 0, 1), radius, height, arrowVertices);
		m_nbArrowsFrame = arrowVertices.size();
		if(m_vertexArrowBuffer.empty())
			m_vertexArrowBuffer.generateBuffer(arrowVertices.size(), 512 * 512, 3, GL_FLOAT);
		m_vertexArrowBuffer.updateBuffer(arrowVertices);
	}

	void Camera::generateDirectionalArrow(const poca::core::Vec3mf& _direction, const poca::core::Vec3mf& _axis1, const poca::core::Vec3mf& _axis2, const float _radius, const float _height, std::vector <poca::core::Vec3mf>& _arrowVertices)
	{
		float radius = _radius, height = _height, step = 2 * M_PI / 40.f, cur = 0.f, next = step;
		poca::core::Vec3mf heightCoordinate = _direction * _height, heightCoordinateArrow = _direction * (_height + _height / 4.f), xc, yc, xn, yn;
		while (next <= 2 * M_PI) {
			xc = _axis1 * (radius * cos(cur));
			yc = _axis2 * (radius * sin(cur));
			xn = _axis1 * (radius * cos(next));
			yn = _axis2 * (radius * sin(next));
			_arrowVertices.push_back(xc + yc);
			_arrowVertices.push_back(xc + yc + heightCoordinate);
			_arrowVertices.push_back(xn + yn);
			_arrowVertices.push_back(xn + yn);
			_arrowVertices.push_back(xc + yc + heightCoordinate);
			_arrowVertices.push_back(xn + yn + heightCoordinate);
			cur = next;
			next += step;
		}
		cur = 0.f;
		next = step;
		while (next <= 2 * M_PI) {
			xc = _axis1 * (radius * cos(cur));
			yc = _axis2 * (radius * sin(cur));
			xn = _axis1 * (radius * cos(next));
			yn = _axis2 * (radius * sin(next));
			_arrowVertices.push_back(poca::core::Vec3mf(0.f, 0.f, 0.f));
			_arrowVertices.push_back(xc + yc);
			_arrowVertices.push_back(xn + yn);
			cur = next;
			next += step;
		}
		radius *= 2.f;
		cur = 0.f;
		next = step;
		while (next <= 2 * M_PI) {
			xc = _axis1 * (radius * cos(cur));
			yc = _axis2 * (radius * sin(cur));
			xn = _axis1 * (radius * cos(next));
			yn = _axis2 * (radius * sin(next));
			_arrowVertices.push_back(heightCoordinate);
			_arrowVertices.push_back(xc + yc + heightCoordinate);
			_arrowVertices.push_back(xn + yn + heightCoordinate);
			_arrowVertices.push_back(heightCoordinateArrow);
			_arrowVertices.push_back(xc + yc + heightCoordinate);
			_arrowVertices.push_back(xn + yn + heightCoordinate);
			cur = next;
			next += step;
		}
	}

	void Camera::recomputeLegendThumbnailFrame()
	{
		const glm::mat4& proj = m_matrixProjectionThumbnailFrame, & view = getViewMatrix(), & model = glm::mat4(1.f);
		float value = 0.5f;
		glm::vec2 coordX = worldToScreenCoordinates(proj, view, model, m_viewportThumbnailFrame, glm::vec3(value, 0.f, 0.f));
		glm::vec2 coordY = worldToScreenCoordinates(proj, view, model, m_viewportThumbnailFrame, glm::vec3(0.f, value, 0.f));
		glm::vec2 coordZ = worldToScreenCoordinates(proj, view, model, m_viewportThumbnailFrame, glm::vec3(0.f, 0.f, value));
		m_frameTextsThumbnail.clear();
		m_frameTextsThumbnail.push_back(std::make_pair(poca::core::Vec3mf(coordX[0], m_viewport[3] - coordX[1], 0.f), "X"));
		m_frameTextsThumbnail.push_back(std::make_pair(poca::core::Vec3mf(coordY[0], m_viewport[3] - coordY[1], 0.f), "Y"));
		if(m_dimension == 3)
			m_frameTextsThumbnail.push_back(std::make_pair(poca::core::Vec3mf(coordZ[0], m_viewport[3] - coordZ[1], 0.f), "Z"));
	}

	void Camera::displayArrowsFrame()
	{
		//if (m_dimension == 2) return;
		GLfloat bkColor[4];
		glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
		for (size_t n = 0; n < 4; n++)
			bkColor[n] *= 255.f;
		poca::core::Color4D colorFront = poca::core::contrastColor(poca::core::Color4D(bkColor[0], bkColor[1], bkColor[2], bkColor[3]));
		double a = 1 - (0.299 * bkColor[0] + 0.587 * bkColor[1] + 0.114 * bkColor[2]) / 255;
		poca::core::Color4D	colorBack;
		if (a < 0.5)
			colorBack.set(0.f, 0.f, 0.f, 1); // bright colors - black font
		else
			colorBack.set(1.f, 1.f, 1.f, 1); // dark colors - white font
		poca::opengl::Shader* shader = getShader("uniformColorShader");
		const glm::mat4& proj = m_matrixProjectionThumbnailFrame, & view = getViewMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view);
		shader->setVec4("singleColor", colorBack[0], colorBack[1], colorBack[2], colorBack[3]);
		shader->setBool("clip", clip());
		glEnableVertexAttribArray(0);
		m_vertexArrowBuffer.bindBuffer(0, 0);
		glDrawArrays(m_vertexArrowBuffer.getMode(), 0, m_vertexArrowBuffer.getSizeBuffers()[0]); // Starting from vertex 0; 3 vertices total -> 1 triangle
		glDisableVertexAttribArray(0);
		shader->release();
	}

	void Camera::setCameraInteraction(const int _val) 
	{
		m_currentInteractionMode = _val;
		if (m_ROI != NULL) {
			m_object->addROI(m_ROI);
			m_ROI = NULL;
		}
	}

	void Camera::toggleSrcBlendingFactor()
	{
		struct BlendingType{
			GLenum src;
			std::string name;
		};
		std::vector < BlendingType > blend_factor = {
		{ GL_ZERO, "GL_ZERO" },
		{ GL_ONE, "GL_ONE" }, 
		{ GL_SRC_COLOR, "GL_SRC_COLOR" },
		{ GL_ONE_MINUS_SRC_COLOR, "GL_ONE_MINUS_SRC_COLOR" },
		{ GL_DST_COLOR, "GL_DST_COLOR" },
		{ GL_ONE_MINUS_DST_COLOR, "GL_ONE_MINUS_DST_COLOR" },
		{ GL_SRC_ALPHA, "GL_SRC_ALPHA" }, 
		{ GL_ONE_MINUS_SRC_ALPHA, "GL_ONE_MINUS_SRC_ALPHA" },
		{ GL_DST_ALPHA, "GL_DST_ALPHA" },
		{ GL_ONE_MINUS_DST_ALPHA, "GL_ONE_MINUS_DST_ALPHA" },
		{ GL_CONSTANT_COLOR, "GL_CONSTANT_COLOR" },
		{ GL_ONE_MINUS_CONSTANT_COLOR, "GL_ONE_MINUS_CONSTANT_COLOR" },
		{ GL_CONSTANT_ALPHA, "GL_CONSTANT_ALPHA" },
		{ GL_ONE_MINUS_CONSTANT_ALPHA, "GL_ONE_MINUS_CONSTANT_ALPHA" },
		{ GL_SRC_ALPHA_SATURATE, "GL_SRC_ALPHA_SATURATE" },
		};

		m_curIndexSource = (m_curIndexSource + 1) % blend_factor.size();
		m_sourceFactorBlending = blend_factor[m_curIndexSource].src;

		std::cout << "Current source blending factor = " << blend_factor[m_curIndexSource].name << std::endl;
		repaint();
	}

	void Camera::toggleDstBlendingFactor()
	{
		struct BlendingType {
			GLenum src;
			std::string name;
		};
		std::vector < BlendingType > blend_factor = {
		{ GL_ZERO, "GL_ZERO" },
		{ GL_ONE, "GL_ONE" },
		{ GL_SRC_COLOR, "GL_SRC_COLOR" },
		{ GL_ONE_MINUS_SRC_COLOR, "GL_ONE_MINUS_SRC_COLOR" },
		{ GL_DST_COLOR, "GL_DST_COLOR" },
		{ GL_ONE_MINUS_DST_COLOR, "GL_ONE_MINUS_DST_COLOR" },
		{ GL_SRC_ALPHA, "GL_SRC_ALPHA" },
		{ GL_ONE_MINUS_SRC_ALPHA, "GL_ONE_MINUS_SRC_ALPHA" },
		{ GL_DST_ALPHA, "GL_DST_ALPHA" },
		{ GL_ONE_MINUS_DST_ALPHA, "GL_ONE_MINUS_DST_ALPHA" },
		{ GL_CONSTANT_COLOR, "GL_CONSTANT_COLOR" },
		{ GL_ONE_MINUS_CONSTANT_COLOR, "GL_ONE_MINUS_CONSTANT_COLOR" },
		{ GL_CONSTANT_ALPHA, "GL_CONSTANT_ALPHA" },
		{ GL_ONE_MINUS_CONSTANT_ALPHA, "GL_ONE_MINUS_CONSTANT_ALPHA" }
		};

		m_curIndexDest = (m_curIndexDest + 1) % blend_factor.size();
		m_destFactorBlending = blend_factor[m_curIndexDest].src;

		std::cout << "Current dest blending factor = " << blend_factor[m_curIndexDest].name << std::endl;
		repaint();
	}

	void Camera::toggleFontDisplay() 
	{
		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(m_object);
		if (!comObj) return;

		if (!comObj->hasParameter("fontDisplay"))
			return;

		bool displayFont = comObj->getParameter<bool>("fontDisplay");
		displayFont = !displayFont;
		m_object->executeCommand(&poca::core::CommandInfo(true, "fontDisplay", displayFont));
	}

	void Camera::fixPlane(const int _type, const bool _on)
	{
		m_preventRotation = _on;
		m_cropPlane = _type;
		if (!_on)
			return;
		switch (m_cropPlane) {
		case Plane_XY:
			m_stateCamera.m_rotation = glm::quat(1.f, 0, 0, 0);
			break;
		case Plane_XZ:
			m_stateCamera.m_rotation = glm::quat(glm::vec3(M_PI / 2.f, 0, 0));
			break;
		case Plane_YZ:
			m_stateCamera.m_rotation = glm::quat(glm::vec3(0, -M_PI / 2.f, 0));
			break;
		default:
			return;
		}
		m_stateCamera.m_rotationSum = m_stateCamera.m_rotation;
		updateCameraEyeUp(true, true);
		m_stateCamera.m_matrixView = m_stateCamera.m_matrix;
		repaint();
	}

	void Camera::animateCameraPath(const std::array <poca::opengl::StateCamera, 2>& _states, const std::array <float, 2>& _distances, const float _duration, const bool _saveImages, const bool _traveling)
	{
		m_statesPath = _states;

		if (m_timerCameraPath == NULL) {
			m_timerCameraPath = new QTimer(this);
			connect(m_timerCameraPath, SIGNAL(timeout()), this, SLOT(animateCameraPath()));
		}

		m_saveImagesCameraPath = _saveImages;
		m_travelingCameraPath = _traveling;
		float nbs = _duration * 25.f;
		m_currentStepPath = m_stepPath = 1.f / nbs;
		m_stepDistanceCameraPath = (_distances[1] - _distances[0]) / nbs;
		m_stepTranslationCameraPath = _states[1].m_translationModel - _states[0].m_translationModel;
		m_stepTranslationCameraPath = m_stepTranslationCameraPath / nbs;
		m_nbImagesCameraPath = (uint32_t)nbs;
		m_angleRotation = (2 * M_PI) / nbs;

		if (m_travelingCameraPath) {
			m_distanceOrtho = _distances[0];
			m_stateCamera = _states[0];
			//m_stateCamera.m_translationModel = _states[0].m_translationModel;
		}

		m_timerCameraPath->start(10);
	}

	void Camera::animateCameraPath(const std::vector <std::tuple<float, glm::vec3, glm::quat>>& _iterations, const bool _saveImages, const bool _traveling) {
		m_pathIterations = _iterations;
		m_saveImagesCameraPath = _saveImages;
		m_travelingCameraPath = _traveling;
		m_currentStepPath = 0;
		m_angleRotation = (2 * M_PI) / (float)_iterations.size();
		if (m_timerCameraPath == NULL) {
			m_timerCameraPath = new QTimer(this);
			connect(m_timerCameraPath, SIGNAL(timeout()), this, SLOT(animateCameraPath()));
		}
		m_movieFrames.clear();
		m_timerCameraPath->start(10);
	}

	void Camera::animateCameraPath()
	{
		if (m_travelingCameraPath) {
			const std::tuple<float, glm::vec3, glm::quat>& current = m_pathIterations[m_currentStepPath];
			m_distanceOrtho = std::get<0>(current);
			m_stateCamera.m_translationModel = std::get<1>(current);
			m_stateCamera.m_rotationSum = std::get<2>(current);
		}
		else {
			glm::vec3 axis = glm::vec3(0, 1, 0);
			m_stateCamera.m_rotation = glm::angleAxis(m_angleRotation, axis);
			// Reverse so scene moves with cursor and not away due to camera model.
			m_stateCamera.m_rotation = glm::inverse(m_stateCamera.m_rotation);
			m_stateCamera.m_rotationSum *= m_stateCamera.m_rotation; // Accumulate quaternions.
		}
		updateCameraEyeUp(true, true);
		makeCurrent();
		recalcModelView();
		if (m_saveImagesCameraPath)
			drawElements(m_offscreenFBO);
		else
			drawElements();
		repaint();
		cptAnimation++;
		m_currentStepPath++;

		if (m_currentStepPath >= m_pathIterations.size() && m_timerCameraPath->isActive()) {
			m_timerCameraPath->stop();

			if (m_movieFrames.size() > 24)
				emit(askForMovieCreation());
		}
		/*if (m_travelingCameraPath) {
			m_distanceOrtho += m_stepDistanceCameraPath;

			auto stepTranslation = (m_statesPath[1].m_translationModel - m_statesPath[0].m_translationModel) / (float)m_nbImagesCameraPath;
			m_stateCamera.m_translationModel = m_stateCamera.m_translationModel + stepTranslation;

			m_stateCamera.m_rotationSum = glm::mix(m_statesPath[0].m_rotationSum, m_statesPath[1].m_rotationSum, m_currentStepPath);
			//m_stateCamera.m_matrixView = glm::mix(m_statesPath[0].m_matrixView, m_statesPath[1].m_matrixView, m_currentStepPath);
			updateCameraEyeUp(true, true);
		}
		else {
			glm::vec3 axis = glm::vec3(0, 1, 0);
			m_stateCamera.m_rotation = glm::angleAxis(m_angleRotation, axis);
			// Reverse so scene moves with cursor and not away due to camera model.
			m_stateCamera.m_rotation = glm::inverse(m_stateCamera.m_rotation);
			m_stateCamera.m_rotationSum *= m_stateCamera.m_rotation; // Accumulate quaternions.
			updateCameraEyeUp(true, true);
		}
		m_currentStepPath += m_stepPath;

		makeCurrent();
		recalcModelView();
		if(m_saveImagesCameraPath)
			drawElements(m_offscreenFBO);
		else
			drawElements();
		repaint();
		cptAnimation++;

		if(m_currentStepPath > 1.f && m_timerCameraPath->isActive())
			m_timerCameraPath->stop();*/
	}
}

