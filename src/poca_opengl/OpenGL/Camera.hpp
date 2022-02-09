/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Camera.hpp
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

#ifndef Camera_hpp__
#define Camera_hpp__

#include <QtWidgets/QOpenGLWidget>
#include <QtCore/QTimer>

#include <ctime>

#include <glm/glm.hpp>
#define GLM_FORCE_RADIANS
#include <glm/gtc/quaternion.hpp>

#include <Interfaces/MyObjectInterface.hpp>
#include <DesignPatterns/Observer.hpp>
#include <General/Vec2.hpp>
#include <General/Vec4.hpp>
#include <OpenGL/GLBuffer.hpp>

#include "../Interfaces/CameraInterface.hpp"
#include "SsaoShader.hpp"

class QOpenGLFramebufferObject;

namespace glm {
	void to_json(nlohmann::json& j, const glm::mat4& m);

	void from_json(const nlohmann::json& j, glm::mat4& m);

	void to_json(nlohmann::json& j, const glm::vec3& P);

	void from_json(const nlohmann::json& j, glm::vec3& P);

	void to_json(nlohmann::json& j, const glm::quat& q);

	void from_json(const nlohmann::json& j, glm::quat& q);
}

namespace poca::core {
	class ROIInterface;
}

namespace poca::opengl {

	class Shader;
	class TextDisplayer;

	class StateCamera {
	public:
		glm::mat4 m_matrixView;
		glm::quat m_rotationSum, m_rotation;
		glm::vec3 m_center;
		glm::vec3 m_eye;
		glm::mat4 m_matrix;
		glm::vec3 m_up;
	};

	class InfosObjectImages {
	public:
		InfosObjectImages();
		InfosObjectImages(QOpenGLFramebufferObject*, const poca::core::Vec4mf&, const poca::core::BoundingBox&, const StateCamera&, const int);
		~InfosObjectImages();

		void set(QOpenGLFramebufferObject*, const poca::core::Vec4mf&, const poca::core::BoundingBox&, const StateCamera&, const int);
		const bool inside(const int, const int) const;

		QOpenGLFramebufferObject* m_fbo;
		poca::core::Vec4mf m_rect;
		poca::core::BoundingBox m_bbox;
		StateCamera m_state;
		int m_id;
	};

	class Camera : public QOpenGLWidget, public CameraInterface, public poca::core::Observer {
		Q_OBJECT

	public:
		enum ScaleBarPosition { BottomLeft = 0, BottomRight = 1, TopLeft = 2, TopRight = 3, MiddleBottom = 4, MiddleTop = 5, MiddleLeft = 6, MiddleRight = 7 };
		enum InteractionMode
		{
			None = -1,
			Line2DRoiDefinition = 0,
			Circle2DRoiDefinition = 1,
			Polyline2DRoiDefinition = 2,
			Square2DRoiDefinition = 3,
			Triangle2DRoiDefinition = 4,
			Ellipse2DRoiDefinition = 5,
			Sphere3DRoiDefinition = 6,
			ModifyRoi = 7,
			Crop = 8
		};
		enum PlaneType
		{
			Plane_XY = 0,
			Plane_XZ = 1,
			Plane_YZ = 2
		};

		Camera(poca::core::MyObjectInterface*, const size_t, QWidget* = 0, Qt::WindowFlags = 0);
		~Camera();

		virtual const glm::mat4& getProjectionMatrix() const { return m_matrixProjection; }
		virtual const glm::mat4& getViewMatrix() const { return m_stateCamera.m_matrixView; }
		virtual const glm::mat4& getModelMatrix() const { return m_matrixModel; }
		virtual const glm::mat4& getTranslationMatrix() const { return m_translationMatrix; }
		virtual const glm::mat4& getRotationMatrix() const { return m_rotationMatrix; }

		virtual void setTranslationMatrix(const glm::mat4 _mat) { m_translationMatrix = _mat; }

		virtual const glm::vec4& getClipPlaneX() const { return m_clip[0]; }
		virtual const glm::vec4& getClipPlaneY() const { return m_clip[1]; }
		virtual const glm::vec4& getClipPlaneZ() const { return m_clip[2]; }
		virtual const glm::vec4& getClipPlaneW() const { return m_clip[3]; }
		virtual const glm::vec4& getClipPlaneH() const { return m_clip[4]; }
		virtual const glm::vec4& getClipPlaneT() const { return m_clip[5]; }

		virtual const glm::uvec4& getViewport() const { return m_viewport; }

		virtual void resizeWindow(const int, const int, const int, const int);
		virtual std::array<int, 2> sizeHintInterface() const;
		virtual QSize sizeHint() const;

		virtual poca::core::MyObjectInterface* getObject() { return m_object; }
		virtual void update() { QOpenGLWidget::update(); }
		virtual void makeCurrent() { QOpenGLWidget::makeCurrent(); }
		virtual void setDeleteObject(const bool _val) { m_deleteObject = _val; }

		const poca::core::BoundingBox& getCurrentCrop() const { return m_currentCrop; }
		void setCurrentCrop(const poca::core::BoundingBox& _crop) { m_currentCrop = _crop; }

		void update(poca::core::SubjectInterface*, const poca::core::CommandInfo&);

		Shader* getShader(const std::string&);
		void zoomToBoundingBox(const poca::core::BoundingBox&, const bool = true);

		inline void toggleBoundingBoxDisplay() { m_displayBoundingBox = !m_displayBoundingBox; }
		inline void toggleGridDisplay() { m_displayGrid = !m_displayGrid; }
		void toggleFontDisplay();

		int getWidth() const { return this->width(); }
		int getHeight() const { return this->height(); }

		void setCameraInteraction(const int);
		const int getCameraInteraction() const { return m_currentInteractionMode; }
		const bool cullFaceActivated() const { return m_cullFace; }
		const bool polygonFilled() const { return m_fillPolygon; }
		const StateCamera& getStateCamera() const { return m_stateCamera; }
		StateCamera& getStateCamera() { return m_stateCamera; }

		const glm::mat4& getMatrix();
		const float* getMatrixFlat();
		const glm::vec3& getCenter();
		const glm::vec3& getEye();
		const glm::vec3& getUp();
		const glm::quat getRotationSum() const { return m_stateCamera.m_rotationSum; }
		void reset();
		void setCenter(float x, float y, float z);
		void setCenter(const glm::vec3& c);
		void setEye(float x, float y, float z);
		void setEye(const glm::vec3& e);
		void setUp(float x, float y, float z);
		void setUp(const glm::vec3& u);
		void updateCamera();
		void resetProjection();
		void displayBoundingBox(const float, const float);
		void displayGrid();

		glm::vec3 getWorldCoordinates(const glm::vec2&);
		glm::vec2 worldToScreenCoordinates(const glm::vec3&) const;

		void enableClippingPlanes();
		void disableClippingPlanes();

		void fixPlane(const int, const bool);

		inline const float getDistanceOrtho() const { return m_distanceOrtho; }
		inline const float getOriginalDistanceOrtho() const { return m_originalDistanceOrtho; }
		inline void setDistanceOrtho(const float _val) { m_distanceOrtho = _val; }
		inline void setOriginalDistanceOrtho(const float _val) { m_originalDistanceOrtho = _val; }


		template < class T, class M >
		void drawSimpleShader(const GLuint, const SingleGLBuffer<T>&, const SingleGLBuffer<M>&, const GLfloat, const GLfloat, const GLfloat = 1.f, const SingleGLBuffer<T>& = SingleGLBuffer<T>());

		template < class T, class M >
		void drawSimpleShader(const SingleGLBuffer<T>&, const SingleGLBuffer<M>&, const GLfloat = 1.f, const SingleGLBuffer<T> & = SingleGLBuffer<T>());

		template < class T, class M >
		void drawPickingShader(const SingleGLBuffer<T>&, const SingleGLBuffer<M>&, const SingleGLBuffer<M>&, const GLfloat);

		template < class T, class M >
		void drawPickingShader(const SingleGLBuffer<T>&, const SingleGLBuffer<M>&);

		template < class T >
		void drawUniformShader(const SingleGLBuffer<T>&, const poca::core::Color4D&, const SingleGLBuffer<T> & = SingleGLBuffer<T>());

		template < class T, class M >
		void drawLineShader(const SingleGLBuffer<T>&, const SingleGLBuffer<M> &, const poca::core::Color4D&, const SingleGLBuffer<T> & = SingleGLBuffer<T>(), const GLfloat = -FLT_MAX);

		template < class T, class M >
		void drawLineShader(const GLuint, const SingleGLBuffer<T>&, const SingleGLBuffer<M>&, const SingleGLBuffer<T>&, const GLfloat, const GLfloat, const GLfloat = 1.f);
		
		template < class T, class M >
		void drawSphereRendering(const GLuint, const SingleGLBuffer<T>&, const SingleGLBuffer<M>&, const GLfloat, const GLfloat, const uint32_t = 5, const bool = false);

		template < class T, class M >
		void drawSphereRendering(const GLuint, const SingleGLBuffer<T>&, const SingleGLBuffer<T>&, const SingleGLBuffer<M>&, const GLfloat, const GLfloat, const uint32_t = 5, const bool = false);

		template < class T, class M >
		void draw2DGaussianRendering(const GLuint, const SingleGLBuffer<T>&, const SingleGLBuffer<T>&, const SingleGLBuffer<M>&, const GLfloat, const GLfloat, const float = 0.1f, const uint32_t = 5, const bool = false);

		template < class T >
		void drawUniformShader(const GLBuffer<T>&, const poca::core::Color4D&);

		template < class T, class M >
		void drawSimpleShader(const GLuint, const GLBuffer<T>&, const GLBuffer<M>&, const GLfloat, const GLfloat, const GLfloat = 1.f);

		template < class T, class M >
		void drawSimpleShaderWithColor(const GLBuffer<T>&, const GLBuffer<M>&);
			
		template < class T, class M >
		void drawPickingShader(const GLBuffer<T>&, const GLBuffer<M>&, const GLBuffer<M>&, const GLfloat);

		template < class T, class M >
		void drawHeatmap(const SingleGLBuffer<T>&, const SingleGLBuffer<M>&, const GLfloat, const GLfloat, const bool, const GLfloat);

		void drawTexture(const GLuint, const bool = false);
		void drawTextureFBO(const GLuint, const GLuint);

		const bool isAntialiasActivated() const { return m_activateAntialias; }

		void recalcModelView(void);

	protected:
		void initializeGL();
		void paintGL();
		void drawElements(QOpenGLFramebufferObject * = NULL);
		void keyPressEvent(QKeyEvent*);
		void mousePressEvent(QMouseEvent*);
		void mouseMoveEvent(QMouseEvent*);
		void mouseReleaseEvent(QMouseEvent*);
		void mouseDoubleClickEvent(QMouseEvent*);
		void wheelEvent(QWheelEvent*);
		void resizeEvent(QResizeEvent*);
		const bool testWhichFaceFrontCamera();
		void recomputeFrame(const poca::core::BoundingBox&);
		glm::vec2 worldToScreenCoordinates(const glm::mat4&, const glm::mat4&, const glm::mat4&, const glm::uvec4&, const glm::vec3&) const;

	protected:
		void computeRotation();
		void updateCameraEyeUp(bool eye, bool up);
		void computeCameraEye(glm::vec3& eye);
		void computeCameraUp(glm::vec3& up);
		void computePointOnSphere(const glm::vec2& point,
			glm::vec3& result);
		void computeRotationBetweenVectors(const glm::vec3& start,
			const glm::vec3& stop,
			glm::quat& result);
		void setClickPoint(double x, double y);
		void determineLeftRightUpFaces(size_t);
		void drawOffscreen();
		void renderQuad(const bool = false) const;
		void renderRoundedBoxShadow(const float, const float, const float, const float, const float, const float, const float, const float, const float, const float);

		void createArrowsFrame();
		void generateDirectionalArrow(const poca::core::Vec3mf&, const poca::core::Vec3mf&, const poca::core::Vec3mf&, const float, const float, std::vector <poca::core::Vec3mf>&);
		void recomputeLegendThumbnailFrame();
		void displayArrowsFrame();
		void toggleSrcBlendingFactor();
		void toggleDstBlendingFactor();
		void drawSSAO(QOpenGLFramebufferObject* = NULL);

	protected slots:
		void animateRotation();

	signals:
		void clickInsideWindow();

	protected:
		size_t m_dimension;
		glm::mat4 m_matrixProjection, m_matrixModel, m_translationMatrix, m_rotationMatrix;
		glm::uvec4 m_viewport;
		glm::vec4 m_clip[6];
		float m_precX, m_precY, m_distanceOrtho, m_originalDistanceOrtho;
		float m_translationX, m_translationY, m_translationZ;
		bool m_scaling, m_buttonOn, m_leftButtonOn, m_middleButtonOn, m_rightButtonOn, m_displayBoundingBox, m_displayGrid;
		bool m_deleteObject, m_alreadyInitialized;
		bool m_cullFace, m_fillPolygon;

		StateCamera m_stateCamera;

		poca::core::MyObjectInterface* m_object;
		QStringList m_infoPicking;

		poca::core::BoundingBox m_currentCrop;
		poca::core::Vec6mb m_facingDirections;
		float m_nbMainGrid, m_nbIntermediateGrid;
		bool m_backBBoxFaces;
		float m_step;
		poca::core::Vec3mui m_nbStepsDim;
		unsigned int m_nbElementsGrid, m_nbElementsFacingBBox, m_nbElementsNotFacingBBox;

		poca::opengl::TriangleSingleGLBuffer <poca::core::Vec3mf> m_faceGridBuffer;
		poca::opengl::LineSingleGLBuffer <poca::core::Vec3mf> m_lineGridBuffer, m_lineBboxBuffer;

		poca::opengl::PointGLBuffer <poca::core::Vec3mf> m_simplePointBuffer, m_debugPointBuffer;
		poca::opengl::LineGLBuffer <poca::core::Vec3mf> m_cropBuffer;
		std::vector <std::pair<poca::core::Vec3mf, std::string>> m_frameTexts;

		poca::opengl::PointSingleGLBuffer <poca::core::Vec3mf> m_centerSquaresBuffer, m_axisSquaresBuffer;

		float m_speed;
		glm::vec2 m_prevClickPoint, m_clickPoint;
		glm::vec3 m_startVector, m_stopVector, m_translation;

		std::map <std::string, Shader*> m_shaders;
		std::vector <InfosObjectImages*> m_infoObjects;
		int m_sizePatch, m_insidePatchId;

		bool m_undoPossible, m_resetedProj;
		float m_distanceOrthoSaved, m_multAnimation;
		glm::mat4 m_matrixModelSaved, m_matrixViewSaved;

		glm::vec3 m_translationModel;

		size_t m_leftFace, m_rightFace, m_upFace;

		QTimer* m_timer;

		glm::uvec4 m_viewportThumbnailFrame;
		glm::mat4 m_matrixProjectionThumbnailFrame;
		uint32_t m_nbArrowsFrame;
		TriangleGLBuffer <poca::core::Vec3mf> m_vertexArrowBuffer;
		std::vector <std::pair<poca::core::Vec3mf, std::string>> m_frameTextsThumbnail;

		QOpenGLFramebufferObject* m_offscreenFBO;
		PointGLBuffer <poca::core::Vec3mf> m_quadVertBuffer, m_quadVertBufferFlippedH;
		FeatureGLBuffer <poca::core::Vec2mf> m_quadUvBuffer, m_quadUvBufferFlippedH;

		TextDisplayer* m_texDisplayer;
		TriangleStripGLBuffer <poca::core::Vec2mf> m_roundedRectBuffer;

		int m_currentInteractionMode;
		poca::core::ROIInterface* m_ROI;
		glm::vec2 m_tmp;

		GLenum m_sourceFactorBlending, m_destFactorBlending;
		uint32_t m_curIndexSource, m_curIndexDest;
		bool m_activateAntialias, m_preventRotation;

		std::vector <poca::core::Vec3mf> m_pickedPoints;
		poca::core::Vec3mf m_cropPointBegin, m_cropPointEnd;
		int m_cropPlane;

		SsaoShader m_ssaoShader;
	};

	template < class T, class M >
	void Camera::drawSimpleShader(const GLuint _textureLutID, const SingleGLBuffer<T>& _bufferVertex, const SingleGLBuffer<M>& _bufferFeature, const GLfloat _minF, const GLfloat _maxF, const GLfloat _alpha, const SingleGLBuffer<T>& _bufferNormal)
	{
		if (_bufferVertex.empty() || _bufferFeature.empty()) return;
		poca::opengl::Shader* shader = getShader("simpleShader");

		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setInt("lutTexture", 0);
		shader->setFloat("minFeatureValue", _minF);
		shader->setFloat("maxFeatureValue", _maxF);
		shader->setFloat("alpha", _alpha);
		shader->setBool("useSpecialColors", false);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());

		glm::vec3 frwd = getEye();
		frwd = glm::normalize(frwd);
		glm::vec3 orientation = getRotationSum() * glm::vec3(0.f, 0.f, 1.f);
		orientation = glm::normalize(orientation);
		shader->setVec3("cameraForward", orientation);
		shader->setBool("activatedCulling", cullFaceActivated() && !_bufferNormal.empty());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_1D, _textureLutID);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(2);
		_bufferVertex.bindBuffer(0);
		_bufferFeature.bindBuffer(2);
		if (!_bufferNormal.empty()) {
			glEnableVertexAttribArray(4);
			_bufferNormal.bindBuffer(4);
		}
		if (_bufferVertex.getBufferIndices() != 0)
			glDrawElements(_bufferVertex.getMode(), _bufferVertex.getNbIndices(), GL_UNSIGNED_INT, (void*)0);
		else
			glDrawArrays(_bufferVertex.getMode(), 0, _bufferVertex.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(2);
		glDisableVertexAttribArray(4);

		glBindTexture(GL_TEXTURE_1D, 0); // Unbind any textures
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		shader->release();
		GL_CHECK_ERRORS();
	}

	template < class T, class M >
	void Camera::drawSimpleShader(const SingleGLBuffer<T>& _bufferVertex, const SingleGLBuffer<M>& _bufferColor, const GLfloat _alpha, const SingleGLBuffer<T>& _bufferNormal)
	{
		if (_bufferVertex.empty() || _bufferColor.empty()) return;
		poca::opengl::Shader* shader = getShader("simpleShader");

		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setFloat("alpha", _alpha);
		shader->setBool("useSpecialColors", true);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());

		glm::vec3 frwd = getEye();
		frwd = glm::normalize(frwd);
		glm::vec3 orientation = getRotationSum() * glm::vec3(0.f, 0.f, 1.f);
		orientation = glm::normalize(orientation);
		shader->setVec3("cameraForward", orientation);
		shader->setBool("activatedCulling", cullFaceActivated() && !_bufferNormal.empty());

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(3);
		_bufferVertex.bindBuffer(0);
		_bufferColor.bindBuffer(3);
		if (!_bufferNormal.empty()) {
			glEnableVertexAttribArray(4);
			_bufferNormal.bindBuffer(4);
		}
		if (_bufferVertex.getBufferIndices() != 0)
			glDrawElements(_bufferVertex.getMode(), _bufferVertex.getNbIndices(), GL_UNSIGNED_INT, (void*)0);
		else
			glDrawArrays(_bufferVertex.getMode(), 0, _bufferVertex.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(3);
		glDisableVertexAttribArray(4);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		shader->release();
		GL_CHECK_ERRORS();
	}

	template < class T, class M >
	void Camera::drawPickingShader(const SingleGLBuffer<T>& _bufferVertex, const SingleGLBuffer<M>& _bufferId, const SingleGLBuffer<M>& _bufferFeature, const GLfloat _minF)
	{
		if (_bufferVertex.empty() || _bufferId.empty() || _bufferFeature.empty()) return;
		poca::opengl::Shader* shader = getShader("pickShader");

		uint32_t sizeGL = 1;
		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(m_object);
		if (comObj) {
			if (comObj->hasParameter("pointSizeGL"))
				sizeGL = comObj->getParameter<uint32_t>("pointSizeGL");
		}

		glDisable(GL_POINT_SPRITE);
		glDisable(GL_PROGRAM_POINT_SIZE);
		glPointSize(sizeGL);

		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setBool("hasFeature", true);
		shader->setFloat("minFeatureValue", _minF);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		_bufferVertex.bindBuffer(0);
		_bufferId.bindBuffer(1);
		_bufferFeature.bindBuffer(2);
		if (_bufferVertex.getBufferIndices() != 0)
			glDrawElements(_bufferVertex.getMode(), _bufferVertex.getNbIndices(), GL_UNSIGNED_INT, (void*)0);
		else
			glDrawArrays(_bufferVertex.getMode(), 0, _bufferVertex.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);

		shader->release();
	}

	template < class T, class M >
	void Camera::drawPickingShader(const SingleGLBuffer<T>& _bufferVertex, const SingleGLBuffer<M>& _bufferId)
	{
		if (_bufferVertex.empty() || _bufferId.empty()) return;
		poca::opengl::Shader* shader = getShader("pickShader");

		uint32_t sizeGL = 1;
		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(m_object);
		if (comObj) {
			if (comObj->hasParameter("pointSizeGL"))
				sizeGL = comObj->getParameter<uint32_t>("pointSizeGL");
		}

		glDisable(GL_POINT_SPRITE);
		glDisable(GL_PROGRAM_POINT_SIZE);
		glPointSize(sizeGL);

		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setBool("hasFeature", false);
		shader->setFloat("minFeatureValue", 0.f);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		_bufferVertex.bindBuffer(0);
		_bufferId.bindBuffer(1);
		if (_bufferVertex.getBufferIndices() != 0)
			glDrawElements(_bufferVertex.getMode(), _bufferVertex.getNbIndices(), GL_UNSIGNED_INT, (void*)0);
		else
			glDrawArrays(_bufferVertex.getMode(), 0, _bufferVertex.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);

		shader->release();
		GL_CHECK_ERRORS();
	}

	template < class T, class M >
	void Camera::drawSphereRendering(const GLuint _textureLutID, const SingleGLBuffer<T>& _bufferVertex, const SingleGLBuffer<M>& _bufferFeature, const GLfloat _minF, const GLfloat _maxF, const uint32_t _radius, const bool _ssao)
	{
		if (_bufferVertex.empty() || _bufferFeature.empty()) return;
		poca::opengl::Shader* shader = _ssao ? getShader("geometrySSAO") : getShader("sphereRenderingShader");

		if (!_ssao) {
			if (_bufferVertex.getDim() == 3)
				glEnable(GL_DEPTH_TEST);
			else
				glDisable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glEnable(GL_POINT_SPRITE);
			glEnable(GL_PROGRAM_POINT_SIZE);
		}
		else {
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LESS);
			glDisable(GL_BLEND);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glDisable(GL_CULL_FACE);
		}

		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setMat4("view", view);
		shader->setMat4("model", model);
		shader->setMat4("projection", proj);
		shader->setInt("lutTexture", 0);
		shader->setFloat("minFeatureValue", _minF);
		shader->setFloat("maxFeatureValue", _maxF);
		shader->setBool("useSpecialColors", false);
		shader->setFloat("radius", _radius);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());
		shader->setFloat("nbPoints", _bufferVertex.getNbElements());
		shader->setVec3("light_position", getEye());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_1D, _textureLutID);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(2);
		_bufferVertex.bindBuffer(0);
		_bufferFeature.bindBuffer(2);
		if (_bufferVertex.getBufferIndices() != 0)
			glDrawElements(_bufferVertex.getMode(), _bufferVertex.getNbIndices(), GL_UNSIGNED_INT, (void*)0);
		else
			glDrawArrays(_bufferVertex.getMode(), 0, _bufferVertex.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(2);
		GL_CHECK_ERRORS();

		glBindTexture(GL_TEXTURE_1D, 0); // Unbind any textures
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		shader->release();
		GL_CHECK_ERRORS();
	}

	template < class T, class M >
	void Camera::drawSphereRendering(const GLuint _textureLutID, const SingleGLBuffer<T>& _bufferVertex, const SingleGLBuffer<T>& _bufferNormal, const SingleGLBuffer<M>& _bufferFeature, const GLfloat _minF, const GLfloat _maxF, const uint32_t _radius, const bool _ssao)
	{
		if (_bufferVertex.empty() || _bufferFeature.empty()) return;
		poca::opengl::Shader* shader = _ssao ? getShader("geometrySSAO") : getShader("sphereRenderingShader");

		if (!_ssao) {
			if (_bufferVertex.getDim() == 3)
				glEnable(GL_DEPTH_TEST);
			else
				glDisable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glEnable(GL_POINT_SPRITE);
			glEnable(GL_PROGRAM_POINT_SIZE);
		}
		else {
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LESS);
			glDisable(GL_BLEND);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glDisable(GL_CULL_FACE);
		}

		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setMat4("view", view);
		shader->setMat4("model", model);
		shader->setMat4("projection", proj);
		shader->setInt("lutTexture", 0);
		shader->setFloat("minFeatureValue", _minF);
		shader->setFloat("maxFeatureValue", _maxF);
		shader->setBool("useSpecialColors", false);
		shader->setFloat("radius", _radius);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());
		shader->setFloat("nbPoints", _bufferVertex.getNbElements());
		shader->setVec3("light_position", getEye());
		shader->setBool("activatedCulling", cullFaceActivated());
		glm::vec3 frwd = getEye();
		frwd = glm::normalize(frwd);
		glm::vec3 orientation = getRotationSum() * glm::vec3(0.f, 0.f, 1.f);
		orientation = glm::normalize(orientation);
		shader->setVec3("cameraForward", orientation);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_1D, _textureLutID);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		_bufferVertex.bindBuffer(0);
		_bufferNormal.bindBuffer(1);
		_bufferFeature.bindBuffer(2);
		if (_bufferVertex.getBufferIndices() != 0)
			glDrawElements(_bufferVertex.getMode(), _bufferVertex.getNbIndices(), GL_UNSIGNED_INT, (void*)0);
		else
			glDrawArrays(_bufferVertex.getMode(), 0, _bufferVertex.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);
		GL_CHECK_ERRORS();

		glBindTexture(GL_TEXTURE_1D, 0); // Unbind any textures
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		shader->release();
		GL_CHECK_ERRORS();
	}

	template < class T, class M >
	void Camera::draw2DGaussianRendering(const GLuint _textureLutID, const SingleGLBuffer<T>& _bufferVertex, const SingleGLBuffer<T>& _bufferSigmas, const SingleGLBuffer<M>& _bufferFeature, const GLfloat _minF, const GLfloat _maxF, const float _alpha, const uint32_t _pointSize, const bool _fixedSize)
	{
		if (_bufferVertex.empty() || _bufferFeature.empty()) return;
		poca::opengl::Shader* shader = getShader("2DGaussianRenderingShader");

		std::array <uint8_t, 4> colorBack{ 0, 0, 0, 255 };
		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(m_object);
		if (comObj) {
			if (comObj->hasParameter("colorBakground")) {
				std::array <unsigned char, 4> rgba = comObj->getParameter< std::array <unsigned char, 4>>("colorBakground");
				colorBack = { rgba[0], rgba[1], rgba[2], rgba[3] };
			}
		}

		glEnable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		if (colorBack[0] == 255 && colorBack[1] == 255 && colorBack[2] == 255)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		else
			glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		glBlendEquation(GL_FUNC_ADD);
		glDisable(GL_POINT_SPRITE);
		glDisable(GL_PROGRAM_POINT_SIZE);

		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setMat4("projection", proj);
		shader->setInt("lutTexture", 0);
		shader->setFloat("minFeatureValue", _minF);
		shader->setFloat("maxFeatureValue", _maxF);
		shader->setBool("useSpecialColors", false);
		shader->setFloat("radius", _pointSize);
		shader->setBool("fixedRadius", _fixedSize);
		shader->setFloat("alpha", _alpha);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_1D, _textureLutID);

		glEnableVertexAttribArray(0);
		if (!_bufferSigmas.empty())
			glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		_bufferVertex.bindBuffer(0);
		if (!_bufferSigmas.empty())
			_bufferSigmas.bindBuffer(1);
		_bufferFeature.bindBuffer(2);
		if (_bufferVertex.getBufferIndices() != 0)
			glDrawElements(_bufferVertex.getMode(), _bufferVertex.getNbIndices(), GL_UNSIGNED_INT, (void*)0);
		else
			glDrawArrays(_bufferVertex.getMode(), 0, _bufferVertex.getNbElements());
		glDisableVertexAttribArray(0);
		if (!_bufferSigmas.empty())
			glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);
		GL_CHECK_ERRORS();

		glBindTexture(GL_TEXTURE_1D, 0); // Unbind any textures
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		shader->release();
		GL_CHECK_ERRORS();
	}
	
	template < class T >
	void Camera::drawUniformShader(const SingleGLBuffer<T>& _buffer, const poca::core::Color4D& _color, const SingleGLBuffer<T>& _bufferNormal)
	{
		if (_buffer.empty()) return;
		uint32_t sizeGL = 1;
		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(m_object);
		if (comObj) {
			if (comObj->hasParameter("pointSizeGL")) {
				sizeGL = comObj->getParameter<uint32_t>("pointSizeGL");
				glPointSize(sizeGL);
			}
		}
		
		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		Shader* shader = getShader("uniformColorShader");
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setVec4("singleColor", _color[0], _color[1], _color[2], _color[3]);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());

		glm::vec3 frwd = getEye();
		frwd = glm::normalize(frwd);
		glm::vec3 orientation = getRotationSum() * glm::vec3(0.f, 0.f, 1.f);
		orientation = glm::normalize(orientation);
		shader->setVec3("cameraForward", orientation);
		shader->setBool("activatedCulling", cullFaceActivated() && !_bufferNormal.empty());

		glEnableVertexAttribArray(0);
		_buffer.bindBuffer(0);
		if (!_bufferNormal.empty()) {
			glEnableVertexAttribArray(4);
			_bufferNormal.bindBuffer(4);
		}
		if (_buffer.getBufferIndices() != 0)
			glDrawElements(_buffer.getMode(), _buffer.getNbIndices(), GL_UNSIGNED_INT, (void*)0);
		else
			glDrawArrays(_buffer.getMode(), 0, _buffer.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(4);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		shader->release();
	}

	template < class T >
	void Camera::drawUniformShader(const GLBuffer<T>& _buffer, const poca::core::Color4D& _color)
	{
		if (_buffer.empty()) return;
		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		Shader* shader = getShader("uniformColorShader");
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setVec4("singleColor", _color[0], _color[1], _color[2], _color[3]);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());

		glEnableVertexAttribArray(0);
		const std::vector <size_t>& sizeStridesLocs = _buffer.getSizeBuffers();
		for (size_t chunk = 0; chunk < _buffer.getNbBuffers(); chunk++) {
			_buffer.bindBuffer(chunk, 0);
			glDrawArrays(_buffer.getMode(), 0, (GLsizei)sizeStridesLocs[chunk]); // Starting from vertex 0; 3 vertices total -> 1 triangle
		}
		glDisableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		shader->release();
	}

	template < class T, class M >
	void Camera::drawLineShader(const SingleGLBuffer<T>& _buffer, const SingleGLBuffer<M>& _feature, const poca::core::Color4D& _color, const SingleGLBuffer<T> & _normals, const GLfloat _minF)
	{
		if (_buffer.empty()) return;
		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		uint32_t thickness = 5;
		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(getObject());
		if (comObj)
			if (comObj->hasParameter("lineWidthGL"))
				thickness = comObj->getParameter<uint32_t>("lineWidthGL");
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		poca::opengl::Shader* shader = getShader("line2DShader");
		shader->use();

		glm::vec3 frwd = getEye();
		frwd = glm::normalize(frwd);
		glm::vec3 orientation = getRotationSum() * glm::vec3(0.f, 0.f, 1.f);
		orientation = glm::normalize(orientation);
		shader->setVec3("cameraForward", orientation);

		shader->setMat4("MVP", proj * view * model);
		shader->setUVec4("viewport", getViewport());
		shader->setVec2("resolution", width(), height());
		shader->setFloat("thickness", thickness * 2.f);
		shader->setFloat("antialias", 1.f);
		shader->setVec4("singleColor", _color[0], _color[1], _color[2], _color[3]);
		shader->setBool("activatedCulling", cullFaceActivated() && !_normals.empty());
		shader->setBool("useSingleColor", true);
		shader->setFloat("minFeatureValue", _minF);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());
		glEnableVertexAttribArray(0);
		_buffer.bindBuffer(0);
		if (!_normals.empty()) {
			glEnableVertexAttribArray(1);
			_normals.bindBuffer(1);
		}
		if (!_feature.empty()) {
			glEnableVertexAttribArray(2);
			_feature.bindBuffer(2);
		}
		glDrawArrays(_buffer.getMode(), 0, _buffer.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);
		shader->release();
	}

	template < class T, class M >
	void Camera::drawLineShader(const GLuint _textureLutID, const SingleGLBuffer<T>& _bufferVertex, const SingleGLBuffer<M>& _bufferFeature, const SingleGLBuffer<T>& _bufferNormals, const GLfloat _minF, const GLfloat _maxF, const GLfloat _alpha)
	{
		if (_bufferVertex.empty() || _bufferFeature.empty()) return;
		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		uint32_t thickness = 5;
		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(getObject());
		if (comObj)
			if (comObj->hasParameter("lineWidthGL"))
				thickness = comObj->getParameter<uint32_t>("lineWidthGL");
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		poca::opengl::Shader* shader = getShader("line2DShader");
		shader->use();

		glm::vec3 frwd = getEye();
		frwd = glm::normalize(frwd);
		glm::vec3 orientation = getRotationSum() * glm::vec3(0.f, 0.f, 1.f);
		orientation = glm::normalize(orientation);
		shader->setVec3("cameraForward", orientation);

		shader->setMat4("MVP", proj * view * model);
		shader->setUVec4("viewport", getViewport());
		shader->setVec2("resolution", width(), height());
		shader->setFloat("thickness", thickness * 2.f);
		shader->setFloat("antialias", 1.f);
		shader->setBool("activatedCulling", cullFaceActivated() && !_bufferNormals.empty());
		shader->setBool("useSingleColor", false);
		shader->setInt("lutTexture", 0);
		shader->setFloat("minFeatureValue", _minF);
		shader->setFloat("maxFeatureValue", _maxF);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_1D, _textureLutID);
		
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(2);
		_bufferVertex.bindBuffer(0);
		_bufferFeature.bindBuffer(2);

		if (!_bufferNormals.empty()) {
			glEnableVertexAttribArray(1);
			_bufferNormals.bindBuffer(1);
		}
		if (_bufferVertex.getBufferIndices() != 0)
			glDrawElements(_bufferVertex.getMode(), _bufferVertex.getNbIndices(), GL_UNSIGNED_INT, (void*)0);
		else
			glDrawArrays(_bufferVertex.getMode(), 0, _bufferVertex.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);
		glBindTexture(GL_TEXTURE_1D, 0); // Unbind any textures
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		GL_CHECK_ERRORS();
		shader->release();
	}

	template < class T, class M >
	void Camera::drawSimpleShader(const GLuint _textureLutID, const GLBuffer<T>& _bufferVertex, const GLBuffer<M>& _bufferFeature, const GLfloat _minF, const GLfloat _maxF, const GLfloat _alpha)
	{
		if (_bufferVertex.empty() || _bufferFeature.empty()) return;
		uint32_t sizeGL = 1, widthLineGL = 1;
		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(m_object);
		if (comObj) {
			if (comObj->hasParameter("pointSizeGL"))
				sizeGL = comObj->getParameter<uint32_t>("pointSizeGL");
			if (comObj->hasParameter("lineWidthGL"))
				widthLineGL = comObj->getParameter<uint32_t>("lineWidthGL");
		}

		glPointSize(sizeGL);
		glLineWidth(widthLineGL);

		poca::opengl::Shader* shader = getShader("simpleShader");

		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setInt("lutTexture", 0);
		shader->setFloat("minFeatureValue", _minF);
		shader->setFloat("maxFeatureValue", _maxF);
		shader->setFloat("alpha", _alpha);
		shader->setBool("useSpecialColors", false);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_1D, _textureLutID);

		const std::vector <size_t>& sizeStrides = _bufferVertex.getSizeBuffers();
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(2);
		for (unsigned int chunk = 0; chunk < _bufferVertex.getNbBuffers(); chunk++) {
			// 1rst attribute buffer : vertices
			_bufferVertex.bindBuffer(chunk, 0);
			_bufferFeature.bindBuffer(chunk, 2);
			glDrawArrays(_bufferVertex.getMode(), 0, (GLsizei)sizeStrides[chunk]); // Starting from vertex 0; 3 vertices total -> 1 triangle
		}
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(2);

		glBindTexture(GL_TEXTURE_1D, 0); // Unbind any textures
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		shader->release();
		GL_CHECK_ERRORS();
	}

	template < class T, class M >
	void Camera::drawSimpleShaderWithColor(const GLBuffer<T>& _bufferVertex, const GLBuffer<M>& _bufferColor)
	{
		if (_bufferVertex.empty() || _bufferColor.empty()) return;
		uint32_t sizeGL = 1, widthLineGL = 1;
		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(m_object);
		if (comObj) {
			if (comObj->hasParameter("pointSizeGL"))
				sizeGL = comObj->getParameter<uint32_t>("pointSizeGL");
			if (comObj->hasParameter("lineWidthGL"))
				widthLineGL = comObj->getParameter<uint32_t>("lineWidthGL");
		}

		glPointSize(sizeGL);
		glLineWidth(widthLineGL);
		
		poca::opengl::Shader* shader = getShader("simpleShader");

		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setFloat("useSpecialColors", 1);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());

		const std::vector <size_t>& sizeStrides = _bufferVertex.getSizeBuffers();
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(3);
		for (unsigned int chunk = 0; chunk < _bufferVertex.getNbBuffers(); chunk++) {
			// 1rst attribute buffer : vertices
			_bufferVertex.bindBuffer(chunk, 0);
			_bufferColor.bindBuffer(chunk, 3);
			glDrawArrays(_bufferVertex.getMode(), 0, (GLsizei)sizeStrides[chunk]); // Starting from vertex 0; 3 vertices total -> 1 triangle
		}
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(3);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		shader->release();
		GL_CHECK_ERRORS();
	}

	template < class T, class M >
	void Camera::drawPickingShader(const GLBuffer<T>& _bufferVertex, const GLBuffer<M>& _bufferId, const GLBuffer<M>& _bufferFeature, const GLfloat _minF)
	{
		if (_bufferVertex.empty() || _bufferId.empty() || _bufferFeature.empty()) return;
		poca::opengl::Shader* shader = getShader("pickShader");

		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setBool("hasFeature", true);
		shader->setFloat("minFeatureValue", _minF);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());

		const std::vector <size_t>& sizeStrides = _bufferVertex.getSizeBuffers();
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		for (unsigned int chunk = 0; chunk < _bufferVertex.getNbBuffers(); chunk++) {
			// 1rst attribute buffer : vertices
			_bufferVertex.bindBuffer(chunk, 0);
			_bufferId.bindBuffer(chunk, 1);
			_bufferFeature.bindBuffer(chunk, 2);
			glDrawArrays(_bufferVertex.getMode(), 0, (GLsizei)sizeStrides[chunk]); // Starting from vertex 0; 3 vertices total -> 1 triangle
		}
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);

		shader->release();
	}

	template < class T, class M >
	void Camera::drawHeatmap(const SingleGLBuffer<T>& _bufferVertex, const SingleGLBuffer<M>& _selectBuffer, const GLfloat _intensity, const GLfloat _radius, const bool _screenRadius, const GLfloat _minX)
	{
		if (_bufferVertex.empty() || _selectBuffer.empty()) return;
		poca::opengl::Shader* shader = getShader("heatmapShader");

		float px = 2.f / (float)this->width(), py = 2.f / (float)this->height();
		const glm::mat4& proj = getProjectionMatrix(), & view = getViewMatrix(), & model = getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setMat4("projection", proj);
		shader->setVec4("clipPlaneX", getClipPlaneX());
		shader->setVec4("clipPlaneY", getClipPlaneY());
		shader->setVec4("clipPlaneZ", getClipPlaneZ());
		shader->setVec4("clipPlaneW", getClipPlaneW());
		shader->setVec4("clipPlaneH", getClipPlaneH());
		shader->setVec4("clipPlaneT", getClipPlaneT());
		shader->setFloat("u_intensity", _intensity);
		shader->setFloat("radius", _radius);
		shader->setBool("screenRadius", _screenRadius);
		shader->setVec2("pixelSize", px, py);// px, py);//NDC coordinates are from -1 to 1
		shader->setFloat("minX", _minX);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		_bufferVertex.bindBuffer(0);
		_selectBuffer.bindBuffer(1);
		if (_bufferVertex.getBufferIndices() != 0)
			glDrawElements(_bufferVertex.getMode(), _bufferVertex.getNbIndices(), GL_UNSIGNED_INT, (void*)0);
		else
			glDrawArrays(_bufferVertex.getMode(), 0, _bufferVertex.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);

		shader->release();
	}
}

#endif

