/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListMultiObjectDisplayCommand.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef ObjectListMultiObjectDisplayCommand_h__
#define ObjectListMultiObjectDisplayCommand_h__

#include <QtCore/QString>

#include <General/Vec3.hpp>
#include <General/Vec4.hpp>
#include <glm/mat4x4.hpp>
#include <OpenGL/BasicDisplayCommand.hpp>
#include <OpenGL/GLBuffer.hpp>
#include <OpenGL/MultiPrimitiveData.hpp>

class MyMultipleObject;
class ObjectListDisplayCommand;

class ObjectListMultiObjectDisplayCommand : public poca::opengl::BasicDisplayCommand {
public:
	ObjectListMultiObjectDisplayCommand(MyMultipleObject*);
	ObjectListMultiObjectDisplayCommand(const ObjectListMultiObjectDisplayCommand&);
	~ObjectListMultiObjectDisplayCommand();

	void execute(poca::core::CommandInfo*);
	void execute(poca::core::CommandInfo*, const poca::core::CommandExecutionContext&, poca::core::CommandExecutionResult&);
	poca::core::Command* copy();

	void freeGPUMemory();

protected:
	bool canBatch() const;
	bool rebuild();
	bool updateFeatureBuffers();
	ObjectListDisplayCommand* referenceDisplayCommand() const;
	void display(poca::opengl::Camera*, const bool, const bool, poca::core::CommandExecutionResult&);
	void drawElements(poca::opengl::Camera*, const bool, ObjectListDisplayCommand*);
	void drawPicking(poca::opengl::Camera*, ObjectListDisplayCommand*);
	QString getInfosTriangle(const int) const;
	void generateBoundingBoxSelection(const int);

protected:
	MyMultipleObject* m_object;
	bool m_hasOutlinePoints, m_has2DOutlines, m_is3D;
	GLuint m_textureLutID;
	GLfloat m_minOriginalFeature, m_maxOriginalFeature, m_actualValueFeature;

	poca::opengl::PointSingleGLBuffer<poca::core::Vec3mf> m_pointBuffer, m_outlinePointBuffer;
	poca::opengl::PointSingleGLBuffer<float> m_idLocsBuffer;
	poca::opengl::FeatureSingleGLBuffer<float> m_locsFeatureBuffer, m_outlineLocsFeatureBuffer;

	poca::opengl::TriangleSingleGLBuffer<poca::core::Vec3mf> m_triangleBuffer, m_triangleNormalBuffer;
	poca::opengl::TriangleSingleGLBuffer<float> m_idBuffer, m_triangleFeatureBuffer;

	poca::opengl::LineSingleGLBuffer<poca::core::Vec3mf> m_lineBuffer;
	poca::opengl::LineSingleGLBuffer<float> m_lineFeatureBuffer;
	poca::opengl::LineSingleGLBuffer<poca::core::Vec3mf> m_skeletonBuffer, m_linksBuffer;
	poca::opengl::LineSingleGLBuffer<poca::core::Color4D> m_colorSkeletonBuffer, m_colorLinksBuffer;
	poca::opengl::LineSingleGLBuffer<poca::core::Vec3mf> m_boundingBoxSelection;
	poca::opengl::QuadSingleGLBuffer<glm::mat4> m_ellipsoidTransformBuffer;
	poca::opengl::FeatureSingleGLBuffer<float> m_ellipsoidFeatureBuffer;

	bool m_hasEllipsoids;

	std::vector<poca::opengl::PickMappingEntry> m_pointPickMap, m_trianglePickMap;
};

#endif
